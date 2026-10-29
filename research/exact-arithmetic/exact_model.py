"""
Exact-arithmetic ("canonical spec") Qwen2 forward pass.

Every reduction is an exact integer sum; every other step is a single IEEE
fp32 op (mul/add/convert/round) or an integer op. The forward pass is
therefore a pure function of (weights, tokens, seed): any device, any matmul
algorithm, any chunking of the prompt must produce identical bits.

Tests (python exact_model.py):
  1. accuracy vs the HF fp32 model (W8 weights, A16 activations)
  2. CPU vs Apple GPU (different hardware AND different exact-matmul algorithms)
  3. token-by-token generation vs one-pass recomputation of the transcript
     ("regenerate on challenge"), greedy and sampled
  4. hash roots over every layer's hidden state for every token
"""

import hashlib
import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOG2E = 1.4426950408889634
ACT_W, QK_W, V_W, RES_W, WBITS = 16, 16, 16, 24, 8   # integer widths (incl. sign)
P_ATT, P_SIG, P_SAMPLE = 20, 24, 24                   # fixed-point exp output bits
FB, HB = 20, 12                                       # exp: fraction bits, LUT bits
A_FRAC, N_FRAC, S_FRAC = 12, 20, 24                   # fraction bits of divisions
LN2_LO = [round(l * math.log(2) * 2 ** (30 - FB)) for l in range(2 ** (FB - HB))]
# Input domain: row maxima clamped up to ROW_MIN, must stay below ROW_MAX, so all
# power-of-two scales (and sums/doubles of them) stay in the fp32 normal range.
ROW_MIN, ROW_MAX = 2.0**-40, 2.0**40


# ---- elementwise primitives (bit-level, identical on every IEEE device)


def exp2i(k):
    """2^k as fp32 for an integer tensor k in [-126, 127]."""
    return ((k + 127).int() << 23).view(torch.float32)


def frexp_exp(m):
    """frexp exponent of a positive normal fp32 tensor."""
    return (((m.view(torch.int32) >> 23) & 0xFF) - 126).long()


def qmax(w):
    return 2 ** (w - 1) - 1


def pow2_rows(x, w):
    """x == xi * 2^k with |xi| < 2^(w-1), per row, power-of-two scale."""
    m = x.abs().amax(-1, keepdim=True)
    assert m.max().item() < ROW_MAX, "input outside the spec domain"
    m = m.clamp(min=ROW_MIN)
    k = frexp_exp(m) - w + 1
    return torch.round(x * exp2i(-k)).clamp(-qmax(w), qmax(w)).long(), k


def exp_fixed(z, p):
    """Integer ~ 2^p * exp(-z) for z >= 0 (fp32), integer ops after one round."""
    u = torch.round(z.clamp(max=64.0) * (LOG2E * 2**FB)).long()
    ip = u >> FB
    frac = u & (2**FB - 1)
    hi, lo = frac >> (FB - HB), frac & (2 ** (FB - HB) - 1)
    lut, ln2lo = _tables(p, z.device)
    mult = (1 << 30) - ln2lo[lo]
    e = ((lut[hi] * mult) >> 30) >> ip.clamp(max=62)
    return torch.where(ip > p, torch.zeros_like(e), e)


_TABLES = {}


def _tables(p, device):
    key = (p, str(device))
    if key not in _TABLES:
        _TABLES[key] = (
            torch.tensor([round(2**p * 2 ** (-j / 2**HB)) for j in range(2**HB)],
                         dtype=torch.int64, device=device),
            torch.tensor(LN2_LO, dtype=torch.int64, device=device))
    return _TABLES[key]


def isqrt(m):
    """Exact floor(sqrt(m)) for int64 m >= 0 (float estimate + integer fix-up)."""
    s = torch.sqrt(m.float()).long()
    for _ in range(3):
        s = torch.where(s * s > m, s - 1, s)
        s = torch.where((s + 1) * (s + 1) <= m, s + 1, s)
    return s


def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & (2**64 - 1)
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & (2**64 - 1)
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & (2**64 - 1)
    return x ^ (x >> 31)


# ---- exact integer matmul, two independent algorithms


class Exact:
    """kind="f64": one float64 BLAS call (exact because |result| < 2^53).
    kind="limb": 8-bit limbs, fp32 matmuls over K-chunks of 256 so every
    partial sum is an integer < 2^24 (what an int8 tensor-core kernel does)."""

    def __init__(self, kind):
        self.kind = kind

    def mm(self, a, b, abits, bbits):
        K = a.shape[-1]
        if self.kind == "f64":
            assert abits + bbits + math.ceil(math.log2(K)) <= 53
            return (a.double() @ b.double()).long()
        na, nb = math.ceil((abits + 1) / 8), math.ceil((bbits + 1) / 8)
        acc = 0
        for k0 in range(0, K, 256):
            al = self.limbs(a[..., k0:k0 + 256], na)
            bl = self.limbs(b[..., k0:k0 + 256, :], nb)
            for i, ai in enumerate(al):
                for j, bj in enumerate(bl):
                    acc = acc + ((ai @ bj).long() << (8 * (i + j)))
        return acc

    @staticmethod
    def limbs(x, n):
        """Single-limb operands may be pre-converted fp32 (int8 weights);
        multi-limb operands must be integer tensors."""
        assert n == 1 or not x.is_floating_point()
        out = []
        for _ in range(n - 1):
            out.append((x & 255).float())
            x = x >> 8
        return out + [x.float()]


# ---- model


class ExactQwen2:
    def __init__(self, hf, device, kind):
        c = hf.config
        self.dev, self.mm = device, Exact(kind).mm
        self.H, self.nh, self.nkv = c.hidden_size, c.num_attention_heads, c.num_key_value_heads
        self.hd = self.H // self.nh
        self.eps = c.rms_norm_eps
        self.scaling = self.hd ** -0.5
        assert self.scaling == 2.0 ** round(math.log2(self.scaling))  # exact power of two
        sd = {k: v.detach().float() for k, v in hf.state_dict().items()}
        self.embed = sd["model.embed_tokens.weight"].to(device)

        def lin(name, bias=True):
            W = sd[name + ".weight"]
            s = W.abs().amax(1) / qmax(WBITS)
            Wq = torch.round(W / s[:, None]).clamp(-qmax(WBITS), qmax(WBITS)).long()
            b = sd.get(name + ".bias") if bias else None
            Wt = Wq.T.contiguous()
            Wt = Wt.double() if kind == "f64" else Wt.float()  # int8 values, exact
            return (Wt.to(device), s.to(device),
                    None if b is None else b.to(device))

        self.layers = []
        for i in range(c.num_hidden_layers):
            p = f"model.layers.{i}."
            self.layers.append({
                "ln1": sd[p + "input_layernorm.weight"].to(device),
                "ln2": sd[p + "post_attention_layernorm.weight"].to(device),
                "q": lin(p + "self_attn.q_proj"), "k": lin(p + "self_attn.k_proj"),
                "v": lin(p + "self_attn.v_proj"), "o": lin(p + "self_attn.o_proj", False),
                "g": lin(p + "mlp.gate_proj", False), "u": lin(p + "mlp.up_proj", False),
                "d": lin(p + "mlp.down_proj", False),
            })
        self.norm = sd["model.norm.weight"].to(device)
        lm = "lm_head" if "lm_head.weight" in sd else "model.embed_tokens"
        self.lm = lin(lm, False)
        theta = getattr(c, "rope_theta", None) or c.rope_parameters["rope_theta"]
        inv = 1.0 / (theta
                     ** (torch.arange(0, self.hd, 2, dtype=torch.float64) / self.hd))
        ang = torch.arange(8192, dtype=torch.float64)[:, None] * inv[None]
        ang = torch.cat([ang, ang], -1)
        self.cos, self.sin = ang.cos().float().to(device), ang.sin().float().to(device)
        self.vexp = None  # static V exponents [nh, 1, hd], set by calibrate()
        self.float_attn = False
        self.vmax = None
        self.clipped = 0

    def linear(self, x, wsb):
        W, s, b = wsb
        xi, k = pow2_rows(x, ACT_W)
        y = (self.mm(xi, W, ACT_W - 1, WBITS - 1).float() * exp2i(k)) * s
        return y if b is None else y + b

    def rmsnorm(self, x, g):
        xi, k = pow2_rows(x, RES_W)
        ss = (xi * xi).sum(-1, keepdim=True)
        epsi = torch.round(torch.tensor(self.eps, device=self.dev) * exp2i(-2 * k)).long()
        r = isqrt(ss // x.shape[-1] + epsi)
        yi = torch.div(xi << N_FRAC, r, rounding_mode="floor")
        return (yi.float() * 2.0**-N_FRAC) * g

    def rope(self, x, pos):
        c, s = self.cos[pos][None], self.sin[pos][None]
        half = self.hd // 2
        rot = torch.cat([-x[..., half:], x[..., :half]], -1)
        return x * c + rot * s

    def silu_mul(self, g, u):
        e = exp_fixed(g.abs(), P_SIG)
        den = (1 << P_SIG) + e
        pos = torch.div(torch.full_like(e, 1 << (P_SIG + S_FRAC)), den, rounding_mode="floor")
        neg = torch.div(e << S_FRAC, den, rounding_mode="floor")
        sig = torch.where(g >= 0, pos, neg)
        return ((sig.float() * 2.0**-S_FRAC) * g) * u

    def attention(self, q, k, v, offset):
        """q: [nh, nq, hd]; k, v: [nh, nk, hd] (already repeated for GQA)."""
        nq, nk = q.shape[1], k.shape[1]
        pos = torch.arange(max(nq, nk), device=self.dev)
        valid = pos[None, :nk] <= (pos[:nq, None] + offset)
        if self.float_attn:  # calibration only
            m = v.abs().amax(1, keepdim=True)
            self.vmax = m if self.vmax is None else torch.maximum(self.vmax, m)
            s = (q @ k.transpose(1, 2)) * self.scaling
            return torch.softmax(s.masked_fill(~valid, -math.inf), -1) @ v
        qi, kq = pow2_rows(q, QK_W)
        ki, kk = pow2_rows(k, QK_W)
        vr = v * exp2i(-self.vexp)
        self.clipped += int((vr.abs() > qmax(V_W)).sum().item())
        vi = torch.round(vr).clamp(-qmax(V_W), qmax(V_W)).long()
        S = self.mm(qi, ki.transpose(1, 2), QK_W - 1, QK_W - 1)
        L = (S.float() * exp2i(kq + kk.transpose(1, 2))) * self.scaling
        L = L.masked_fill(~valid, -math.inf)
        e = exp_fixed(L.amax(-1, keepdim=True) - L, P_ATT)
        E = e.sum(-1, keepdim=True)
        O = self.mm(e, vi, P_ATT + 1, V_W - 1)
        assert O.abs().max().item() < 2**(62 - A_FRAC)
        a = torch.div(O << A_FRAC, E, rounding_mode="floor")
        return a.float() * exp2i(self.vexp - A_FRAC)

    def forward(self, ids, cache, capture=None):
        """ids: 1-D LongTensor of new tokens; cache: list of (K, V) per layer.
        Returns logits for every new position."""
        start = cache["len"]
        n = len(ids)
        pos = torch.arange(start, start + n, device=self.dev)
        x = self.embed[ids.to(self.dev)]
        g = self.nh // self.nkv
        for li, L in enumerate(self.layers):
            xn = self.rmsnorm(x, L["ln1"])
            q = self.linear(xn, L["q"]).view(n, self.nh, self.hd).transpose(0, 1)
            k = self.linear(xn, L["k"]).view(n, self.nkv, self.hd).transpose(0, 1)
            v = self.linear(xn, L["v"]).view(n, self.nkv, self.hd).transpose(0, 1)
            q, k = self.rope(q, pos), self.rope(k, pos)
            K, V = cache["kv"][li]
            K = k if K is None else torch.cat([K, k], 1)
            V = v if V is None else torch.cat([V, v], 1)
            cache["kv"][li] = (K, V)
            a = self.attention(q, K.repeat_interleave(g, 0), V.repeat_interleave(g, 0), start)
            x = x + self.linear(a.transpose(0, 1).reshape(n, self.H), L["o"])
            xn = self.rmsnorm(x, L["ln2"])
            h = self.silu_mul(self.linear(xn, L["g"]), self.linear(xn, L["u"]))
            x = x + self.linear(h, L["d"])
            if capture is not None:
                for j in range(n):
                    capture.setdefault(start + j, []).append(x[j].cpu().clone())
        cache["len"] += n
        return self.linear(self.rmsnorm(x, self.norm), self.lm)

    def new_cache(self):
        return {"len": 0, "kv": [(None, None)] * len(self.layers)}

    def calibrate(self, ids, headroom=2.0):
        self.float_attn = True
        self.forward(ids, self.new_cache())
        self.float_attn = False
        vmax = self.vmax.clamp(min=2.0**-100) * headroom
        return frexp_exp(vmax) - V_W + 1


# ---- decoding


def select(logits, step, seed, inv_temp):
    """Greedy (seed None) or exact sampling: integer weights + integer PRNG."""
    m = logits.max()
    if seed is None:
        return int(torch.nonzero(logits == m)[0].item())
    e = exp_fixed((m - logits) * inv_temp, P_SAMPLE)
    cs = e.cumsum(0)
    r = splitmix64((seed << 32) + step) % int(cs[-1].item())
    return int((cs <= r).sum().item())


def generate(model, prompt, n_new, seed=None, inv_temp=1.25, chunks=None):
    cache, cap = model.new_cache(), {}
    bounds = [0] + (chunks or []) + [len(prompt)]
    for a, b in zip(bounds, bounds[1:]):
        logits = model.forward(prompt[a:b], cache, cap)
    out = []
    for step in range(n_new):
        t = select(logits[-1], step, seed, inv_temp)
        out.append(t)
        logits = model.forward(torch.tensor([t]), cache, cap)
    return out, cap


def recompute(model, prompt, gen, seed=None, inv_temp=1.25):
    """Regenerate on challenge: one pass over prompt+output, re-derive tokens."""
    cap = {}
    full = torch.tensor(prompt.tolist() + gen)
    logits = model.forward(full, model.new_cache(), cap)
    toks = [select(logits[len(prompt) - 1 + s], s, seed, inv_temp) for s in range(len(gen))]
    return toks, cap


def root(cap):
    h = hashlib.sha256()
    for p in sorted(cap):
        h.update(hashlib.sha256(b"".join(t.numpy().tobytes() for t in cap[p])).digest())
    return h.hexdigest()[:16]


def diff(c1, c2):
    """Count (position, layer) hidden states that differ in any bit."""
    bad = sum(int(not torch.equal(a.view(torch.int32), b.view(torch.int32)))
              for p in c1 for a, b in zip(c1[p], c2[p]))
    return bad, sum(len(v) for v in c1.values())


def main():
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    torch.set_grad_enabled(False)

    from datasets import load_dataset
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    calib = tok("\n\n".join(train["text"][:400]), return_tensors="pt").input_ids[0][:512]
    evalids = tok("\n\n".join(test["text"][:400]), return_tensors="pt").input_ids[0][:512]

    msgs = [{"role": "user", "content": "Write a short story about a robot learning to paint."}]
    prompt = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                 return_tensors="pt").input_ids[0]
    N = 40

    # Static V exponents are part of the spec: calibrated once, shipped to every device.
    models = {"cpu": ExactQwen2(hf, "cpu", "f64"), "mps": ExactQwen2(hf, "mps", "limb")}
    vexp = models["cpu"].calibrate(calib)
    models["cpu"].vexp, models["mps"].vexp = vexp, vexp.to("mps")

    # 1. accuracy vs HF fp32
    ref = hf(evalids[None]).logits[0].float()
    ex = models["cpu"].forward(evalids, models["cpu"].new_cache()).float()
    lr, le = torch.log_softmax(ref[:-1], -1), torch.log_softmax(ex[:-1], -1)
    tgt = evalids[1:, None]
    print(f"\n1. accuracy (512 wikitext tokens): ppl HF fp32 {math.exp(-lr.gather(-1, tgt).mean()):.3f}"
          f" | exact spec {math.exp(-le.gather(-1, tgt).mean()):.3f}"
          f" | top-1 agree {(lr.argmax(-1) == le.argmax(-1)).float().mean():.4f}"
          f" | KL {(lr.exp() * (lr - le)).sum(-1).mean():.4f}")

    # 2-4. decode on each device, prompt chunked differently, then recompute in one pass
    for label, seed in [("greedy", None), ("sampled T=0.8", 42)]:
        print(f"\n== {label}")
        runs = {}
        for dev, chunks in [("cpu", None), ("mps", [5, 17])]:
            m = models[dev]
            t0 = time.time()
            gen, cap = generate(m, prompt, N, seed, chunks=chunks)
            t1 = time.time()
            re_toks, re_cap = recompute(m, prompt, gen, seed)
            t2 = time.time()
            runs[dev] = (gen, cap, re_toks, re_cap)
            print(f"[{dev}] decode {t1 - t0:.1f}s, recompute {t2 - t1:.1f}s | "
                  f"root decode {root(cap)} recompute {root(re_cap)} | "
                  f"tokens re-derived equal: {re_toks == gen} | "
                  f"hidden-state mismatches decode vs recompute: {diff(cap, re_cap)}")
        (gc, cc, _, _), (gm, cm, _, _) = runs["cpu"], runs["mps"]
        print(f"CPU vs MPS: tokens equal {gc == gm} | hidden-state mismatches {diff(cc, cm)}")
        print("text:", repr(tok.decode(gc)))
    print("\nV values clipped by static scale:", models["cpu"].clipped + models["mps"].clipped)


if __name__ == "__main__":
    main()
