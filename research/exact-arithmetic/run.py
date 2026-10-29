"""
Does integer attention work? Accuracy + bit-exactness checks.

  python run.py --model Qwen/Qwen2.5-0.5B-Instruct --seqs 4 --seqlen 512
"""

import argparse
import json
import math
import random
import re
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import int_attention as ia

# variant: "ref" | "calib" | "qfloat_Q-K-V" | "int_Q-K-V[_lbX_pY]"
STATE = {"variant": "ref", "capture": None, "vmax": {}, "vs": {}, "kv": {}, "clip": [],
         "inexact": []}
HEADROOM = 2.0


def parse(var):
    parts = var.split("_")
    bits = tuple(int(x) for x in parts[1].split("-"))
    kw = {m[1]: int(m[2]) for m in (re.fullmatch(r"([a-z]+)(\d+)", x) for x in parts[2:])}
    return parts[0], bits, kw


def attention(q, k, v, scaling, offset, layer):
    var = STATE["variant"]
    if var == "ref":
        return ia.float_attention(q, k, v, scaling, offset)
    if var == "calib":
        m = v.abs().amax(1, keepdim=True)
        STATE["vmax"][layer] = torch.maximum(STATE["vmax"].get(layer, m), m)
        return ia.float_attention(q, k, v, scaling, offset)
    if var == "stockbf16":
        return ia.stock_bf16_attention(q, k, v, scaling, offset)
    if var == "exactA":
        out, inexact = ia.exact_attention(q, k, v, scaling, offset, STATE["kv"][layer])
        STATE["inexact"].append(inexact)
        return out
    kind, bits, kw = parse(var)
    vs = STATE["vs"][layer]
    if kind == "qfloat":
        return ia.quant_float_attention(q, k, v, scaling, offset, vs, bits)
    out, clip = ia.int_attention(q, k, v, scaling, offset, vs, bits, **kw)
    STATE["clip"].append(clip)
    return out


def attn_fn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    assert query.shape[0] == 1, "batch 1 only (mask is rebuilt as plain causal)"
    g = module.num_key_value_groups
    q = query[0].float()
    k = key[0].float().repeat_interleave(g, dim=0)
    v = value[0].float().repeat_interleave(g, dim=0)
    offset = k.shape[1] - q.shape[1]
    cap = STATE["capture"]
    if cap is not None and module.layer_idx == cap["layer"]:
        cap["qkv"] = (q.clone(), k.clone(), v.clone(), scaling, offset)
    out = attention(q, k, v, scaling, offset, module.layer_idx)
    return out.to(query.dtype).transpose(0, 1).unsqueeze(0).contiguous(), None


ALL_ATTENTION_FUNCTIONS.register("intattn", attn_fn)


def wikitext_ids(tok, split):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return tok("\n\n".join(ds["text"][:20000]), return_tensors="pt").input_ids[0]


def chunks_of(ids, n, seqlen):
    step = len(ids) // n
    return [ids[i * step : i * step + seqlen] for i in range(n)]


@torch.no_grad()
def logits_for(model, ids):
    return model(ids[None]).logits[0].float()


def compare(ref_logits, logits, ids):
    lp_ref = torch.log_softmax(ref_logits[:-1], -1)
    lp = torch.log_softmax(logits[:-1], -1)
    nll = -lp.gather(-1, ids[1:, None]).mean().item()
    kl = (lp_ref.exp() * (lp_ref - lp)).sum(-1)
    top1 = (lp_ref.argmax(-1) == lp.argmax(-1)).float().mean().item()
    return {"nll": nll, "kl_mean": kl.mean().item(), "kl_p99": kl.quantile(0.99).item(),
            "kl_max": kl.max().item(), "top1_agree": top1}


PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "Explain in two sentences why the sky is blue.",
    "Q: A train travels 120 km in 1.5 hours. What is its average speed?\nA:",
    "Write a haiku about cryptography.",
]


@torch.no_grad()
def greedy(model, tok, prompt, n):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) \
        if tok.chat_template else prompt
    ids = tok(text, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=n, do_sample=False, use_cache=True)
    return out[0, ids.shape[1]:].tolist()


def bitexact_checks(qkv, vs, bits, lb=8, p=20, n_rows=6, seed=0):
    """(1) vectorized spec == independent per-row reference (numpy fp32 + Python ints)
    (2) two-pass evaluation over random tiles in random order == vectorized result."""
    q, k, v, scaling, offset = qkv
    rng = random.Random(seed)
    full, _ = ia.int_attention(q, k, v, scaling, offset, vs, bits, lb=lb, p=p)
    h, nq, d = q.shape
    rows = [(rng.randrange(h), rng.randrange(nq)) for _ in range(n_rows)]
    mism_ref = 0
    for hh, i in rows:
        r = ia.reference_row(q[hh, i].numpy(), k[hh].numpy(), v[hh].numpy(),
                             vs[hh, 0].numpy(), scaling, i, offset, bits, lb=lb, p=p)
        mism_ref += int(not np.array_equal(r, full[hh, i].numpy()))

    lut = ia.make_lut(lb, p)
    qi, sq = ia.quant_rows(q, bits[0])
    ki, sk = ia.quant_rows(k, bits[1])
    vi = ia.quant_static(v, vs, bits[2]).long()
    mism_tile = 0
    for hh, i in rows:
        n = i + offset + 1
        cuts = sorted(rng.sample(range(1, n), min(n - 1, rng.randint(1, 12)))) if n > 1 else []
        tiles = list(zip([0] + cuts, cuts + [n]))
        S = (qi[hh, i].double() @ ki[hh, :n].double().T)[None]
        # pass 1: running max over tiles in random order
        rng.shuffle(tiles)
        L = S.float() * ((sq[hh, i][None] * sk[hh, :n].T) * scaling)
        m = max(L[0, a:b].max() for a, b in tiles)
        assert m == L.max()
        # pass 2: integer sums over tiles in another random order
        rng.shuffle(tiles)
        valid = torch.ones(1, n, dtype=torch.bool)
        e = ia.int_weights(S, sq[hh, i][None], sk[hh, :n].T, scaling, valid, lut, lb, p)[0]
        E = torch.zeros((), dtype=torch.int64)
        O = torch.zeros(d, dtype=torch.int64)
        for a, b in tiles:
            E += e[a:b].sum()
            O += (e[a:b, None] * vi[hh, a:b]).sum(0)
        tiled = (O.float() / E.float()) * vs[hh, 0]
        mism_tile += int(not torch.equal(tiled, full[hh, i]))
    return {"bits": bits, "lb": lb, "p": p, "rows_checked": n_rows,
            "ref_mismatch": mism_ref, "tiled_mismatch": mism_tile}


def bitexact_exact(qkv, kv, n_rows=6, seed=1):
    """Option A: vectorized == per-row reference, and == random-tile-order sums."""
    q, k, v, scaling, offset = qkv
    rng = random.Random(seed)
    full, _ = ia.exact_attention(q, k, v, scaling, offset, kv)
    h, nq, d = q.shape
    rows = [(rng.randrange(h), rng.randrange(nq)) for _ in range(n_rows)]
    mism_ref = sum(int(not np.array_equal(
        ia.reference_row_exact(q[hh, i].numpy(), k[hh].numpy(), v[hh].numpy(),
                               kv[hh, 0].numpy(), scaling, i, offset),
        full[hh, i].numpy())) for hh, i in rows)
    qb, kb, vb = (x.to(torch.bfloat16).float() for x in (q, k, v))
    qi, kq = ia.pow2_rows(qb, 24)
    ki, kk = ia.pow2_rows(kb, 24)
    vi = torch.round(torch.ldexp(vb, -kv)).clamp_(-ia.qmax(24), ia.qmax(24)).long()
    lut = ia.make_exp_lut(24)
    mism_tile = 0
    for hh, i in rows:
        n = i + offset + 1
        cuts = sorted(rng.sample(range(1, n), min(n - 1, rng.randint(1, 12)))) if n > 1 else []
        tiles = list(zip([0] + cuts, cuts + [n]))
        S = qi[hh, i].long() @ ki[hh, :n].long().T  # exact int64 (CPU int matmul)
        L = torch.ldexp(S.float(), kq[hh, i] + kk[hh, :n, 0]) * scaling
        rng.shuffle(tiles)
        m = max(L[a:b].max() for a, b in tiles)
        e = ia.exp_fixed(m - L, lut, 24)
        rng.shuffle(tiles)
        E = torch.zeros((), dtype=torch.int64)
        O = torch.zeros(d, dtype=torch.int64)
        for a, b in tiles:
            E += e[a:b].sum()
            O += (e[a:b, None] * vi[hh, a:b]).sum(0)
        tiled = torch.ldexp(O.float() / E.float(), kv[hh, 0])
        mism_tile += int(not torch.equal(tiled, full[hh, i]))
    return {"rows_checked": n_rows, "ref_mismatch": mism_ref, "tiled_mismatch": mism_tile}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--calib-seqs", type=int, default=4)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--variants", default="qfloat_16-16-16,int_16-16-16")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="intattn")
    model.eval()
    variants = args.variants.split(",")
    res = {"model": args.model, "seqlen": args.seqlen, "seqs": args.seqs,
           "ppl": {}, "vs_ref": {}}

    # Calibrate static per-(layer, head, channel) V scales on the *train* split.
    STATE["variant"] = "calib"
    for ids in chunks_of(wikitext_ids(tok, "train"), args.calib_seqs, args.seqlen):
        logits_for(model, ids)
    vbits = {parse(v)[1][2] for v in variants if v.startswith(("int_", "qfloat_"))}
    assert len(vbits) <= 1, "one V width per run (static scales depend on it)"
    if vbits:
        vq = ia.qmax(vbits.pop())
        STATE["vs"] = {l: m.clamp(min=1e-6) * HEADROOM / vq for l, m in STATE["vmax"].items()}
    STATE["kv"] = {l: ia.pow2_static(m.clamp(min=1e-6) * HEADROOM, 24)
                   for l, m in STATE["vmax"].items()}

    chunks = chunks_of(wikitext_ids(tok, "test"), args.seqs, args.seqlen)
    acc = {v: [] for v in variants}
    nll_ref = []
    for ci, ids in enumerate(chunks):
        t0 = time.time()
        STATE["variant"] = "ref"
        ref = logits_for(model, ids)
        nll_ref.append(compare(ref, ref, ids)["nll"])
        for var in variants:
            STATE["variant"] = var
            acc[var].append(compare(ref, logits_for(model, ids), ids))
        print(f"chunk {ci}: {time.time() - t0:.1f}s", flush=True)
    res["ppl"]["ref"] = math.exp(np.mean(nll_ref))
    for var in variants:
        res["ppl"][var] = math.exp(np.mean([m["nll"] for m in acc[var]]))
        res["vs_ref"][var] = {
            "kl_mean": float(np.mean([m["kl_mean"] for m in acc[var]])),
            "kl_p99": float(np.mean([m["kl_p99"] for m in acc[var]])),
            "kl_max": float(max(m["kl_max"] for m in acc[var])),
            "top1_agree": float(np.mean([m["top1_agree"] for m in acc[var]])),
        }
    res["v_clip_frac_max"] = max(STATE["clip"], default=0.0)
    res["exactA_inexact_frac_mean"] = float(np.mean(STATE["inexact"])) if STATE["inexact"] else None
    print(json.dumps({k: res[k] for k in ("ppl", "vs_ref", "v_clip_frac_max",
                                          "exactA_inexact_frac_mean")}, indent=2),
          flush=True)

    if args.gen:
        gen = {}
        for var in ["ref"] + variants:
            STATE["variant"] = var
            gen[var] = [greedy(model, tok, p, args.gen) for p in PROMPTS]
        res["gen_first_divergence"] = {}
        for var in variants:
            res["gen_first_divergence"][var] = [
                next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
                for a, b in zip(gen["ref"], gen[var])]
        res["gen"] = {k: [tok.decode(x) for x in v] for k, v in gen.items()}
        print(f"first divergence vs ref (of {args.gen}):", res["gen_first_divergence"],
              flush=True)

    # Bit-exactness on captured real Q/K/V (middle layer, 256-token chunk).
    layer = model.config.num_hidden_layers // 2
    STATE["variant"] = "ref"
    STATE["capture"] = {"layer": layer}
    logits_for(model, chunks[0][:256])
    qkv = STATE["capture"]["qkv"]
    res["bitexact"] = {}
    for var in [v for v in variants if v.startswith("int_")]:
        _, bits, kw = parse(var)
        res["bitexact"][var] = bitexact_checks(qkv, STATE["vs"][layer], bits, **kw)
    if "exactA" in variants:
        res["bitexact"]["exactA"] = bitexact_exact(qkv, STATE["kv"][layer])
    print("bitexact:", res["bitexact"], flush=True)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
