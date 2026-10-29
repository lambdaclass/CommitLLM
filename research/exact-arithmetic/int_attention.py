"""
Integer-attention spec (v1) + simulation helpers.

Every reduction (Q.K over d, sum of weights over t, weights.V over t) is an
integer sum, so the result is independent of tiling / split-KV / batch
order. Everything else is elementwise: single IEEE fp32 ops (no FMA) or
integer ops. A CPU verifier can therefore reproduce any (layer, head, token)
bit-exactly.

Scales applied *after* a reduction may be per token (Q, K: per row). A scale
inside the reduction over t would break exactness, so V uses a static
per-(layer, head, channel) scale `vs` fixed offline (calibration).

Spec, per (head, query i), keys t <= i + offset, widths QB/KB/VB:
  qi, sq = quant(q, QB)     ki_t, sk_t = quant(k_t, KB)
  vi_t   = clamp(round(v_t / vs), VB)
  S_t   = sum_d qi*ki_t                                (int)
  L_t   = f32(S_t) * ((sq * sk_t) * scaling)           (fp32, fixed op order)
  z_t   = max_t L_t - L_t                              (fp32, >= 0)
  u_t   = round(min(z_t, 64) * (log2e * 2^LB))         (int)
  e_t   = LUT[u_t & (2^LB-1)] >> (u_t >> LB)           (int, <= 2^P)
          LUT[j] = round(2^P * 2^(-j / 2^LB))
  E     = sum_t e_t                                    (int64)
  O_c   = sum_t e_t * vi_tc                            (int64, < 2^(P+VB+log2 n))
  a_c   = (f32(O_c) / f32(E)) * vs_c                  (fp32)
"""

import math

import numpy as np
import torch

LOG2E = 1.4426950408889634
# Input domain: row maxima are clamped up to ROW_MIN and must stay below ROW_MAX,
# so every power-of-two scale (and sums of two of them) stays in the fp32
# normal range. Rows below ROW_MIN are ~0 and behave identically after clamping.
ROW_MIN, ROW_MAX = 2.0**-40, 2.0**40
SPLIT = 20  # bits per half when emulating int64 matmul with float64 BLAS


def qmax(bits):
    return float(2 ** (bits - 1) - 1)


def row_max(x):
    m = x.abs().amax(-1, keepdim=True)
    assert m.max().item() < ROW_MAX, "input outside the spec domain"
    return m.clamp(min=ROW_MIN)


def quant_rows(x, bits):
    """Symmetric per-row (last dim) quantization. Returns (ints as fp32, fp32 scale)."""
    s = row_max(x) / qmax(bits)
    return torch.round(x / s).clamp_(-qmax(bits), qmax(bits)), s


def quant_static(x, vs, bits):
    return torch.round(x / vs).clamp_(-qmax(bits), qmax(bits))


def make_lut(lb, p):
    return torch.tensor([round(2**p * 2 ** (-j / 2**lb)) for j in range(2**lb)],
                        dtype=torch.int64)


def exact_mm(a, b):
    """Exact int64 matmul of non-negative a (< 2^(2*SPLIT)) and small-int b,
    via two float64 BLAS calls. Bounded a priori: every partial sum of every
    call is at most 2^SPLIT * max|b| * K < 2^53."""
    assert a.min().item() >= 0 and a.max().item() < 2 ** (2 * SPLIT)
    assert 2**SPLIT * b.abs().max().item() * a.shape[-1] < 2**53
    lo = (a & ((1 << SPLIT) - 1)).double() @ b.double()
    hi = (a >> SPLIT).double() @ b.double()
    return (hi.round().long() << SPLIT) + lo.round().long()


def int_weights(S, sq, skT, scaling, valid, lut, lb, p):
    """S: exact int scores (float64), valid: bool mask. Returns int64 e_t."""
    L = S.float() * ((sq * skT) * scaling)
    L = L.masked_fill(~valid, -math.inf)
    z = (L.amax(-1, keepdim=True) - L).clamp(max=64.0)
    u = torch.round(z * (LOG2E * 2**lb)).long()
    ip = u >> lb
    e = lut[u & (2**lb - 1)] >> ip.clamp(max=62)
    return torch.where(ip > p, torch.zeros_like(e), e)


def int_attention(q, k, v, scaling, offset, vs, bits=(16, 16, 16), lb=8, p=20,
                  qblock=256):
    """q: [h, nq, d], k/v: [h, nk, d] fp32 (kv already repeated for GQA),
    vs: [h, 1, d] static V scale. Returns (fp32 [h, nq, d], V clip fraction)."""
    nq, nk = q.shape[1], k.shape[1]
    lut = make_lut(lb, p)
    qi, sq = quant_rows(q, bits[0])
    ki, sk = quant_rows(k, bits[1])
    vi = quant_static(v, vs, bits[2])
    clip = ((v / vs).abs() > qmax(bits[2])).float().mean().item()
    kiT = ki.double().transpose(1, 2)
    skT = sk.transpose(1, 2)
    assert qmax(bits[0]) * qmax(bits[1]) * q.shape[-1] < 2**53
    assert 2**p * qmax(bits[2]) * nk < 2**63, "int64 range: context too long for widths"
    out = torch.empty_like(q)
    kpos = torch.arange(nk)
    for i0 in range(0, nq, qblock):
        i1 = min(nq, i0 + qblock)
        S = qi[:, i0:i1].double() @ kiT  # exact: qmax^2 * d < 2^53 (asserted below)
        valid = kpos[None, :] <= (torch.arange(i0, i1)[:, None] + offset)
        e = int_weights(S, sq[:, i0:i1], skT, scaling, valid, lut, lb, p)
        E = e.sum(-1, keepdim=True)
        O = exact_mm(e, vi)
        out[:, i0:i1] = (O.float() / E.float()) * vs
    return out, clip


def quant_float_attention(q, k, v, scaling, offset, vs, bits=(16, 16, 16)):
    """Same Q/K/V quantization, but float softmax: isolates the quantization
    error from the integer-softmax error."""
    (qi, sq), (ki, sk) = quant_rows(q, bits[0]), quant_rows(k, bits[1])
    return float_attention(qi * sq, ki * sk, quant_static(v, vs, bits[2]) * vs,
                           scaling, offset)


def float_attention(q, k, v, scaling, offset):
    nq, nk = q.shape[1], k.shape[1]
    s = (q @ k.transpose(1, 2)) * scaling
    pos = torch.arange(max(nq, nk), device=q.device)
    valid = pos[None, :nk] <= (pos[:nq, None] + offset)
    s = s.masked_fill(~valid, -math.inf)
    return torch.softmax(s, dim=-1) @ v


# ---- independent reference for a single (head, query): numpy fp32 + Python ints


def reference_row(q, k, v, vs, scaling, i, offset, bits=(16, 16, 16), lb=8, p=20):
    """Recompute one head/query output the way a standalone verifier would:
    per-element numpy float32 ops and arbitrary-precision Python ints, keys
    visited one at a time. q, vs: [d]; k, v: [nk, d] numpy float32."""
    f = np.float32

    def clampq(x, b):
        m = 2 ** (b - 1) - 1
        return int(max(-m, min(m, np.round(x))))

    def quant_row(x, b):
        s = f(max(f(np.abs(x).max()), f(ROW_MIN)) / f(qmax(b)))
        return [clampq(f(xi / s), b) for xi in x], s

    lut = [round(2**p * 2 ** (-j / 2**lb)) for j in range(2**lb)]
    qi, sq = quant_row(q, bits[0])
    n = i + offset + 1
    L = []
    for t in range(n):
        kt, sk = quant_row(k[t], bits[1])
        S = sum(a * b for a, b in zip(qi, kt))
        L.append(f(f(S) * f(f(sq * sk) * f(scaling))))
    m = max(L)
    E = 0
    O = [0] * len(qi)
    for t in range(n):
        z = min(f(m - L[t]), f(64.0))
        u = int(np.round(f(z * f(LOG2E * 2**lb))))
        ip = u >> lb
        e = 0 if ip > p else lut[u & (2**lb - 1)] >> ip
        E += e
        for c in range(len(O)):
            O[c] += e * clampq(f(v[t, c] / vs[c]), bits[2])
    return np.array([f(f(np.int64(O[c])) / f(np.int64(E))) * vs[c]
                     for c in range(len(O))], dtype=np.float32)


# ---- Option A: bf16 inputs, exact accumulation (Kulisch / Ozaki style)
#
#   x = bf16(x)                                        (as served)
#   xi, kx = pow2_rows(x, W)     x == xi * 2^kx exactly unless |x| < 2^-(W-9) * rowmax
#   vi = round(v * 2^-kv_c)      kv_c static power of two per (layer, head, channel)
#   S_t = sum_d qi*ki_t                                (exact int64)
#   L_t = (f32(S_t) * 2^(kq + kk_t)) * scaling         (one rounding + exact scale + one)
#   u_t = round(min(max L - L_t, 64) * log2e * 2^FB)   (fixed point, FB frac bits)
#   e_t = (LUT[hi(u_t)] * (2^30 - round(lo(u_t) * ln2 * 2^(30-FB)))) >> 30 >> int(u_t)
#   E = sum e_t ; O_c = sum e_t * vi_tc                (exact int64)
#   a_c = (f32(O_c) / f32(E)) * 2^kv_c

FB = 20       # fractional bits of u
HB = 12       # LUT index bits (top of the fraction); remaining bits use a linear term
LIMB = 16


def pow2_rows(x, w):
    """Per-row power-of-two scale so the row max uses w-1 magnitude bits.
    Row maxima are clamped to the spec domain (ROW_MIN), so zero rows get a scale."""
    _, e = torch.frexp(row_max(x))
    k = e - w + 1
    xi = torch.round(torch.ldexp(x, -k)).clamp_(-qmax(w), qmax(w))
    return xi, k


def pow2_static(vmax, w):
    _, e = torch.frexp(vmax)
    return e - w + 1


def exact_mm_signed(a, b):
    """Exact int64 a @ b (|a|,|b| < 2^31) via 16-bit limbs and float64 BLAS.
    Every limb is < 2^16 in magnitude, so partial sums are < 2^32 * K < 2^53."""
    assert a.abs().max().item() < 2**31 and b.abs().max().item() < 2**31
    assert a.shape[-1] < 2**21

    def split(x):
        return [(x & 0xFFFF).double(), (x >> LIMB).double()]
    acc = None
    for i, ai in enumerate(split(a)):
        for j, bj in enumerate(split(b)):
            r = ai @ bj
            part = r.round().long() << (LIMB * (i + j))
            acc = part if acc is None else acc + part
    return acc


def make_exp_lut(p):
    return torch.tensor([round(2**p * 2 ** (-j / 2**HB)) for j in range(2**HB)],
                        dtype=torch.int64)


LN2_LO = [round(l * math.log(2) * 2 ** (30 - FB)) for l in range(2 ** (FB - HB))]


def exp_fixed(z, lut, p):
    """e = round-ish(2^p * exp(-z)) with ~fp32 accuracy, integer ops only after u."""
    u = torch.round(z.clamp(max=64.0) * (LOG2E * 2**FB)).long()
    ip = u >> FB
    frac = u & (2**FB - 1)
    hi, lo = frac >> (FB - HB), frac & (2 ** (FB - HB) - 1)
    mult = (1 << 30) - torch.tensor(LN2_LO, dtype=torch.int64, device=z.device)[lo]
    e = ((lut[hi] * mult) >> 30) >> ip.clamp(max=62)
    return torch.where(ip > p, torch.zeros_like(e), e)


def exact_attention(q, k, v, scaling, offset, kv, w=24, p=24, qblock=256):
    """Option A. q: [h, nq, d], k/v: [h, nk, d] fp32 (bf16-rounded here),
    kv: [h, 1, d] static V exponents. Returns (fp32 out, fraction of Q/K/V
    elements that were *not* represented exactly)."""
    q, k, v = (x.to(torch.bfloat16).float() for x in (q, k, v))
    nq, nk = q.shape[1], k.shape[1]
    lut = make_exp_lut(p)
    qi, kq = pow2_rows(q, w)
    ki, kk = pow2_rows(k, w)
    vi = torch.round(torch.ldexp(v, -kv)).clamp_(-qmax(w), qmax(w))
    inexact = sum(((torch.ldexp(xi, s) != x).float().mean().item())
                  for xi, s, x in ((qi, kq, q), (ki, kk, k), (vi, kv, v))) / 3
    qi, kiT, vi = qi.long(), ki.long().transpose(1, 2), vi.long()
    # int64 range: O_c = sum_t e_t * vi_tc <= 2^p * qmax(w) * nk. With p = w = 24
    # this caps context at 2^16 keys; longer contexts need smaller p or wider sums.
    assert 2**p * qmax(w) * nk < 2**63, "int64 range: context too long for widths"
    kkT = kk.transpose(1, 2)
    out = torch.empty_like(q)
    kpos = torch.arange(nk)
    for i0 in range(0, nq, qblock):
        i1 = min(nq, i0 + qblock)
        S = exact_mm_signed(qi[:, i0:i1], kiT)
        L = torch.ldexp(S.float(), kq[:, i0:i1] + kkT) * scaling
        valid = kpos[None, :] <= (torch.arange(i0, i1)[:, None] + offset)
        L = L.masked_fill(~valid, -math.inf)
        e = exp_fixed(L.amax(-1, keepdim=True) - L, lut, p)
        E = e.sum(-1, keepdim=True)
        O = exact_mm_signed(e, vi)
        out[:, i0:i1] = torch.ldexp(O.float() / E.float(), kv)
    return out, inexact


def stock_bf16_attention(q, k, v, scaling, offset):
    """What FlashAttention does numerically: bf16 Q/K/V, fp32 scores/softmax,
    P cast to bf16 before P@V, fp32 accumulation (order effects not modeled)."""
    q, k, v = (x.to(torch.bfloat16).float() for x in (q, k, v))
    nq, nk = q.shape[1], k.shape[1]
    s = (q @ k.transpose(1, 2)) * scaling
    pos = torch.arange(max(nq, nk))
    s = s.masked_fill(~(pos[None, :nk] <= (pos[:nq, None] + offset)), -math.inf)
    m = s.amax(-1, keepdim=True)
    pexp = torch.exp(s - m)
    return (pexp.to(torch.bfloat16).float() @ v) / pexp.sum(-1, keepdim=True)


def reference_row_exact(q, k, v, kv, scaling, i, offset, w=24, p=24):
    """Option A for one (head, query) with numpy fp32 scalars and Python ints,
    keys visited one at a time (independent of the vectorized code)."""
    f = np.float32
    bf = lambda x: torch.tensor(x).to(torch.bfloat16).float().numpy()
    q, k, v = bf(q), bf(k), bf(v)
    m = 2 ** (w - 1) - 1

    def row(x):
        e = int(np.frexp(max(f(np.abs(x).max()), f(ROW_MIN)))[1]) - w + 1
        return [max(-m, min(m, int(np.round(np.ldexp(f(xi), -e))))) for xi in x], e

    lut = [round(2**p * 2 ** (-j / 2**HB)) for j in range(2**HB)]
    qi, eq = row(q)
    n = i + offset + 1
    L = []
    for t in range(n):
        kt, ek = row(k[t])
        S = sum(a * b for a, b in zip(qi, kt))
        L.append(f(f(f(S) * f(2.0 ** (eq + ek))) * f(scaling)))
    mx = max(L)
    E, O = 0, [0] * len(qi)
    for t in range(n):
        z = min(f(mx - L[t]), f(64.0))
        u = int(np.round(f(z * f(LOG2E * 2**FB))))
        ip, frac = u >> FB, u & (2**FB - 1)
        hi, lo = frac >> (FB - HB), frac & (2 ** (FB - HB) - 1)
        e = 0 if ip > p else ((lut[hi] * ((1 << 30) - LN2_LO[lo])) >> 30) >> ip
        E += e
        for c in range(len(O)):
            O[c] += e * max(-m, min(m, int(np.round(np.ldexp(f(v[t, c]), -int(kv[c]))))))
    return np.array([np.ldexp(f(f(np.int64(O[c])) / f(np.int64(E))), int(kv[c]))
                     for c in range(len(O))], dtype=np.float32)
