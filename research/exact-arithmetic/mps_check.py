"""
Cross-hardware bit-exactness: integer attention (int16 spec and Option A) on
the Apple GPU (MPS) vs the CPU reference in int_attention.py.

MPS has no float64/int64 matmul, so integer dot products are computed the way
an int8 tensor-core kernel would: split each integer into 8-bit limbs, run fp32
matmuls whose partial sums are integers < 2^24 (hence exact in any order),
and recombine in int64.
"""

import math

import torch

import int_attention as ia

DEV = "mps"
TCHUNK = 128  # keys per limb matmul: 128 * 255 * 255 < 2^24


def limbs(x, n):
    """Signed int64 tensor -> n limbs, low limbs unsigned 8-bit, top limb signed."""
    out = []
    for _ in range(n - 1):
        out.append((x & 255).float())
        x = x >> 8
    out.append(x.float())
    return out


def exact_mm_limbs(a, b, na, nb):
    """Exact int64 a @ b on the GPU via 8-bit limb fp32 matmuls over key chunks."""
    acc = None
    for t0 in range(0, a.shape[-1], TCHUNK):
        al = limbs(a[..., t0:t0 + TCHUNK], na)
        bl = limbs(b[..., t0:t0 + TCHUNK, :], nb)
        for i, ai in enumerate(al):
            for j, bj in enumerate(bl):
                part = (ai @ bj).long() << (8 * (i + j))
                acc = part if acc is None else acc + part
    return acc


def gpu_int_attention(q, k, v, scaling, vs, bits, lb=8, p=20):
    h, n, d = q.shape
    lut = ia.make_lut(lb, p).to(DEV)
    qi, sq = ia.quant_rows(q, bits[0])
    ki, sk = ia.quant_rows(k, bits[1])
    vi = ia.quant_static(v, vs, bits[2]).long()
    S = exact_mm_limbs(qi.long(), ki.long().transpose(1, 2).contiguous(), 2, 2)
    L = S.float() * ((sq * sk.transpose(1, 2)) * scaling)
    valid = torch.arange(n, device=DEV)[None, :] <= torch.arange(n, device=DEV)[:, None]
    L = L.masked_fill(~valid, -math.inf)
    z = (L.amax(-1, keepdim=True) - L).clamp(max=64.0)
    u = torch.round(z * (ia.LOG2E * 2**lb)).long()
    ip = u >> lb
    e = lut[u & (2**lb - 1)] >> ip.clamp(max=62)
    e = torch.where(ip > p, torch.zeros_like(e), e)
    E = e.sum(-1, keepdim=True)
    O = exact_mm_limbs(e, vi, 3, 2)
    return ((O.float() / E.float()) * vs).cpu()


def add_edge_cases(q, k, v):
    """All-zero rows, subnormal entries, tiny and large rows."""
    q[0, 3] = 0.0
    k[1, 7] = 0.0
    v[2, 11] = 0.0
    k[3, 20, :8] = 1e-40          # subnormal fp32
    q[4, 30] *= 1e-30             # tiny but normal (below ROW_MIN)
    k[5, 40] *= 1e4               # large row
    v[6, 50, :4] = 1e-41


def main():
    torch.manual_seed(0)
    h, n, d = 8, 512, 128
    bits = (16, 16, 16)
    scale = torch.ones(d)
    scale[torch.randperm(d)[:4]] = 30.0  # outlier channels, as in real K
    q = torch.randn(h, n, d) * 2.0
    k = torch.randn(h, n, d) * scale
    v = torch.randn(h, n, d)
    add_edge_cases(q, k, v)
    vs = v.abs().amax(1, keepdim=True) * 2.0 / ia.qmax(bits[2])
    scaling = d ** -0.5

    cpu, _ = ia.int_attention(q, k, v, scaling, 0, vs, bits)
    gpu = gpu_int_attention(q.to(DEV), k.to(DEV), v.to(DEV), scaling, vs.to(DEV), bits)
    flt = ia.float_attention(q, k, v, scaling, 0)
    flt_gpu = ia.float_attention(q.to(DEV), k.to(DEV), v.to(DEV), scaling, 0).cpu()

    def bits_equal(a, b):
        return (a.view(torch.int32) == b.view(torch.int32)).float().mean().item()

    print(f"integer attention  CPU vs MPS: {bits_equal(cpu, gpu) * 100:.4f}% of outputs "
          f"bit-identical, max |diff| {(cpu - gpu).abs().max().item():.3g}")
    print(f"float   attention  CPU vs MPS: {bits_equal(flt, flt_gpu) * 100:.4f}% of outputs "
          f"bit-identical, max |diff| {(flt - flt_gpu).abs().max().item():.3g}")
    print(f"integer vs float (accuracy):   max |diff| {(cpu - flt).abs().max().item():.3g}")


# ---- Option A on the GPU, using only bit operations a kernel would use.


def exp2i(k):
    """2^k as fp32 for integer tensor k in the normal range (bit construction)."""
    return ((k + 127).int() << 23).view(torch.float32)


def frexp_exp(x):
    """frexp exponent of positive normal fp32 x (= floor(log2 x) + 1)."""
    return ((x.view(torch.int32) >> 23) & 0xFF) - 126


def gpu_pow2_rows(x, w):
    k = frexp_exp(x.abs().amax(-1, keepdim=True).clamp(min=ia.ROW_MIN)) - w + 1
    return torch.round(x * exp2i(-k)).clamp_(-ia.qmax(w), ia.qmax(w)).long(), k


def gpu_exact_attention(q, k, v, scaling, kv, w=24, p=24):
    q, k, v = (x.to(torch.bfloat16).float() for x in (q, k, v))
    n = q.shape[1]
    qi, kq = gpu_pow2_rows(q, w)
    ki, kk = gpu_pow2_rows(k, w)
    vi = torch.round(v * exp2i(-kv)).clamp_(-ia.qmax(w), ia.qmax(w)).long()
    S = exact_mm_limbs(qi, ki.transpose(1, 2).contiguous(), 3, 3)
    L = (S.float() * exp2i(kq + kk.transpose(1, 2))) * scaling
    pos = torch.arange(n, device=DEV)
    L = L.masked_fill(~(pos[None, :] <= pos[:, None]), -math.inf)
    lut = ia.make_exp_lut(p).to(DEV)
    e = ia.exp_fixed(L.amax(-1, keepdim=True) - L, lut, p)
    E = e.sum(-1, keepdim=True)
    O = exact_mm_limbs(e, vi, 4, 3)
    return ((O.float() / E.float()) * exp2i(kv)).cpu()


def main_exact():
    torch.manual_seed(1)
    h, n, d = 8, 512, 128
    scale = torch.ones(d)
    scale[torch.randperm(d)[:4]] = 30.0
    q = torch.randn(h, n, d) * 2.0
    k = torch.randn(h, n, d) * scale
    v = torch.randn(h, n, d)
    add_edge_cases(q, k, v)
    kv = ia.pow2_static(v.abs().amax(1, keepdim=True) * 2.0, 24)
    scaling = d ** -0.5
    cpu, _ = ia.exact_attention(q, k, v, scaling, 0, kv)
    gpu = gpu_exact_attention(q.to(DEV), k.to(DEV), v.to(DEV), scaling, kv.to(DEV))
    same = (cpu.view(torch.int32) == gpu.view(torch.int32)).float().mean().item()
    print(f"Option A (bf16 + exact sums) CPU vs MPS: {same * 100:.4f}% bit-identical, "
          f"max |diff| {(cpu - gpu).abs().max().item():.3g}")
    stock = ia.stock_bf16_attention(q, k, v, scaling, 0)
    print(f"Option A vs stock-bf16 numerics: max |diff| {(cpu - stock).abs().max().item():.3g}")


if __name__ == "__main__":
    main()
    main_exact()
