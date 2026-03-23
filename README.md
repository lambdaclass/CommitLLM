# VeriLM

VeriLM is a commit-and-audit protocol for open-weight LLM inference. The provider serves responses normally with a 100-byte receipt. On demand, a verifier challenges random tokens and cryptographically checks the computation against the public model weights.

A provider who swaps or downgrades the model is caught on the first audited token with probability ≥ 1 − 1/p per checked matrix (≤ 1/2³² for p ≥ 2³²) — this is a hard, unconditional guarantee. Manipulation within the attention interior (softmax, α@V) is directly constrained by attention replay and statistically anchored by KV provenance, but not exactly verifiable due to FP16 non-determinism; see [The Attention Gap](#the-attention-gap).

### What's solved and what isn't

**Solved — model identity:** a provider who swaps, downgrades, or re-quantizes the model is caught with probability ≥ 1 − 1/p per checked matrix (≤ 1/2³² for p ≥ 2³²) on a single audit. This covers the most common economic incentive for cheating (serving a cheaper model while charging for an expensive one). The guarantee is unconditional — no statistical sampling, no threshold calibration.

**Nearly solved — attention correctness:** the verifier independently recomputes attention from shell-verified inputs and compares the output. Gross manipulation (skipped attention, local-window approximation, suppressed context) is caught. The remaining adversarial freedom is bounded by FP16↔FP64 rounding disagreement within the INT8 quantization corridor — a margin that is expected to be very small but has not yet been measured empirically.

**Statistical — prefix history:** the KV cache for earlier tokens is Merkle-committed and spot-checked. Commitment binding is exact; correctness of unsampled positions depends on sampling rate.

### Irreducible gaps

For VeriLM as designed — a sidecar commit-and-audit protocol over unmodified GPU inference — the remaining non-exactness comes from two sources only:

1. **Approximate replay of native FP16/BF16 attention.** The GPU computes attention in FP16/BF16, which is not bit-reproducible across hardware or even across runs. The verifier's FP64 reference will always differ slightly from the GPU's result. This bounds the precision of attention replay and means the protocol can constrain — but never exactly verify — the attention interior. The requantization corridor (how many INT8 elements disagree between FP16 and FP64) is the concrete manifestation of this gap. Within this sidecar design, no fix exists without replacing the serving computation with a deterministic canonical attention kernel or a stronger proof target.

2. **Statistical anchoring of prefix KV via sampling.** The verifier can only shell-verify a subset of prefix positions. Merkle binding is exact (unsampled positions match `R_KV`), but correctness of unsampled positions is statistical under the current commit-and-open design. Detection probability improves with more sampling (`P(catch) = 1 − (1 − f/n)^k`), but it is always statistical, never exact. Batch Freivalds (see [The Attention Gap](#the-attention-gap)) could make weight correctness exact at all prefix positions, but input correctness would remain statistical. A stronger proof system could eliminate the gap entirely, but not at the cost point of this sidecar design.

Everything else in the protocol is either exact (shell verification) or derives its limitations from one of these two sources. The protocol can tighten both — higher-precision GPU attention narrows gap 1, more aggressive sampling narrows gap 2 — but neither reaches zero within the current design.

The protocol has five verification layers, each with a different guarantee type:

| Layer | Type | What it does |
|---|---|---|
| **Shell verification** | **Exact** | Cryptographic verification of all INT8 weight matmuls (Freivalds) plus exact recomputation of requantization, RoPE, and SiLU. RMSNorm is verified by canonical recomputation in the quantized output space. The only layer with hard mathematical guarantees. |
| **KV provenance** | **Statistical** | Merkle-committed per-token `K,V` history with sampled earlier-token shell checks as anchors. Commitment binding is exact; correctness of unsampled positions depends on sampling rate. |
| **Cross-layer consistency** | **Structural** | Opening multiple layers on the same token creates algebraic coupling through the residual stream — fake attention must stay consistent across all opened layers. More layers opened = tighter constraint. |
| **Attention replay** | **Approximate** | The verifier recomputes attention from shell-verified Q and committed prefix K,V in FP64, quantizes the result, and compares against the committed post-attention output. Direct replay-and-compare; limited by FP16↔FP64 mismatch and the KV sampling boundary. |
| **Unopened tokens/layers** | **None** | No verification. Coverage is purely statistical — the provider doesn't know which tokens will be challenged. |

## What VeriLM Verifies

VeriLM is a pay-on-audit protocol:

- the provider serves responses normally
- each response carries a small receipt
- only audited responses open traces
- the verifier checks the opened computation

It is not a full zk proof of arbitrary inference. The protocol does not claim exact reproduction of native GPU FP16/BF16 attention semantics. See [The Attention Gap](#the-attention-gap).

## What Gets Verified at Each Step

Here is the per-layer computation and what verifies each step:

```
Input embedding               → table lookup (exact)
For each layer:
  RMSNorm                     → canonical recomputation
  W_q, W_k, W_v  (INT8 matmul)→ Freivalds
  Requantize i32→i8           → exact recomputation
  RoPE on Q, K                → exact recomputation from position index
  Attention (Q@K^T, softmax, α@V) → attention replay + cross-layer consistency (see The Attention Gap)
  W_o  (INT8 matmul)          → Freivalds
  Requantize i32→i8           → exact recomputation
  Residual add                → exact
  RMSNorm                     → canonical recomputation
  W_gate, W_up  (INT8 matmul) → Freivalds
  Requantize i32→i8           → exact recomputation
  SiLU(gate) ⊙ up             → 256-entry LUT + exact elementwise multiply
  W_down  (INT8 matmul)       → Freivalds
  Requantize i32→i8           → exact recomputation
  Residual add                → exact
Final RMSNorm                  → canonical recomputation
LM head  (weight matmul)       → Freivalds
```

Every operation is either checked exactly on opened traces, constrained by attention replay, cross-layer consistency, and KV provenance, or covered statistically through sampling.

## The Full Protocol

### Phase 0 — Setup (once per model)

Public. Compute a Merkle root `R_W` over all weight matrices. This is the model's identity — anyone can recompute it from the published weights on HuggingFace.

Verifier. Each verifier independently:

1. Fix a prime `p ≥ 2³²`. All Freivalds checks operate modulo `p`: INT8 inputs and INT32 accumulators are lifted into the finite field F_p = Z/pZ.
2. For each of the 7 matrix types (Q, K, V, O, gate, up, down), sample a secret random vector `r_j` uniformly from F_p^m (where `m` is the output dimension of the weight matrix).
3. For each layer `i`, precompute `v_j^(i) = r_j^T × W_j^(i) mod p`. This is one matrix-vector multiply per matrix per layer — done once.
4. Store the verifier key: ~25 MB for Llama 70B.
5. Delete the weights. The verifier never needs them again.

The `r_j` vectors are the verifier's secret. If the prover learns them, they can forge passing checks.

### Phase 1 — Commitment (every response)

The server runs inference normally. No changes to weights or GPU kernels. The serving path is unchanged except for a tracing layer that captures intermediates (inputs, accumulators, post-attention outputs, and quantization scales) alongside normal execution.

While generating, it captures intermediates for every token at every layer:

- The INT8 input to each weight matrix
- The INT32 accumulator output (the exact result of `W_i8 @ x_i8`)
- The requantized INT8 values at each bridge
- The post-attention output
- The per-tensor quantization scales at each requantization bridge (needed by the verifier to dequantize for attention replay and to verify requantization)

After generation completes, the server builds two separate Merkle trees. The trees are separate because they serve different access patterns: shell verification opens a few positions from `R_T`, while KV provenance opens the full prefix from `R_KV`. A single tree would force opening all intermediates at every prefix position just to extract the KV values.

| Tree | Over what | Binds |
|---|---|---|
| `R_T` (trace) | All intermediates at all tokens (inputs, accumulators, nonlinear outputs, embeddings, logits) | Every activation — the prover can't change any intermediate after committing |
| `R_KV` | Per-token KV state across all layers | The prefix history — the prover can't retroactively rewrite earlier tokens' `K,V`. Allows efficient opening of the full prefix KV without opening the complete trace at every position |

The server also computes a deployment manifest binding everything outside the forward pass:

`M = H(tokenizer_hash ‖ R_W ‖ quant_hash ‖ sampling_params ‖ eos_policy ‖ system_prompt_hash)`

The server returns: the response + a **100-byte receipt** (`R_T`, `R_KV`, `M`, `N`).

That's it. Most responses are never audited. The receipt is the only overhead on the normal serving path.

### Phase 2 — Verification (on demand)

The client decides to audit. The prover doesn't know which responses will be challenged or which tokens within them.

#### Step 1: Challenge

The verifier picks random token indices and, for each, a set of layers to open. Multiple layers on the same token are preferred — this enables cross-layer consistency checks.

#### Step 2: Shell verification

For the challenged token `t` at each opened layer `i`, the prover opens:

- `x_i8` (INT8 input to each of the 7 matrices)
- `z_i32` (INT32 accumulator output from each)
- Merkle proof against `R_T`

The verifier checks:

`(a)` **Freivalds on each weight matrix.**
`v_j^(i) · x ≡ r_j^T · z (mod p)`
Left side: dot product of precomputed vector with the opened input. Right side: dot product of secret vector with the opened output. Both are two dot products in F_p — `O(n)`. If the prover used the wrong weight matrix, the false-accept probability is ≤ `1/p` per matrix. The bound follows from the Schwartz–Zippel lemma: a nonzero linear form over F_p vanishes on at most a `1/p` fraction of inputs. With `p ≥ 2³²`, this gives ≤ `1/2³²`.

This runs on all 7 matrices at the opened layers. Any corruption in any matrix is caught independently.

`(b)` **Requantization bridge (exact).** The verifier recomputes the `i32 → i8` conversion elementwise from the opened `i32` accumulators. This is deterministic integer arithmetic — the verifier gets the exact same `i8` values. Without this, the prover could pass Freivalds (correct `r · z`) but feed fabricated `i8` values into the next stage.

`(c)` **SiLU (exact).** INT8 inputs take 256 possible values. The verifier checks every element against a precomputed 256-entry lookup table.

`(d)` **RoPE (exact) and RMSNorm (canonical recomputation).** The verifier recomputes RoPE from the position index — this is deterministic integer-scaled arithmetic. RMSNorm is recomputed from the opened residual stream values using a canonical formula; the result is verified in the quantized output space (the recomputed value must requantize to the same `i8` as the committed value).

Result: the verifier has verified the full shell for the challenged token. All 7 weight matrices (W_q, W_k, W_v, W_o, W_gate, W_up, W_down) are checked by Freivalds; all requantization bridges, RMSNorm, RoPE, and SiLU are recomputed exactly. Concretely, for each opened layer:

- Freivalds confirms all 7 matrices produced the opened `i32` accumulators
- Exact recomputation confirms every `i32 → i8` requantization bridge
- Canonical recomputation confirms RMSNorm (verified in quantized output space) and RoPE (exact)

The verifier now has trusted `Q_t`, `K_t`, `V_t` (as requantized `i8` values with verified RoPE) for the challenged token.

#### Step 3: KV provenance

Attention at token `t` requires the full prefix `K` and `V` from all earlier tokens. The shell check gives trusted `K_t`, `V_t` for the challenged token but not the prefix.

The prover opens:

- KV values for all prefix positions (`K_1..K_t`, `V_1..V_t`)
- Merkle proofs against `R_KV`

The verifier then:

1. Verifies the Merkle proofs — confirms these are the values the prover committed to in Phase 1.
2. Randomly selects some earlier token positions (say, `k` of them).
3. Runs full shell verification on each of those earlier tokens — Freivalds on all 7 matrices + exact requantization + RoPE + RMSNorm + SiLU.
4. Compares the shell-verified `K`, `V` from those earlier tokens against what was opened from `R_KV`.

If they match: those sampled positions are anchored to real verified computation. If the prover lied about `K,V` at any position, they risk the verifier sampling that exact position.

Detection probability: if the prover tampered with `f` out of `n` prefix positions, and the verifier samples `k`:

`P(catch) = 1 − (1 − f/n)^k`

Commitment binding is exact (collision resistance of the hash). Correctness of the committed values is statistical — depends on the sampling rate.

#### Step 4: Cross-layer consistency

When the verifier opens multiple layers on the same challenged token, the adversary's fake attention at layer L must produce a post-attention output that, after requantization and W_o, feeds into layer L+1's RMSNorm and produces a residual stream consistent with the layer L+1 trace. This creates coupled constraints across layers — much harder to satisfy than manipulating a single layer in isolation.

The more layers opened on the same token, the tighter the constraint. In full-audit mode (all layers opened), the adversary must produce fake attention at every layer that stays mutually consistent through the entire residual stream.

#### Step 5: Attention replay

For challenged tokens with opened KV prefix, the verifier recomputes attention independently. The shell-verified `Q_t` and the committed prefix `K_1..K_t`, `V_1..V_t` (opened in Step 3) already have RoPE applied — the KV cache stores post-RoPE values — so the verifier dequantizes them directly to FP64 and computes:

1. Scores: `Q_t · K_j / √d` for all prefix positions `j`
2. Softmax over scores
3. Weighted sum: `Σ α_j × V_j`

The verifier quantizes the result to INT8 using the quantization scales opened from the trace (committed in `R_T` alongside the other intermediates) and compares element-wise against the committed post-attention output (the `W_o` input opened in Step 2).

This is a direct consistency check: the committed attention output must be consistent with the committed Q, K, V inputs. Without it, the adversary could fabricate a plausible-looking but wrong attention output — for example, a local-window approximation that ignores distant context, suppressed attention to specific positions, or a cheaper model's attention pattern — as long as it satisfied cross-layer consistency. With the replay, the output is pinned to a specific computation over the committed values.

**Limitations.** The replay proves consistency with the *committed* prefix, not necessarily with the true execution at every earlier token. The prefix K, V values are commitment-verified (they match `R_KV`) but only statistically anchored to real computation via the sampled shell checks in Step 3. If the prover fabricated K, V at unsampled prefix positions, the replay proves consistency with those fabricated values, not correctness. The replay also cannot match the GPU's FP16 attention exactly — the verifier's FP64 reference will differ slightly, and some INT8 elements may legitimately disagree near quantization bucket boundaries. The acceptance threshold must be calibrated empirically; see [The Attention Gap](#the-attention-gap).

**Cost.** Attention replay is `O(n × h × d)` per layer, where `n` is sequence length, `h` is the number of heads, and `d` is head dimension — quadratic in sequence length per token. For short sequences and partial-layer audits, it adds modest verifier CPU time. For long sequences with many opened layers, it becomes the dominant verification cost and may exceed the shell verification time by an order of magnitude or more. The verifier can trade off by opening fewer layers on long sequences or sampling a subset of attention heads. Concrete timings depend on hardware and implementation; they should be measured rather than assumed.

### Composition

Model identity (which weights were used) is verified exactly — a single audit catches any swap or downgrade. Attention correctness is constrained but not exact.

| Layer | What it guarantees | Guarantee type |
|---|---|---|
| Shell verification | Correct weights (model identity), requantization, RoPE, SiLU; RMSNorm (canonical, verified in quantized output space) | **Exact.** Cryptographic (≤ `1/2³²`) + deterministic recomputation. Catches model swaps unconditionally. |
| KV provenance | Committed KV matches real computation at sampled positions | **Statistical.** Binding is exact; correctness depends on sampling rate |
| Cross-layer consistency | Fake attention must survive residual stream across opened layers | **Structural.** Algebraic coupling; more opened layers = tighter constraint |
| Attention replay | Committed attention output must be consistent with committed Q, K, V inputs | **Approximate.** Direct replay-and-compare; limited by FP16↔FP64 mismatch and KV sampling boundary |
| Unopened tokens/layers | — | **None.** |

### What Catches What

**Provider swaps or downgrades model:** wrong weights fail Freivalds on the first opened token. This is the protocol's strongest guarantee — caught immediately with cryptographic certainty (≤ 1/2³² false-accept per matrix). A provider serving Llama 8B when claiming Llama 70B, or using different quantized weights, is detected unconditionally. No statistical sampling is needed; a single audit suffices.

**Provider manipulates attention:** unlike model swaps, attention manipulation is *not* caught with certainty. The shell locks the inputs and outputs of every weight matmul — the adversary cannot change Q, K, V, or the post-attention linear path. Attention replay directly checks that the committed output is consistent with the committed Q, K, V inputs, catching gross manipulation such as local-window approximations, suppressed context positions, or substituted attention patterns. However, the replay inherits the statistical boundary of KV provenance (unsampled prefix positions are not shell-verified) and cannot match the GPU's FP16 arithmetic exactly. The remaining adversarial freedom is limited by FP16/FP64 disagreement and the width of the requantization corridor, neither of which has been measured. See [The Attention Gap](#the-attention-gap).

**Provider lies about earlier context:** committed KV can't change after `R_KV` is sent. Sampled shell checks anchor the commitment to real computation. Broader tampering increases the probability that the audit hits a corrupted position.

**Provider changes tokenizer/quantization/sampling:** the deployment manifest `M` is bound into the receipt (see Phase 1). The verifier checks it against known-good values (e.g., the published tokenizer for Llama 3, the weight Merkle root from HuggingFace).

## Who Audits

The client audits directly. In the intended deployment, the client's verifier software automatically and randomly audits a small fraction of responses — typically 1-5%. The provider commits to every response before knowing whether it will be challenged, which tokens will be opened, or which layers will be checked. This unpredictability is what creates the deterrent: the provider must behave as if any response could be audited.

Routine verification is CPU-feasible. Shell verification and KV provenance are lightweight (dot products and hash checks). Attention replay scales as O(n²) with sequence length and dominates for long contexts. The client stores the 100-byte receipt, the verifier software sends a challenge when it decides to audit, and verifies the opening with its own Freivalds key. No third party sees the conversation.

The commitment (100 bytes of hashes) is safe to share — it reveals nothing about the content. The audit opening is not — it contains activations from which the conversation can be reconstructed. This is why the client audits themselves rather than delegating to a third party.

## Provider-Side Costs

The receipt (100 bytes) and verifier key (~25 MB) describe the client's costs. The provider's costs are storage and tracing overhead.

### What the provider stores

Most intermediates in the forward pass are deterministic INT8 arithmetic — given the same inputs, they produce bit-identical outputs. The provider does not need to store them; they can be re-derived on demand.

This storage discussion is about retained per-token trace state. The deployment manifest `M` is different: it is computed and bound into the receipt at commit time, not treated as non-derivable trace state. The verifier later checks `M` against known-good deployment values.

The only non-derivable data is the post-attention output at each layer. The GPU computes attention in FP16, which is non-deterministic, and the result is requantized to INT8 before entering `W_o`. This INT8 vector cannot be re-derived because re-running attention might produce different FP16 values that cross quantization bucket boundaries.

**Stored per output token:** `attn_out_i8` (one INT8 vector of dimension `hidden_dim` per layer) plus the associated quantization scale (one float per layer).

| Model | Layers | hidden_dim | Per token |
|---|---|---|---|
| Llama 8B | 32 | 4096 | ~128 KB |
| Llama 70B | 80 | 8192 | ~640 KB |
| Llama 405B | 126 | 16384 | ~2 MB |

Everything else — all shell intermediates (matmul inputs/outputs, requantized values, RMSNorm, RoPE, SiLU), the full KV prefix, the residual stream, and the Merkle trees — is deterministically re-derivable from: input tokens + output tokens + stored `attn_out_i8`.

### Audit window

The provider must retain traces until the audit window closes. The audit window is how long the client has to decide whether to challenge a response. This is the protocol's core storage-flexibility tradeoff.

At a **short audit window (1–2 minutes)**, traces fit in a RAM ring buffer with no disk I/O:

| Model | Write rate per instance | Per DGX H100 (2 min) | % of 2 TB system RAM |
|---|---|---|---|
| Llama 70B (4× H100) | ~1.3 GB/s | ~315 GB | ~16% |

At this timescale, the provider allocates a fixed-size ring buffer in system RAM. New traces overwrite the oldest. No NVMe, no storage array, no flush to disk. The tracing overhead is a memcpy per layer per token.

Short audit windows require **automated auditing** — the client's audit decision must be programmatic (e.g., a script that randomly challenges a fraction of responses), not human-initiated. For a round-trip of receive-response → decide → send-challenge, 1–2 minutes is generous for software.

At **longer audit windows** (hours to days), traces spill to NVMe or networked storage. A 1-hour window fits on the DGX H100's built-in NVMe (~15 TB, enough for ~1.6 hours). A 24-hour window requires ~113 TB per instance — a dedicated storage tier. See `storage_analysis.py` for detailed estimates across cluster sizes.

## The Attention Gap

The shell — all 7 weight matmuls per layer, requantization, RoPE, RMSNorm, SiLU — is exactly verifiable. The matmuls are checked cryptographically via Freivalds; the bridge operations (requantization, RoPE, SiLU, RMSNorm) are verified by deterministic or canonical recomputation. The shell includes nonlinear operations (RMSNorm, SiLU), but all are exactly checkable in the quantized output space.

Attention is not. The GPU computes `Q @ K^T`, softmax, and `α @ V` in FP16/BF16, which is not bit-reproducible across hardware. There is no way to verify that the FP16 result matches a verifier-side replay without forcing the prover to change their attention implementation — which defeats the sidecar design.

The protocol constrains the attention interior from both sides:

1. The **inputs** to attention (Q, K, V) are exact — shell verification on W_q, W_k, W_v *(exact)*
2. The **output** of attention is correctly handed to the next linear operation — shell verification on W_o + requantization check *(exact)*
3. **Attention replay** recomputes attention from shell-verified Q and committed prefix K, V in FP64, quantizes the result, and compares against the committed output *(constrains, not exact)*
4. **Cross-layer consistency** forces fake attention to produce residuals that survive downstream shell verification *(constrains, not exact)*
5. **KV provenance** anchors the prefix history via sampled shell checks *(constrains, not exact)*

Points 1 and 2 are exact — they are part of the shell. Point 3 directly checks the attention computation against an independent reference, catching gross manipulation (wrong attention pattern, skipped computation, suppressed context). However, the replay inherits the statistical boundary of point 5 — the prefix K, V values are only statistically anchored, so the replay proves consistency with the committed prefix, not necessarily with the true execution at every earlier token. The GPU's FP16 softmax and `α @ V` also cannot be matched exactly by the verifier's FP64 reference — some INT8 elements may legitimately disagree near quantization bucket boundaries. The protocol is designed around this: exact verification on the shell, direct replay on the attention interior constrained by FP precision and KV sampling, and statistical coverage on the rest.

**Open empirical question: how tight is the requantization corridor?** The attention replay compares the verifier's FP64 reference output (quantized to INT8) against the provider's committed INT8 output. These will not match on every element — FP16 and FP64 arithmetic produce slightly different results, and elements near quantization bucket boundaries may cross into adjacent buckets. The acceptance threshold (what fraction of INT8 elements must agree) needs to be calibrated empirically. An experiment running attention in both FP16 and FP64 on real model activations, quantizing both outputs to INT8, and measuring the per-element agreement rate would set this threshold. High agreement (e.g., >99%) means the adversary's freedom is tightly bounded. Lower agreement means a looser threshold, giving the adversary more room.

**Possible future strengthening: score anchoring.** The prover could additionally commit pre-softmax attention scores and the verifier could spot-check `score[t,j] =? Q_t · K_j` at sampled positions. Since shell-verified Q and K are in INT8 with known scales, the dot product `Q_t · K_j` can be computed as an exact INT32 inner product, giving a high-precision reference score. This would constrain individual score entries independently of the output-level replay, but faces a limitation: the GPU computes softmax on FP16 scores, not the INT32 values the verifier checks, so the check cannot chain exactly to the softmax output. Score anchoring may add value on top of attention replay for long sequences where the output-level comparison has more room for cancellation, but this has not been demonstrated.

**Optional strengthening: batch Freivalds on prefix.** The current KV provenance step (Step 3) shell-verifies `k` sampled prefix positions. Freivalds is batchable: for a matrix `W` with `n` input-output pairs `(x_1, z_1)...(x_n, z_n)`, the verifier picks random scalars `α_1...α_n` and checks `v · (Σ αᵢ xᵢ) =? r^T · (Σ αᵢ zᵢ)`. One check, two dot products, covers all `n` positions simultaneously — if any `z_i` was computed with wrong weights, it fails with probability ≥ 1 − 1/2³². Concretely, the provider opens per-prefix inputs `x_j`, INT32 accumulator outputs `z_j`, and quantization scales for `W_k` and `W_v` at all prefix positions at the challenged layers. The verifier applies batch Freivalds, then verifies exact requantization and RoPE on each position. This upgrades KV provenance from "correct K,V derivation at sampled positions" to "exact K,V derivation from committed per-position inputs at all positions" (≤ 1/2³²). It does not prove those inputs were the true upstream hidden states — the input `x_j` at layer `L` depends on layer `L-1`'s attention output, so end-to-end correctness remains limited by the attention gap and the sampling of earlier activations. The cost is audit bandwidth: the provider must open inputs, INT32 accumulators, and Merkle proofs for all prefix positions at challenged layers. For Llama 70B at 4K context with 10 opened layers, this is ~300–400 MB per audit versus ~80 MB in the default protocol. This makes batch Freivalds better suited as an optional deep audit mode rather than the default.
