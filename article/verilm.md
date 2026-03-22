# VeriLM: Provenance for Open-Weight LLM Inference

When you get a response from an LLM, there's no way to know what produced it. Not which model ran. Not whether the output was modified between generation and delivery. Not whether the provider used the weights they claimed.

Today, LLM outputs are unsigned assertions. There is no cryptographic link between a response and the computation that produced it. VeriLM changes that: it gives any response a verifiable tie to a specific set of public model weights.

Model swap detection — catching a provider who serves 8B while charging for 70B — is one application of this, and the most obvious one. But provenance is the deeper primitive, and once you have it, the use cases multiply:

- **Regulatory compliance.** The EU AI Act (Article 15, enforcement August 2026) mandates accuracy and robustness guarantees for high-risk AI systems. Healthcare, finance, and legal applications need audit trails proving which model produced a given output. VeriLM's receipts provide exactly this: a cryptographic record tying each response to a specific model, tokenizer, quantization scheme, and serving configuration.

- **Content authentication.** As AI-generated content proliferates, the question shifts from "is this AI-generated?" to "which AI generated this, and can you prove it?" A VeriLM receipt is a verifiable claim of origin — not just "an LLM wrote this" but "Llama 70B with these specific weights produced this exact output."

- **Decentralized compute.** Networks like Gensyn, Ritual, or Bittensor run inference on untrusted nodes. The current approach is 2-3x redundant execution for consensus — run the same inference multiple times and compare. VeriLM replaces redundant execution with receipts and random audits, cutting compute costs while providing stronger guarantees.

- **Supply chain integrity.** Between the model running on a GPU and the text appearing in your application, there are proxies, gateways, caching layers, and middleware. Any of these can modify the output. A VeriLM receipt, committed at generation time, lets the end consumer verify that what they received matches what was generated — regardless of how many intermediaries touched it.

- **Contractual SLAs.** If you're paying for Llama 70B inference, you want proof you're getting it. Not a promise. Not a dashboard. A cryptographic receipt that you can independently verify with a 25 MB key and two dot products.

- **Scientific reproducibility.** If a research paper's results depend on LLM outputs, a VeriLM receipt proves which model produced them. Reviewers and replicators can verify the claim without re-running the inference.

The common thread: open-weight models have public weights, and public weights can be audited. VeriLM turns that observation into a protocol.

## The core trick: verify a huge matrix multiply with two dot products

Suppose a provider claims they computed

```
z = W @ x
```

for some public weight matrix `W`. The naive way to verify that claim is to recompute `W @ x`. For transformer layers, that's expensive enough that you might as well just rerun inference.

Freivalds' algorithm gives a much cheaper check. During setup, the verifier picks a secret random vector `r` and precomputes

```
v = r^T @ W
```

Then, at audit time, instead of recomputing the matrix multiply, the verifier checks

```
v · x  =?  r^T · z
```

If `z` really equals `W @ x`, this equality always holds. If the provider used the wrong weight matrix, it fails except with probability at most `2^-32`.

That matters because transformer inference is mostly weight matrix multiplication. Once you can audit those multiplies cheaply, you can verify the model's identity without rerunning the model.

## The key insight: public weights are an auditor's gift

Open-weight models have a property that most verifiable computation work ignores: the weights are public. Anyone can download Llama 70B from HuggingFace. Anyone can compute a fingerprint over those weights.

This means we don't need a general-purpose proof system. We don't need zkSNARKs (100-1000x prover overhead). We don't need TEEs (hardware trust assumptions). We need something much simpler: a way to check, after the fact, that the provider's computation is consistent with the public weights.

VeriLM is a commit-and-audit protocol. The provider serves responses normally. Each response carries a 100-byte receipt. Most responses are never checked. But when a client decides to audit, the provider must open their computation, and the verifier can cryptographically check it against the known weights.

Think of it like a tax audit. Everyone files returns. Most are never examined. But the possibility of an audit keeps people honest — especially when getting caught is catastrophic.

## What a transformer actually computes (and why most of it is easy to verify)

To understand VeriLM, you need to see what happens inside a transformer layer. Here's the computation for a single token at a single layer:

```
RMSNorm
W_q, W_k, W_v  (INT8 matrix multiplies)
Requantize i32 → i8
RoPE on Q, K
Attention (Q@K^T, softmax, α@V)     ← the hard part
W_o  (INT8 matrix multiply)
Requantize i32 → i8
Residual add
RMSNorm
W_gate, W_up  (INT8 matrix multiplies)
Requantize i32 → i8
SiLU(gate) ⊙ up
W_down  (INT8 matrix multiply)
Requantize i32 → i8
Residual add
```

Count the operations. There are seven weight matrix multiplications per layer. There's requantization (converting INT32 accumulator outputs back to INT8). There's RoPE (positional encoding), RMSNorm (normalization), and SiLU (activation function).

All of these are deterministic integer arithmetic. Given the same inputs, they produce bit-identical outputs on any hardware. This is the **linear shell** — and it's exactly verifiable.

Then there's attention. The GPU computes `Q @ K^T`, softmax, and `α @ V` in FP16 or BF16. Floating-point arithmetic is not bit-reproducible across hardware, or even across runs on the same hardware. This is the one part we can't verify exactly.

The crucial observation: the linear shell is where the weights live. If you want to check which model ran, you check the weight multiplications. Attention is a function of Q, K, V — which are themselves outputs of weight multiplications. So you can verify the model's identity exactly, even though you can't verify attention exactly.

## How Freivalds makes matrix verification cheap

The naive way to verify a matrix multiplication `z = W @ x` is to redo it. For Llama 70B, that means re-multiplying weight matrices with dimensions in the thousands — for every matrix, at every layer, at every token. That's just re-running inference.

Freivalds' algorithm (from 1977) does something much cleverer. During a one-time setup, the verifier picks a secret random vector `r` and precomputes `v = r^T × W` for every weight matrix. This requires having the weights once. After precomputation, the verifier stores just `v` (a vector, not a matrix) and deletes the weights forever.

At audit time, to check that `z = W @ x`, the verifier computes:

```
v · x  =?  r^T · z
```

Two dot products. O(n) instead of O(n²). If the provider used the wrong weight matrix, this check fails with probability at least 1 - 1/2³². That's one in four billion. Run it on all seven matrices at even a single layer, and a model swap is caught with overwhelming certainty.

For Llama 70B, the entire verifier key — all the precomputed vectors for all matrices at all layers — is about 25 MB. The client stores this once and can audit any response, forever, without ever touching the weights again.

## The full protocol in three phases

### Phase 0: Setup (once per model)

Someone publishes the weights (e.g., Meta puts Llama 3 on HuggingFace). Anyone can compute a Merkle root `R_W` over all weight matrices — this is the model's public identity. The verifier generates secret random vectors, precomputes the Freivalds vectors, stores the 25 MB key, and deletes the weights.

### Phase 1: Commitment (every response)

The provider runs inference normally. No changes to weights or GPU kernels. The only addition is a tracing layer that captures intermediates alongside normal execution: the INT8 inputs and INT32 outputs of each matrix multiplication, post-attention outputs, and quantization scales.

After generation completes, the provider builds two Merkle trees:

- **R_T** (trace tree): commits all intermediates at all tokens. The provider can't change any activation after committing.
- **R_KV** (KV tree): commits the per-token K,V history. The provider can't retroactively rewrite earlier tokens' context.

Plus a deployment manifest that binds the tokenizer, quantization scheme, sampling parameters, and system prompt.

The provider returns: the response + a **100-byte receipt** (R_T, R_KV, manifest hash, token count). That's it. Most responses are never audited. The receipt is the only overhead on the serving path.

### Phase 2: Verification (on demand)

The client decides to audit. The provider doesn't know which responses will be challenged, or which tokens within them.

**Step 1: Challenge.** The verifier picks random token positions and layers to open.

**Step 2: Shell verification.** For each challenged position, the provider opens the INT8 inputs and INT32 outputs with Merkle proofs. The verifier runs Freivalds on all seven weight matrices (two dot products each), recomputes every requantization bridge exactly, checks SiLU against a 256-entry lookup table, recomputes RoPE from the position index, and verifies RMSNorm in the quantized output space. If any check fails, the provider cheated.

After this step, the verifier has cryptographically verified Q, K, and V for the challenged token. These are trusted values.

**Step 3: KV provenance.** Attention at token t needs the K,V from all earlier tokens. The provider opens the full prefix from R_KV. The verifier randomly samples some earlier positions and runs full shell verification on those too, checking that the committed K,V values match real computation. If the provider lied about K,V at any position, they risk the verifier hitting that exact position. The detection probability follows a clean formula: if they tampered with m out of n positions and the verifier samples k, P(catch) = 1 − (1 − m/n)^k.

**Step 4: Cross-layer consistency.** When multiple layers are opened on the same token, fake attention at layer L must produce output that, after passing through W_o and requantization, feeds into layer L+1 consistently with the committed trace. This creates coupled algebraic constraints across layers. The more layers opened, the tighter the constraint.

**Step 5: Attention replay.** The verifier recomputes attention independently using the shell-verified Q and the committed prefix K,V, working in FP64. The result is quantized to INT8 and compared against the committed post-attention output. This catches gross manipulation — local-window approximations that ignore distant context, suppressed attention to specific positions, substituted attention patterns.

## What's solved, what's not, and why we're telling you

**Solved — model provenance.** A single audit cryptographically ties a response to a specific set of weights with certainty ≤ 1/2³². This is the foundation everything else builds on. Whether you're checking for a model swap, establishing an audit trail, or proving which model produced a scientific result, provenance is the primitive. The guarantee is unconditional. No statistical sampling, no threshold calibration. One audit, one Freivalds check, done.

**Nearly solved — attention correctness.** The verifier independently recomputes attention from shell-verified inputs and compares the output. Gross manipulation (skipped attention, local-window approximation, suppressed context) is caught. The remaining adversarial freedom is bounded by FP16/FP64 rounding disagreement within the INT8 quantization corridor — a margin that is expected to be very small but hasn't been measured empirically yet.

**Statistical — prefix history.** The KV cache for earlier tokens is Merkle-committed and spot-checked. Commitment binding is exact; correctness of unsampled positions depends on sampling rate.

We're being explicit about these boundaries because most verifiable inference work isn't. The honest decomposition is: exact verification on the linear shell, approximate replay on the attention interior, statistical coverage everywhere else. We think this matters more than claiming more than we can deliver.

## The attention gap

This is the protocol's most interesting limitation, and it's irreducible within the sidecar design.

The linear shell is exactly verifiable because INT8 arithmetic is deterministic. Attention is not. The GPU computes Q@K^T, softmax, and α@V in FP16, which is not bit-reproducible across hardware. There's no way to verify the FP16 result matches a verifier-side replay without forcing the provider to change their attention kernel — which defeats the whole point.

The protocol constrains attention from both sides. The inputs (Q, K, V) and outputs (post-W_o) are exactly verified by the shell. Attention replay checks that the committed output is consistent with the committed inputs. Cross-layer consistency forces fake attention to survive the entire residual stream.

What remains is the **requantization corridor**: the fraction of INT8 elements that disagree between FP16 and FP64 computation of the same attention operation. This is an open empirical question. If the agreement is >99% (which we expect), the adversary's freedom is tightly bounded. Measuring this corridor is the single most important next step.

## What providers pay

Most intermediates in the forward pass are deterministic — given the same inputs, they reproduce exactly. The provider doesn't need to store them.

The only thing that must be stored is the post-attention output at each layer. This is the INT8 result after requantizing the FP16 attention output, and it can't be re-derived because re-running attention might produce slightly different FP16 values that land in different quantization buckets.

| Model | Per token storage |
|---|---|
| Llama 8B | ~128 KB |
| Llama 70B | ~640 KB |
| Llama 405B | ~2 MB |

Everything else is re-derivable from: input tokens + output tokens + stored attention outputs.

With a short audit window (1-2 minutes), traces fit in a RAM ring buffer. For Llama 70B on 4x H100, that's about 315 GB — 16% of system RAM. No disk I/O required. The tracing overhead is a memcpy per layer per token.

Short audit windows require automated auditing: the client's decision to challenge must be programmatic, not human-initiated. For software making a round-trip decision, 1-2 minutes is generous.

## Why not ZK?

The obvious question: why not just prove inference in zero knowledge?

Because it's too expensive. Current zkSNARK/zkSTARK systems impose 100-1000x prover overhead. For a model that costs dollars per response to run, that means hundreds to thousands of dollars per proved response. That's not a viable product.

VeriLM's overhead is a 100-byte receipt per response and a memcpy per layer per token for tracing. The expensive part — verification — only happens on the small fraction of responses that get audited. And even then, verification is CPU-feasible for the client: dot products and hash checks for the shell, O(n²) matrix arithmetic for attention replay on longer sequences.

The tradeoff is honest: ZK gives you a proof that anyone can verify without interaction. VeriLM gives you an interactive audit that requires the provider to open their traces. But the audit establishes provenance with the same cryptographic certainty, at a fraction of the cost.

## What's next

The protocol is specified. The [paper](https://github.com/lambdaclass/verishell/releases/tag/v0.1.0) has the full details. What comes next:

1. **Measuring the requantization corridor.** Run attention in both FP16 and FP64 on real model activations, quantize both to INT8, measure per-element agreement. This sets the security parameter for attention replay.

2. **Integration with serving stacks.** VeriLM is designed to deploy as a sidecar over vLLM and llama.cpp. Building the tracing layer and receipt generation.

3. **Formalization.** We want a Lean formalization of the protocol's security properties, particularly the Freivalds soundness bound and the composition argument.

The code is at [github.com/lambdaclass/verishell](https://github.com/lambdaclass/verishell).
