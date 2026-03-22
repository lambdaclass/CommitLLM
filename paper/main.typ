#set document(
  title: "VeriLM: A Commit-and-Audit Protocol for Open-Weight LLM Inference",
  author: ("Federico Carrone", "Diego Kingston", "Mariano Nicolini", "Pedro Fontana", "Manuel Puebla"),
  date: auto,
)

#set text(font: "New Computer Modern", size: 9pt)
#set page(margin: (x: 1.6cm, y: 1.8cm), numbering: "1", columns: 2)
#set par(justify: true, leading: 0.58em, spacing: 0.8em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#set enum(indent: 0.5em)
#set list(indent: 0.5em)

#show heading.where(level: 1): it => {
  v(0.8em)
  text(size: 11pt, weight: "bold", it)
  v(0.4em)
}

#show heading.where(level: 2): it => {
  v(0.5em)
  text(size: 9.5pt, weight: "bold", it)
  v(0.3em)
}

#show heading.where(level: 3): it => {
  v(0.4em)
  text(size: 9pt, weight: "bold", style: "italic", it)
  v(0.2em)
}

// --- Title + Abstract (spans both columns) ---

#place(top + center, scope: "parent", float: true)[
  #block(width: 100%)[
    #v(0.5em)
    #align(center)[
      #text(size: 15pt, weight: "bold")[
        VeriLM: A Commit-and-Audit Protocol for\ Open-Weight LLM Inference
      ]
      #v(0.8em)
      #text(size: 10pt)[Federico Carrone, Diego Kingston, Mariano Nicolini, Pedro Fontana, Manuel Puebla \ LambdaClass]
      #v(0.3em)
      #text(size: 9pt, style: "italic")[Preprint]
      #v(0.8em)
    ]

    #pad(x: 1.5em)[
      #text(size: 8.5pt)[
        #text(weight: "bold")[Abstract — ]
        Users of open-weight LLMs currently have no technical mechanism to verify which model actually ran, or whether the output was altered by the provider or an intermediary. We present VeriLM, a system for auditing open-weight LLM inference that provides computational integrity and verifiable provenance: a verifier can confirm that a response is tied to the claimed model weights and serving configuration, detecting both silent downgrades (for example, serving 8B in place of 70B) and post-commitment tampering (the commitment scheme binds all activations, so any modification after receipt issuance is detectable).
        The core insight is that public weights permit audit rather than proof-system-heavy verification. VeriLM deploys as a sidecar-style audit layer over existing serving stacks. In the default deployment path, it leaves model weights unchanged, requires no kernel modification, and shifts the main cost to rare audits rather than the normal token-generation path. Each response carries a 100-byte receipt; only audited responses open traces. The verification architecture has five layers: (1) exact shell verification --- cryptographic Freivalds checks on all INT8 weight matmuls plus exact recomputation of requantization, RoPE, RMSNorm, and SiLU; (2) KV provenance --- hash-committed per-token $K,V$ history with exact commitment binding and statistical correctness from sampled shell checks on earlier tokens; (3) cross-layer consistency --- opening multiple layers on the same token creates algebraic coupling through the residual stream, forcing any fake attention to stay consistent across all opened layers; (4) attention replay --- the verifier recomputes attention from shell-verified Q and committed prefix K,V in FP64, quantizes the result, and compares against the committed output; and (5) statistical coverage on unopened tokens and layers. This decomposition gives exact verification on the linear shell (model identity is caught unconditionally, $lt.eq 1\/2^(32)$), statistical provenance on prefix state, and approximate replay on the attention interior --- limited by FP16#sym.arrow.l.r{}FP64 mismatch and the KV sampling boundary. Because receipts are small, VeriLM supports continuous random auditing at low amortized cost, keeping providers under persistent technical accountability.
      ]
    ]
    #v(0.5em)
    #line(length: 100%, stroke: 0.5pt + luma(180))
    #v(0.3em)
  ]
]


// =================================================================
= Introduction

Open-weight LLM inference presents a trust problem: the client sends a prompt to a provider who claims to run a specific model (e.g., Llama 70B), but the client has no way to verify that the provider actually used those weights. The economic incentive to cheat is clear --- serving a smaller or more aggressively quantized model reduces compute costs while the client pays the same price.

Existing approaches to verifiable inference fall into two categories. Cryptographic proof systems (zkSNARKs, zkSTARKs) can prove arbitrary computations but impose overhead that makes real-time LLM inference impractical. Trusted execution environments (TEEs) provide attestation but require hardware trust assumptions and limit deployment flexibility.

VeriLM takes a different approach: a lightweight sidecar protocol that runs alongside unmodified GPU inference. The key insight is that INT8 quantized inference is mostly deterministic integer arithmetic, which can be verified exactly. The only non-deterministic component is FP16/BF16 attention, which the protocol constrains from both sides without requiring exact reproduction.

The protocol provides five layers of verification with different guarantee types, ranging from exact cryptographic checks on weight matrices to statistical sampling of prefix history.

= Protocol Overview

== Threat Model

The adversary controls the inference server. They may swap model weights, modify attention computation, fabricate intermediate activations, or lie about earlier context. The adversary does not know which responses will be audited or which tokens within them will be challenged.

The client holds a verifier key ($tilde 25$ MB for Llama 70B) derived from the public model weights and a secret random vector. The client audits responses directly --- no trusted third party sees the conversation.

VeriLM provides interactive, client-held verification. The client who holds the verifier key checks the computation directly. The protocol does not produce a transferable proof that third parties can verify offline --- this is a deliberate tradeoff for low overhead. Audit openings contain intermediate activations from which the conversation content can be reconstructed; this is why the client audits themselves rather than delegating to a third party.

== Design Principles

VeriLM is designed around three constraints:

+ *No serving-path changes.* The provider runs unmodified GPU inference. The only addition is a tracing layer that captures intermediates alongside normal execution.
+ *Minimal per-response overhead.* Each response carries a 100-byte receipt (two Merkle roots, a deployment manifest hash, and a token count). Most responses are never audited.
+ *Client-side verification.* The client performs all verification using CPU-feasible operations (dot products, hash checks, and --- for attention replay --- matrix arithmetic scaling as $O(n^2)$ in sequence length).

= Verification Layers

VeriLM's verification is structured in five layers, each with a distinct guarantee type.

== Shell Verification (Exact)

The linear path through each transformer layer consists of seven INT8 weight matrix multiplications ($W_q$, $W_k$, $W_v$, $W_o$, $W_"gate"$, $W_"up"$, $W_"down"$), requantization bridges, RoPE, RMSNorm, and SiLU. All of these are deterministic and exactly verifiable.

*Freivalds' algorithm* checks each weight multiplication. During setup, the verifier precomputes $v_j^((i)) = r_j^T times W_j^((i))$ for secret random vector $r_j$. At audit time, the verifier checks:

$ v_j^((i)) dot x eq.quest r_j^T dot z $

where $x$ is the opened INT8 input and $z$ is the opened INT32 accumulator. This reduces matrix verification to two dot products. If the prover used wrong weights, false-accept probability is $lt.eq 1\/2^(32)$.

Requantization ($"i32" arrow.r "i8"$) is verified by exact recomputation. SiLU is checked against a 256-entry lookup table. RoPE is recomputed from the position index. RMSNorm is canonically recomputed and verified in the quantized output space.

== KV Provenance (Statistical)

Attention at token $t$ requires the full prefix $K_(1..t)$, $V_(1..t)$. The prover commits per-token KV state in a Merkle tree $R_"KV"$. The verifier samples $k$ earlier positions and runs full shell verification on each, comparing the result against the committed values.

If the prover tampered with $m$ out of $n$ prefix positions:

$ P("catch") = 1 - (1 - m\/n)^k $

Commitment binding is exact (hash collision resistance). Correctness of unsampled positions is statistical.

== Cross-Layer Consistency (Structural)

When multiple layers are opened on the same token, fake attention at layer $L$ must produce a post-attention output that, after requantization and $W_o$, feeds into layer $L+1$'s RMSNorm consistently with the committed trace. More opened layers create tighter algebraic coupling through the residual stream.

== Attention Replay (Approximate)

The verifier recomputes attention from shell-verified $Q_t$ and committed prefix $K$, $V$ in FP64:

+ Scores: $Q_t dot K_j \/ sqrt(d)$ for all prefix positions $j$
+ Softmax over scores
+ Weighted sum: $sum alpha_j times V_j$

The result is quantized to INT8 and compared against the committed post-attention output. This catches gross manipulation (local-window approximation, suppressed context, wrong attention pattern) but cannot match FP16 arithmetic exactly.

== Unopened Tokens and Layers

No verification. Coverage relies on the provider not knowing which tokens will be challenged.

= The Protocol

== Phase 0: Setup

A Merkle root $R_W$ is computed over all weight matrices --- the model's public identity. Each verifier generates secret random vectors $r_j$ and precomputes $v_j^((i)) = r_j^T times W_j^((i))$ for all layers. The resulting verifier key is $tilde 25$ MB for Llama 70B. After precomputation, the verifier deletes the weights.

== Phase 1: Commitment

The provider runs inference normally, capturing intermediates (INT8 inputs, INT32 accumulators, post-attention outputs, quantization scales) alongside execution. After generation, two Merkle trees are built:

- $R_T$ (trace): over all intermediates at all tokens
- $R_"KV"$: over per-token KV state across all layers

A deployment manifest binds everything outside the forward pass:

$ M = H(&"tokenizer_hash" || R_W || "quant_hash" \
  || &"sampling_params" || "eos_policy" \
  || &"system_prompt_hash") $

The provider returns the response plus a 100-byte receipt ($R_T$, $R_"KV"$, $M$, $N$).

== Phase 2: Verification

The client challenges random tokens and layers. For each opened position, the verifier runs shell verification (Freivalds on all 7 matrices, exact requantization, SiLU, RoPE, RMSNorm), KV provenance sampling, cross-layer consistency checks, and attention replay. Any failure indicates the provider deviated from the claimed model.

= Security Analysis

This section maps concrete adversarial strategies to the protocol layers that detect them.

== Model Swap or Downgrade

Wrong weights fail the Freivalds check at any opened layer. A single audit of a single token is enough to catch a provider serving different model weights, because the Freivalds check at any opened layer will fail against the public checkpoint. False-accept probability is $lt.eq 1\/2^(32)$ per matrix. This is the protocol's strongest guarantee --- it is unconditional and does not depend on statistical sampling or threshold calibration.

== Attention Manipulation

The shell locks the inputs and outputs of every weight matmul --- the adversary cannot change $Q$, $K$, $V$, or the post-attention linear path. Attention replay directly checks that the committed output is consistent with the committed $Q$, $K$, $V$ inputs, catching gross manipulation such as local-window approximations, suppressed context positions, or substituted attention patterns. However, the replay inherits the statistical boundary of KV provenance (unsampled prefix positions are not shell-verified) and cannot match the GPU's FP16 arithmetic exactly. The remaining adversarial freedom is bounded by FP16/FP64 disagreement and the width of the requantization corridor.

== Fabricated Prefix Context

Committed KV values cannot change after $R_"KV"$ is sent. The verifier samples earlier positions and runs full shell verification, anchoring the commitment to real computation. If the prover tampered with $m$ out of $n$ prefix positions and the verifier samples $k$, detection probability is $P("catch") = 1 - (1 - m\/n)^k$. Broader tampering increases the probability of hitting a corrupted position.

== Deployment Configuration Tampering

The deployment manifest $M$ binds the tokenizer, weight Merkle root, quantization scheme, sampling parameters, EOS policy, and system prompt hash into the receipt. The verifier checks $M$ against known-good values (e.g., the published tokenizer for a given model family, the weight Merkle root from the public checkpoint). Any change to the deployment configuration produces a different manifest.

= The Attention Gap

The linear shell is exactly verifiable because INT8 arithmetic is deterministic. Attention ($Q K^T$, softmax, $alpha V$) is computed in FP16/BF16, which is not bit-reproducible across hardware.

The protocol constrains the attention interior from both sides: inputs ($Q$, $K$, $V$) and outputs (post-$W_o$) are exactly verified by the shell. Attention replay directly checks consistency. Cross-layer constraints force fake attention to survive the residual stream.

The remaining adversarial freedom is bounded by the *requantization corridor* --- the fraction of INT8 elements that disagree between FP16 and FP64 computation. This is an open empirical question. High agreement ($> 99%$) means tight constraint; lower agreement means more adversarial room.

= Provider Costs

== Storage

Only post-attention INT8 outputs and the associated per-tensor quantization scales require storage (everything else is re-derivable):

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    [*Model*], [*Layers*], [*hidden_dim*], [*Per token*],
    [Llama 8B], [32], [4096], [$tilde 128$ KB],
    [Llama 70B], [80], [8192], [$tilde 640$ KB],
    [Llama 405B], [126], [16384], [$tilde 2$ MB],
  ),
  caption: [Per-token storage for non-derivable intermediates],
)

== Audit Window

With a short audit window (1--2 minutes), traces fit in a RAM ring buffer with no disk I/O. For Llama 70B on 4$times$ H100, write rate is $tilde 1.3$ GB/s, requiring $tilde 315$ GB ($tilde 16%$ of 2 TB system RAM) for a 2-minute window. Longer windows spill to NVMe or networked storage.

Short audit windows require automated auditing --- the client's audit decision must be programmatic.

= What Gets Verified

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    [*Operation*], [*Verification*],
    [Input embedding], [Table lookup (exact)],
    [$W_q, W_k, W_v$ (INT8)], [Freivalds],
    [Requantize i32 $arrow.r$ i8], [Exact recomputation],
    [RoPE on $Q$, $K$], [Exact recomputation],
    [Attention], [Replay + cross-layer],
    [$W_o$ (INT8)], [Freivalds],
    [RMSNorm], [Canonical recomputation],
    [$W_"gate"$, $W_"up"$ (INT8)], [Freivalds],
    [SiLU $dot.o$ up], [256-entry LUT + exact],
    [$W_"down"$ (INT8)], [Freivalds],
    [LM head], [Freivalds],
  ),
  caption: [Per-layer verification methods],
)

= Related Work

The most direct alternative to VeriLM is proving inference in zero knowledge. Systems based on zkSNARKs or zkSTARKs can prove arbitrary computations, producing a static, non-interactive proof that anyone can verify offline. However, current ZK systems impose 100--1000$times$ prover overhead. For a model that costs dollars per response to serve, this translates to hundreds or thousands of dollars per proved response --- prohibitive for production inference.

VeriLM makes the opposite tradeoff: verification is interactive (the provider must respond to challenges) and non-transferable (only the verifier who holds the key can check). In exchange, the normal serving path is unchanged and the per-response overhead is a 100-byte receipt. The expensive part --- verification --- only occurs on the small fraction of responses that are audited.

Trusted execution environments (TEEs) provide hardware-level attestation that specific code ran on specific hardware. TEEs avoid the overhead of proof systems but introduce hardware trust assumptions, limit deployment flexibility, and tie verification to a specific vendor's attestation chain. VeriLM requires no hardware trust beyond standard computation.

= Discussion

VeriLM solves the model identity problem with cryptographic certainty. A provider who swaps, downgrades, or re-quantizes the model is caught on a single audit. This covers the most common economic incentive for cheating.

Attention correctness is constrained but not exact --- an irreducible limitation of the sidecar design over unmodified GPU inference. Closing this gap entirely would require either deterministic attention kernels or stronger proof systems, both of which violate the design constraints.

The protocol's practical viability depends on two open questions: the empirical width of the requantization corridor, and whether the storage and bandwidth costs are acceptable to providers at scale.

= Future Directions

Two optional strengthenings could tighten the protocol's guarantees:

*Score anchoring.* The prover commits pre-softmax attention scores; the verifier spot-checks $"score"[t,j] eq.quest Q_t dot K_j$ as exact INT32 inner products. This constrains individual score entries independently of the output-level replay. The limitation is that the GPU computes softmax on FP16 scores, not the INT32 values the verifier checks, so the check cannot chain exactly to the softmax output.

*Batch Freivalds on prefix.* The verifier checks all prefix positions simultaneously using random linear combinations: pick random scalars $alpha_1 dots alpha_n$ and check $v dot (sum alpha_i x_i) eq.quest r^T dot (sum alpha_i z_i)$. One check covers all $n$ positions with false-accept probability $lt.eq 1\/2^(32)$. This upgrades KV provenance from statistical sampling to exact weight verification at all positions. Cost: $tilde 4 times$ audit bandwidth, making it better suited as an optional deep audit mode.

The most important empirical next step is measuring the requantization corridor: running attention in both FP16 and FP64 on real model activations, quantizing both outputs to INT8, and measuring per-element agreement rates. This directly sets the security parameter for attention replay.

= Conclusion

VeriLM demonstrates that meaningful verification of LLM inference is possible without modifying the serving path or requiring expensive proof systems. The protocol's honest decomposition into exact, statistical, and approximate layers allows clients to understand precisely what is and is not guaranteed. The 100-byte receipt imposes negligible overhead on normal serving; the full audit is CPU-feasible for the client.

The most important next step is empirical measurement of the requantization corridor, which will determine how tightly the protocol constrains attention manipulation in practice.
