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
        The core insight is that public weights permit audit rather than proof-system-heavy verification. VeriLM deploys as a sidecar-style audit layer over existing serving stacks. In the default deployment path, it leaves model weights unchanged, requires no kernel modification, and shifts the main cost to rare audits rather than the normal token-generation path. Each response carries a 100-byte receipt; only audited responses open traces. The verification architecture has five layers: (1) exact shell verification --- cryptographic Freivalds checks on all INT8 weight matmuls plus exact recomputation of requantization, RoPE, RMSNorm, and SiLU; (2) KV provenance --- hash-committed per-token $K,V$ history with exact commitment binding and statistical correctness from sampled shell checks on earlier tokens; (3) cross-layer consistency --- opening multiple layers on the same token creates algebraic coupling through the residual stream, forcing any fake attention to stay consistent across all opened layers; (4) attention replay --- the verifier recomputes attention from shell-verified Q and committed prefix K,V in FP64, quantizes the result, and compares against the committed output; and (5) statistical coverage on unopened tokens and layers. This decomposition gives exact verification on the shell (model identity is caught unconditionally, $lt.eq 1\/2^(32)$), statistical provenance on prefix state, and approximate replay on the attention interior --- limited by FP16#sym.arrow.l.r{}FP64 mismatch and the KV sampling boundary. Because receipts are small, VeriLM supports continuous random auditing at low amortized cost, keeping providers under persistent technical accountability.
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

Existing approaches to verifiable inference fall into two categories. Cryptographic proof systems and specialized verifiable-inference systems such as SafetyNets @ghodsi2017safetynets and zkCNN @liu2021zkcnn can prove arbitrary or restricted computations, but impose overhead that makes real-time LLM inference impractical. Trusted execution environments (TEEs) provide attestation @birkholz2023rats but require hardware trust assumptions and limit deployment flexibility.

VeriLM takes a different approach: a lightweight sidecar protocol that runs alongside unmodified GPU inference. The key insight is that INT8 quantized inference consists of operations that are either deterministic or canonically recomputable, and can be verified exactly. The only non-deterministic component is FP16/BF16 attention, which the protocol constrains from both sides without requiring exact reproduction.

The protocol provides five layers of verification with different guarantee types, ranging from exact cryptographic checks on weight matrices to statistical sampling of prefix history.

= System Model

== Threat Model

The adversary controls the inference server. They may swap model weights, modify attention computation, fabricate intermediate activations, or lie about earlier context. The adversary does not know which responses will be audited or which tokens within them will be challenged. In the intended deployment, the client's verifier software automatically and randomly audits a small fraction of responses (e.g., 1--5%). Because every response is committed before the provider knows whether it will be challenged, the provider must behave as if any response could be audited.

The client holds a verifier key ($tilde 25$ MB for Llama 70B) derived from the public model weights and a secret random vector. The client audits responses directly --- no trusted third party sees the conversation. The attester / verifier / relying-party terminology here follows standard remote-attestation usage @birkholz2023rats.

VeriLM provides interactive, client-held verification. The client who holds the verifier key checks the computation directly. The protocol does not produce a transferable proof that third parties can verify offline --- this is a deliberate tradeoff for low overhead. Audit openings contain intermediate activations from which the conversation content can be reconstructed; this is why the client audits themselves rather than delegating to a third party.

== Design Principles

VeriLM is designed around three constraints:

+ *No serving-path changes.* The provider runs unmodified GPU inference. The only addition is a tracing layer that captures intermediates alongside normal execution.
+ *Minimal per-response overhead.* Each response carries a 100-byte receipt (two Merkle roots, a deployment manifest hash, and a token count). Most responses are never audited.
+ *Client-side verification.* The client performs all verification using CPU-feasible operations (dot products, hash checks, and --- for attention replay --- matrix arithmetic scaling as $O(n^2)$ in sequence length).

= Verification Layers

VeriLM's verification is structured in five layers. This section defines the taxonomy and key terms; the full procedure is specified in @sec-protocol.

The *shell* is the non-attention path through each transformer layer: seven INT8 weight matrix multiplications ($W_q$, $W_k$, $W_v$, $W_o$, $W_"gate"$, $W_"up"$, $W_"down"$), *requantization bridges* (the $"i32" arrow.r "i8"$ conversions between consecutive matmul stages), RoPE, RMSNorm, and SiLU. The shell includes nonlinear operations (RMSNorm, SiLU), but all shell operations are deterministic or canonically recomputable. Its exactness is a composition: cryptographic checks (Freivalds @freivalds1979) on the weight matmuls, plus deterministic or canonical recomputation on the bridge operations.

Attention ($Q K^T$, softmax, $alpha V$) is computed in FP16/BF16, which is not bit-reproducible across hardware --- this is the only non-exact component. The five layers are:

+ *Shell verification (exact).* Cryptographic Freivalds checks ($lt.eq 1\/2^(32)$ false-accept) on all weight matmuls, plus deterministic recomputation of requantization bridges, RoPE, RMSNorm, and SiLU.
+ *KV provenance (statistical).* Merkle-committed per-token K,V history; sampled positions are shell-verified. Binding is exact; correctness of unsampled positions depends on sampling rate.
+ *Cross-layer consistency (structural).* Opening multiple layers on the same token creates algebraic coupling through the residual stream --- fake attention must stay consistent across all opened layers.
+ *Attention replay (approximate).* The verifier recomputes attention from shell-verified Q and committed prefix K,V in FP64, quantizes to INT8, and compares. Limited by FP16$arrow.l.r$FP64 mismatch.
+ *Unopened positions (none).* No direct verification. Deterrence from the provider not knowing which responses will be audited or which tokens challenged.

= The Protocol <sec-protocol>

== Phase 0: Setup

*Public identity.* A Merkle root @merkle1987 $R_W$ is computed over all weight matrices of the published checkpoint. This root is the model's public identity --- anyone can recompute it from the published weights.

*Verifier key generation.* Each verifier independently generates, for each of the 7 matrix types ($W_q$, $W_k$, $W_v$, $W_o$, $W_"gate"$, $W_"up"$, $W_"down"$), a secret random vector $r_j$. For each layer $i$, the verifier precomputes $v_j^((i)) = r_j^T times W_j^((i))$ --- one matrix-vector multiply per matrix per layer, performed once. The resulting verifier key is $tilde 25$ MB for Llama 70B. After precomputation, the verifier deletes the weights. The $r_j$ vectors are the verifier's secret; if the prover learns them, it can forge passing checks.

== Phase 1: Commitment

The provider runs inference normally. The serving path is unchanged except for a tracing layer that captures intermediates alongside execution. For every token at every layer (both prefill and decode), the provider records:

- The INT8 input to each of the 7 weight matrices
- The INT32 accumulator output ($W_"i8" times x_"i8"$, exact)
- The requantized INT8 values at each bridge
- The post-attention INT8 output
- The per-tensor quantization scales at each requantization bridge

After generation completes, the provider builds two Merkle trees @merkle1987:

- $R_T$ (trace tree): over all intermediates at all tokens. The provider cannot change any activation after committing.
- $R_"KV"$ (KV tree): over per-token $K, V$ state across all layers. The provider cannot retroactively rewrite earlier tokens' context.

A deployment manifest binds everything outside the forward pass:

$ M = H(&"tokenizer_hash" || R_W || "quant_hash" \
  || &"sampling_params" || "eos_policy" \
  || &"system_prompt_hash") $

The provider returns the response plus a 100-byte receipt ($R_T$, $R_"KV"$, $M$, $N$). Most responses are never audited. The receipt is the only overhead on the normal serving path.

== Phase 2: Verification

The client decides to audit a response. The provider does not know which responses will be challenged or which tokens within them will be opened.

=== Step 1: Challenge

The verifier selects $c$ random token positions from the $N$-token response and, for each, chooses $ell$ layers to open. Opening multiple layers on the same token is preferred --- this enables cross-layer consistency checks (Step 4).

=== Step 2: Shell Verification

For each challenged token $t$ at each opened layer $i$, the provider opens the INT8 inputs $x_"i8"$ and INT32 accumulators $z_"i32"$ for all 7 weight matrices, with Merkle proofs against $R_T$. The verifier performs four checks:

+ *Freivalds on each weight matrix.* For each of the 7 matrices, the verifier checks $v_j^((i)) dot x eq.quest r_j^T dot z$. Each check is two dot products --- $O(n)$. If the prover used wrong weights, false-accept probability is $lt.eq 1\/2^(32)$ per matrix.

+ *Requantization bridges (exact).* The verifier recomputes the $"i32" arrow.r "i8"$ conversion elementwise from the opened accumulators and quantization scales. This is deterministic integer arithmetic. Without this check, the prover could pass Freivalds (correct $r dot z$) but feed fabricated $"i8"$ values into the next stage.

+ *SiLU (exact).* INT8 inputs take 256 possible values. The verifier checks every element against a precomputed 256-entry lookup table.

+ *RoPE and RMSNorm.* RoPE is recomputed from the position index --- deterministic integer-scaled arithmetic. RMSNorm is canonically recomputed from the opened residual stream values; the result is verified in the quantized output space (the recomputed value must requantize to the same $"i8"$ as the committed value).

After this step, the verifier has trusted $Q_t$, $K_t$, $V_t$ (as requantized INT8 values with verified RoPE) for the challenged token.

=== Step 3: KV Provenance

Attention at token $t$ requires the full prefix $K_(1..t)$, $V_(1..t)$. Shell verification gives trusted values for the challenged token but not the prefix.

The provider opens KV values for all prefix positions with Merkle proofs against $R_"KV"$. The verifier then:

+ Verifies the Merkle proofs --- confirming these are the values the prover committed in Phase 1.
+ Randomly samples $k$ earlier token positions.
+ Runs full shell verification on each sampled position (Freivalds on all 7 matrices, requantization, RoPE, RMSNorm, SiLU).
+ Compares the shell-verified $K$, $V$ from those sampled positions against the values opened from $R_"KV"$.

If the prover tampered with $m$ out of $n$ prefix positions and the verifier samples $k$:

$ P("catch") = 1 - (1 - m\/n)^k $

Commitment binding is exact (hash collision resistance). Correctness of unsampled positions is statistical --- the prover risks the verifier sampling any corrupted position.

=== Step 4: Cross-Layer Consistency

When the verifier opens multiple layers on the same token, fake attention at layer $L$ must produce a post-attention output that, after requantization and $W_o$, feeds into layer $L+1$'s RMSNorm consistently with the committed trace. This creates coupled algebraic constraints across layers --- much harder to satisfy than manipulating a single layer in isolation. The more layers opened, the tighter the constraint. In full-audit mode (all layers opened), the adversary must produce fake attention at every layer that stays mutually consistent through the entire residual stream.

=== Step 5: Attention Replay

This step combines the trusted $Q_t$ from Step 2 with the commitment-verified prefix $K_(1..t)$, $V_(1..t)$ from Step 3. The verifier recomputes attention independently in FP64. Since the KV cache stores post-RoPE values, the verifier dequantizes directly and computes:

+ Scores: $Q_t dot K_j \/ sqrt(d)$ for all prefix positions $j$
+ Softmax over scores
+ Weighted sum: $sum alpha_j times V_j$

The verifier quantizes the result to INT8 using the quantization scales opened from the trace (committed in $R_T$) and compares element-wise against the committed post-attention output. A response passes if at least a fraction $tau$ of INT8 elements agree; $tau$ must be calibrated empirically by measuring the FP16$arrow.l.r$FP64 requantization corridor on real model activations (see @sec-attention-gap).

This pins the attention output to a specific computation over the committed values. Without it, the adversary could fabricate a plausible-looking but wrong attention output --- a local-window approximation, suppressed attention to specific positions, or a cheaper model's attention pattern --- as long as it satisfied cross-layer consistency.

*Limitations.* The replay proves consistency with the _committed_ prefix, not necessarily with true execution at every earlier token. Prefix KV values are commitment-verified (they match $R_"KV"$) but only statistically anchored to real computation via the sampled shell checks in Step 3. The replay also cannot match the GPU's FP16 attention exactly --- the verifier's FP64 reference will differ slightly near quantization bucket boundaries.

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

= The Attention Gap <sec-attention-gap>

The shell is exactly verifiable because its operations are deterministic or canonically recomputable: INT8 matmuls are checked cryptographically via Freivalds, while requantization, RoPE, SiLU, and RMSNorm are verified by exact or canonical recomputation. Attention ($Q K^T$, softmax, $alpha V$) is computed in FP16/BF16, which is not bit-reproducible across hardware.

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

= Related Work

The most direct alternative to VeriLM is proving inference in zero knowledge. Earlier verifiable-inference systems such as SafetyNets @ghodsi2017safetynets and later zk approaches such as zkCNN @liu2021zkcnn show that inference correctness can be proved by reducing neural networks to arithmetic-circuit-style verification. However, current ZK systems still impose large prover overheads. For a model that costs dollars per response to serve, this translates to hundreds or thousands of dollars per proved response --- prohibitive for production inference.

Related cryptographic ML systems such as Delphi @mishra2020delphi address a different problem: privacy-preserving neural inference, where neither party should reveal its inputs or model. VeriLM instead targets model-provenance auditing for open weights, where the model is public and the goal is to verify that the claimed weights were actually used.

VeriLM makes the opposite tradeoff: verification is interactive (the provider must respond to challenges) and non-transferable (only the verifier who holds the key can check). In exchange, the normal serving path is unchanged and the per-response overhead is a 100-byte receipt. The expensive part --- verification --- only occurs on the small fraction of responses that are audited.

Trusted execution environments (TEEs) and remote attestation provide another alternative @birkholz2023rats @menetrey2022attestation. TEEs can attest that specific code ran on specific hardware, avoiding the overhead of proof systems, but they introduce hardware trust assumptions, limit deployment flexibility, and tie verification to a specific vendor's attestation chain. VeriLM requires no hardware trust beyond standard computation.

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

#bibliography("refs.bib", title: [References])
