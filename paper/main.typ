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
        Users of open-weight LLMs have no technical mechanism to verify which model actually ran or whether the output was altered. We present VeriLM, a commit-and-audit protocol for open-weight LLM inference. The core insight is that public weights permit lightweight audit rather than expensive proof systems. VeriLM deploys as a sidecar over unmodified serving stacks: each response carries a 100-byte receipt, and only audited responses open traces. The protocol provides three tiers of assurance: exact model-identity checks on opened layers ($lt.eq 1\/2^(32)$ false-accept per matrix via Freivalds on INT8 weight matmuls), statistical prefix-state provenance (Merkle-committed KV history with sampled shell verification), and approximate attention replay (FP64 recomputation compared against the committed quantized post-attention output, bounded by the requantization corridor). Because receipts are small, VeriLM supports continuous random auditing at low amortized cost.
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

The adversary controls the inference server. They may swap model weights, modify attention computation, fabricate intermediate activations, or lie about earlier context. The adversary may cheat adaptively --- choosing different strategies for different responses. The following timing and knowledge constraints apply:

+ *Commitment precedes challenge.* The provider sends the receipt ($R_T$, $R_"KV"$, $M$, $N$) before learning whether the response will be audited.
+ *Unpredictable selection.* The provider cannot predict which response, token position, or layer the verifier will challenge.
+ *Secret key.* The verifier holds $r_j$ and $v_j^((i))$ secret; the provider never observes them or the audit decision rule.
+ *Local verification.* The verifier checks locally --- no third party sees the challenge, the opening, or the conversation.

In the intended deployment, the client's verifier software automatically and randomly audits a small fraction of responses (e.g., 1--5%).

The client holds a verifier key ($tilde 25$ MB for Llama 70B) derived from the public model weights and a secret random vector. The client audits responses directly --- no trusted third party sees the conversation. The attester / verifier / relying-party terminology here follows standard remote-attestation usage @birkholz2023rats.

VeriLM provides interactive, client-held verification. The client who holds the verifier key checks the computation directly. The protocol does not produce a transferable proof that third parties can verify offline --- this is a deliberate tradeoff for low overhead. Audit openings contain intermediate activations from which the conversation content can be reconstructed; this is why the client audits themselves rather than delegating to a third party.

== Design Principles

VeriLM is designed around three constraints:

+ *No serving-path changes.* The provider runs unmodified GPU inference. The only addition is a tracing layer that captures intermediates alongside normal execution.
+ *Minimal per-response overhead.* Each response carries a 100-byte receipt (two Merkle roots, a deployment manifest hash, and a token count). Most responses are never audited.
+ *Client-side verification.* The client performs all verification using CPU-feasible operations (dot products, hash checks, and --- for attention replay --- matrix arithmetic scaling as $O(n^2)$ in sequence length).

#block(
  width: 100%,
  inset: 8pt,
  stroke: 0.5pt + luma(120),
  radius: 2pt,
)[
  *Protocol guarantees at a glance.*
  + *Exact.* Opened-layer model identity ($lt.eq 1\/2^(32)$ false-accept) and commitment binding (hash-immutable after receipt).
  + *Statistical.* Prefix KV correctness under sampling ($k$ prefix positions checked).
  + *Approximate.* Attention replay under the FP16$arrow.l.r$FP64 requantization corridor (@sec-attention-gap).
]

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    [*Symbol*], [*Meaning*],
    [$R_W$], [Merkle root over model weights (public identity)],
    [$R_T$], [Merkle root over trace intermediates],
    [$R_"KV"$], [Merkle root over per-token $K, V$ history],
    [$M$], [Deployment manifest hash],
    [$N$], [Token count in response],
    [$c, ell$], [Number of challenged tokens; number of opened layers per token],
    [$m$], [Output dimension of the checked weight matrix (length of $r_j$)],
    [$k$], [Sampled prefix positions for KV provenance],
    [$tau$], [Acceptance threshold for attention replay],
    [$p$], [Prime modulus for Freivalds checks ($p gt.eq 2^(32)$); arithmetic in $bb(F)_p$],
  ),
  caption: [Notation],
)

= Verification Layers

VeriLM's verification is structured in five layers. This section defines the taxonomy and key terms; the full procedure is specified in @sec-protocol.

The *shell* is the non-attention path through each transformer layer: seven INT8 weight matrix multiplications ($W_q$, $W_k$, $W_v$, $W_o$, $W_"gate"$, $W_"up"$, $W_"down"$), *requantization bridges* (the $"i32" arrow.r "i8"$ conversions between consecutive matmul stages), RoPE, RMSNorm, and SiLU. The shell includes nonlinear operations (RMSNorm, SiLU), but all shell operations are deterministic or canonically recomputable. Its exactness is a composition: cryptographic checks (Freivalds @freivalds1979) on the weight matmuls, plus deterministic or canonical recomputation on the bridge operations.

Attention ($Q K^T$, softmax, $alpha V$) is computed in FP16/BF16, which is not bit-reproducible across hardware --- this is the only non-exact component. The five layers are:

+ *Shell verification (exact).* Cryptographic Freivalds checks ($lt.eq 1\/2^(32)$ false-accept) on all weight matmuls, plus deterministic recomputation of requantization bridges, RoPE, RMSNorm, and SiLU.
+ *KV provenance (statistical).* Merkle-committed per-token K,V history; sampled positions are shell-verified. Binding is exact; correctness of unsampled positions depends on sampling rate.
+ *Cross-layer consistency (structural).* Opening multiple layers on the same token creates algebraic coupling through the residual stream --- fake attention must stay consistent across all opened layers.
+ *Attention replay (approximate).* The verifier recomputes attention from shell-verified Q and committed prefix K,V in FP64, quantizes to INT8, and compares. Limited by FP16$arrow.l.r$FP64 mismatch.
+ *Unopened positions (none).* No direct verification. Deterrence from the provider not knowing which responses will be audited or which tokens challenged.

@fig-forward-pass traces the full forward pass for one output token. Each operation is labeled with its verification type: Freivalds-checked weight matmuls, canonically recomputable bridge operations, and the single non-exact point --- FP16 attention. The only non-replayable provider-side state is the per-layer `attn_out_i8` and its quantization scale; everything below that point replays exactly or canonically from the stored values, the transcript, and the public weights.

#figure(
  block(width: 100%, inset: (x: 4pt, y: 6pt))[
    #let t(body) = align(right, text(style: "italic", fill: luma(80), size: 7pt)[#body])
    #let sp = table.cell(colspan: 2)[#v(1pt)]
    #table(
      columns: (1fr, auto),
      stroke: none,
      inset: (x: 2pt, y: 1.5pt),
      align: (left, right),
      raw("residual = embedding[token_id]"), t[exact lookup],
      sp,
      table.cell(colspan: 2, raw("for each layer:")),
      raw("  x_norm = RMSNorm(residual)"), t[canonical],
      raw("  x_i8, s = quantize(x_norm)"), t[exact],
      sp,
      raw("  q_i32 = W_q @ x_i8"), t[Freivalds],
      raw("  k_i32 = W_k @ x_i8"), t[Freivalds],
      raw("  v_i32 = W_v @ x_i8"), t[Freivalds],
      sp,
      raw("  q = RoPE(dequant(q_i32, ...), pos)"), t[canonical],
      raw("  k = RoPE(dequant(k_i32, ...), pos)"), t[canonical],
      raw("  v = dequant(v_i32, ...)"), t[canonical],
      sp,
      raw("  attn = softmax(q @ k^T / sqrt(d)) @ v"), t[non-exact],
      raw("  attn_out_i8, sa = quantize(attn)"), t[*STORED*],
      sp,
      raw("  o_i32 = W_o @ attn_out_i8"), t[Freivalds],
      raw("  residual += dequant(o_i32, ...)"), t[canonical],
      sp,
      raw("  x_norm = RMSNorm(residual)"), t[canonical],
      raw("  x_i8, s = quantize(x_norm)"), t[exact],
      sp,
      raw("  gate_i32 = W_gate @ x_i8"), t[Freivalds],
      raw("  up_i32   = W_up   @ x_i8"), t[Freivalds],
      raw("  x = SiLU(dequant(gate_i32)) * dequant(up_i32)"), t[canonical],
      raw("  x_i8, s = quantize(x)"), t[exact],
      raw("  down_i32 = W_down @ x_i8"), t[Freivalds],
      raw("  residual += dequant(down_i32, ...)"), t[canonical],
      sp,
      raw("x_norm = RMSNorm(residual)"), t[canonical],
      raw("x_i8, s = quantize(x_norm)"), t[exact],
      raw("logits_i32 = LM_head @ x_i8"), t[Freivalds],
      raw("token = sample(dequant(logits_i32))"), t[transcript],
    )
  ],
  caption: [Annotated forward pass for one output token. Each operation is labeled with its verification type. The only non-exact stage is FP16 attention; storing `attn_out_i8` and its scale bridges the gap.],
) <fig-forward-pass>

= The Protocol <sec-protocol>

== Phase 0: Setup

*Public identity.* A Merkle root @merkle1987 $R_W$ is computed over all weight matrices of the published checkpoint. This root is the model's public identity --- anyone can recompute it from the published weights.

*Verifier key generation.* The verifier fixes a prime $p gt.eq 2^(32)$. All Freivalds checks operate in $bb(F)_p = ZZ \/ p ZZ$: INT8 inputs and INT32 accumulators are lifted into this field. For each of the 7 matrix types ($W_q$, $W_k$, $W_v$, $W_o$, $W_"gate"$, $W_"up"$, $W_"down"$), the verifier samples a secret random vector $r_j$ uniformly from $bb(F)_p^m$. For each layer $i$, the verifier precomputes $v_j^((i)) = r_j^T W_j^((i)) mod p$ --- one matrix-vector multiply per matrix per layer, performed once. The resulting verifier key is $tilde 25$ MB for Llama 70B. After precomputation, the verifier deletes the weights. The $r_j$ vectors are the verifier's secret; if the prover learns them, it can forge passing checks.

== Phase 1: Commitment

The provider runs inference normally. The serving path is unchanged except for a tracing layer that captures intermediates alongside execution. In batched serving (e.g., vLLM with paged attention), multiple requests share GPU resources; the tracing layer captures per-request intermediates, extracting per-request activations from the batched computation without modifying the batched kernels.

For every token at every layer (both prefill and decode), the provider records:

- The INT8 input to each of the 7 weight matrices
- The INT32 accumulator output ($W_"i8" times x_"i8"$, exact)
- The requantized INT8 values at each bridge
- The post-attention INT8 output
- The per-tensor quantization scales at each requantization bridge

After generation completes, the provider builds two Merkle trees @merkle1987. The trees are separate because they serve different access patterns: shell verification (Step 2) opens a few positions from $R_T$, while KV provenance (Step 3) opens the full prefix from $R_"KV"$. A single tree would force opening all intermediates at every prefix position just to extract the KV values.

- $R_T$ (trace tree): over all intermediates at all tokens. The provider cannot change any activation after committing.
- $R_"KV"$ (KV tree): over per-token $K, V$ state across all layers. The provider cannot retroactively rewrite earlier tokens' context. This tree allows efficient opening of the full prefix KV without opening the complete trace at every position.

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

For each challenged token $t$ at each opened layer $i$, the provider opens the INT8 inputs $x_"i8"$, INT32 accumulators $z_"i32"$, per-tensor quantization scales, the residual stream values at the layer boundaries, and the post-attention output $"attn_out_i8"$ --- all with Merkle proofs against $R_T$. The verifier performs four checks:

+ *Freivalds on each weight matrix.* For each of the 7 matrices, the verifier checks $v_j^((i)) dot x equiv r_j^T dot z space (mod p)$. Each check is two dot products in $bb(F)_p$ --- $O(n)$. If the prover used wrong weights, false-accept probability is $lt.eq 1\/p$ per matrix. The bound follows from the Schwartz--Zippel lemma: a nonzero linear form over $bb(F)_p$ vanishes on at most a $1\/p$ fraction of inputs.

+ *Requantization bridges (exact).* The verifier recomputes the $"i32" arrow.r "i8"$ conversion elementwise from the opened accumulators and quantization scales. This is deterministic integer arithmetic. Without this check, the prover could pass Freivalds (correct $r dot z$) but feed fabricated $"i8"$ values into the next stage.

+ *SiLU (exact).* INT8 inputs take 256 possible values. The verifier checks every element against a precomputed 256-entry lookup table.

+ *RoPE and RMSNorm.* RoPE is recomputed from the position index --- deterministic integer-scaled arithmetic. RMSNorm is canonically recomputed from the opened residual stream values; the result is verified in the quantized output space (the recomputed value must requantize to the same $"i8"$ as the committed value).

After this step, the verifier has trusted $Q_t$, $K_t$, $V_t$ (as requantized INT8 values with verified RoPE) for the challenged token.

=== Step 3: KV Provenance

Attention at token $t$ requires the full prefix $K_(1..t)$, $V_(1..t)$. Shell verification gives trusted values for the challenged token but not the prefix.

For each opened layer $i$, the provider opens the $K$, $V$ values at all prefix positions $j < t$ with Merkle proofs against $R_"KV"$. The verifier then:

+ Verifies the Merkle proofs --- confirming these are the values the prover committed in Phase 1.
+ Randomly samples $k$ earlier token positions $j_1, dots, j_k < t$.
+ For each sampled position $j_s$, runs full shell verification at layer $i$ (Freivalds on all 7 matrices, requantization, RoPE, RMSNorm, SiLU), producing independently verified $K_(j_s)^((i))$, $V_(j_s)^((i))$.
+ Compares these shell-verified values against the corresponding entries opened from $R_"KV"$.

If the prover tampered with $m$ out of $n$ prefix positions and the verifier samples $k$:

$ P("catch") = 1 - (1 - m\/n)^k $

Commitment binding is exact (hash collision resistance). Correctness of unsampled positions is statistical --- the prover risks the verifier sampling any corrupted position.

=== Step 4: Cross-Layer Consistency

There is no separate proof object here; the consistency check comes from opening multiple layers and verifying the committed bridges between them. When the verifier opens layers $L$ and $L+1$ on the same token, fake attention at layer $L$ must produce a post-attention output that, after requantization and $W_o$, feeds into layer $L+1$'s RMSNorm consistently with the committed trace. Both sides of this boundary are shell-verified, so the adversary cannot fabricate a consistent bridge without matching the committed values exactly. The more layers opened, the tighter the constraint. In full-audit mode (all layers opened), fake attention at every layer must stay mutually consistent through the entire residual stream.

=== Step 5: Attention Replay

This step combines the trusted $Q_t$ from Step 2 with the commitment-verified prefix $K_(1..t)$, $V_(1..t)$ from Step 3. The verifier recomputes attention independently in FP64. Since the KV cache stores post-RoPE values, the verifier dequantizes directly and computes:

+ Scores: $Q_t dot K_j \/ sqrt(d)$ for all prefix positions $j$
+ Softmax over scores
+ Weighted sum: $sum alpha_j times V_j$

The verifier quantizes the result to INT8 using the quantization scales opened from the trace (committed in $R_T$) and compares element-wise against the committed post-attention output. A response passes if at least a fraction $tau$ of INT8 elements agree; $tau$ must be calibrated empirically by measuring the FP16$arrow.l.r$FP64 requantization corridor on real model activations (see @sec-attention-gap).

This pins the attention output to a specific computation over the committed values. Without it, the adversary could fabricate a plausible-looking but wrong attention output --- a local-window approximation, suppressed attention to specific positions, or a cheaper model's attention pattern --- as long as it satisfied cross-layer consistency.

*Limitations.* The replay proves consistency with the _committed_ prefix, not necessarily with true execution at every earlier token. Prefix KV values are commitment-verified (they match $R_"KV"$) but only statistically anchored to real computation via the sampled shell checks in Step 3. The replay also cannot match the GPU's FP16 attention exactly --- the verifier's FP64 reference will differ slightly near quantization bucket boundaries.

== Summary

@tab-verification-methods summarizes the verification method applied to each operation in the forward pass.

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
) <tab-verification-methods>

== Audit Walkthrough: One Token, Two Layers

Audit token $t = 47$ at layers 12 and 13 of a 70B model, sampling $k = 8$ prefix positions.

+ *Challenge.* Verifier sends $(t = 47, {12, 13})$.
+ *Shell.* Provider opens trace at both layers from $R_T$. Verifier runs Freivalds on all 7 matrices per layer (28 dot products total), recomputes all bridges. Yields trusted $Q_(47)$, $K_(47)$, $V_(47)$ at both layers.
+ *KV provenance.* Provider opens $K_(1..46)$, $V_(1..46)$ at both layers from $R_"KV"$. Verifier picks 8 random earlier positions, runs full shell verification at each, confirms committed $K$, $V$ match.
+ *Cross-layer.* Post-attention output at layer 12 feeds into layer 13's RMSNorm --- both sides shell-verified, so boundary must match exactly.
+ *Replay.* At each opened layer, the verifier recomputes attention in FP64 from shell-verified $Q_(47)$ and the commitment-verified prefix. Quantizes to INT8, compares against committed `attn_out_i8`. Passes if at least fraction $tau$ of INT8 elements agree at both layers.

== Data Lifecycle

@tab-data-lifecycle shows what data exists at each stage of the protocol.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    [*Receipt fields (100 B total)*], [*Provider retains (ring buffer)*], [*Revealed on audit*],
    [$R_T$ (trace Merkle root)], [INT8 inputs to all 7 matrices], [Merkle openings from $R_T$],
    [$R_"KV"$ (KV Merkle root)], [INT32 accumulator outputs], [Merkle openings from $R_"KV"$],
    [Manifest hash $M$], [Requantized INT8 at each bridge], [Opened intermediates at challenged positions],
    [Token count $N$], [`attn_out_i8` + quantization scales], [Full prefix $K$, $V$ at opened layers],
    [], [Per-tensor quantization scales], [],
  ),
  caption: [Data lifecycle: receipt (every response), retained state (RAM ring buffer, discarded after audit window), and audit openings (revealed only when challenged).],
) <tab-data-lifecycle>

= Security Analysis

== Security Game

We formalize model-identity soundness via an interactive game between a challenger $cal(C)$ (verifier) and an adversary $cal(A)$ (provider).

$bold("Game")^("id")_(cal(A))(lambda)$:

+ *Setup.* $cal(C)$ publishes weights $W$ and Merkle root $R_W$. $cal(C)$ samples $r_j in.rev bb(F)_p^m$ for each matrix type $j$, precomputes $v_j^((i)) = r_j^T W_j^((i)) mod p$ for every layer $i$, and stores the secret verifier key $"sk" = {r_j, v_j^((i))}$. $cal(A)$ receives $W$, $R_W$ but never observes $"sk"$.
+ *Commit.* $cal(A)$ receives a prompt, runs inference (using any strategy), and returns a response plus receipt $(R_T, R_"KV", M, N)$.
+ *Challenge.* $cal(C)$ selects $c$ token positions and $ell$ layers per token uniformly at random. $cal(A)$ learns the challenge only now.
+ *Open.* $cal(A)$ opens the trace at challenged positions with Merkle proofs against $R_T$.
+ *Verify.* $cal(C)$ runs Freivalds and bridge checks at every opened position. Outputs accept or reject.

*Definition (Model-identity soundness).* $"Adv"^("id")_(cal(A)) = Pr["Verify accepts" and cal(A) "used" W'_j eq.not W_j "at any Freivalds-checked matrix"]$. The protocol is $epsilon$-sound if $"Adv"^("id")_(cal(A)) lt.eq epsilon$ for all adversaries $cal(A)$.

== Model Swap or Downgrade

*Proposition 1 (Model identity).* _For each Freivalds-checked matrix, if the provider used $W'_j^((i)) eq.not W_j^((i))$, the Freivalds check rejects with probability $gt.eq 1 - 1\/p$._

_Proof sketch._ The adversary commits $(x, z)$ via $R_T$ before learning $r_j$. If $z eq.not W_j^((i)) x$ (wrong weights), let $d = W_j^((i)) x - z eq.not 0$. The check $v_j^((i)) dot x equiv r_j^T dot z space (mod p)$ reduces to $r_j^T dot d equiv 0 space (mod p)$. Since $d eq.not 0$ is fixed before $r_j$ is revealed, and $r_j$ is uniform over $bb(F)_p^m$, exactly $p^(m-1)$ of $p^m$ vectors satisfy this linear equation. Thus $Pr["accept"] = 1\/p lt.eq 2^(-32)$. $square$

A single audit of a single token suffices. The guarantee is unconditional --- it does not depend on statistical sampling or threshold calibration.

== Attention Manipulation

*Proposition 2 (Attention replay bound).* _If attention replay with threshold $tau$ accepts, the committed post-attention output agrees with the verifier's FP64 recomputation on at least fraction $tau$ of INT8 elements. Disagreement is bounded by the FP16$arrow.l.r$FP64 requantization corridor and the KV sampling boundary._

The shell locks the inputs ($Q$, $K$, $V$) and outputs (post-$W_o$) of every attention block. Manipulation that alters the INT8 output beyond the corridor --- local-window approximations, suppressed context, substituted patterns --- produces element-wise disagreement exceeding $tau$ and fails the replay. The remaining adversarial freedom is bounded by the corridor width and the KV sampling rate.

== Fabricated Prefix Context

*Proposition 3 (KV tampering detection).* _If the provider tampered with $m$ out of $n$ prefix positions and the verifier samples $k$, detection probability is $P("catch") = 1 - (1 - m\/n)^k$._

Committed KV values cannot change after $R_"KV"$ is sent (hash collision resistance). The verifier anchors the commitment to real computation by sampling earlier positions and running full shell verification. Detection probability increases monotonically with the tampering fraction $m\/n$.

== Deployment Configuration Tampering

The deployment manifest $M$ binds the tokenizer, weight Merkle root, quantization scheme, sampling parameters, EOS policy, and system prompt hash into the receipt. The verifier checks $M$ against known-good values (e.g., the published tokenizer for a given model family, the weight Merkle root from the public checkpoint). Any change to the deployment configuration produces a different manifest.

= The Attention Gap <sec-attention-gap>

The shell is exactly verifiable because its operations are deterministic or canonically recomputable: INT8 matmuls are checked cryptographically via Freivalds, while requantization, RoPE, SiLU, and RMSNorm are verified by exact or canonical recomputation. Attention ($Q K^T$, softmax, $alpha V$) is computed in FP16/BF16, which is not bit-reproducible across hardware.

The protocol constrains the attention interior from both sides: inputs ($Q$, $K$, $V$) and outputs (post-$W_o$) are exactly verified by the shell. Attention replay directly checks consistency. Cross-layer constraints force fake attention to survive the residual stream.

The remaining adversarial freedom comes from two sources:

+ *FP16$arrow.l.r$FP64 replay mismatch.* The verifier's FP64 reference and the GPU's FP16 computation produce slightly different results. The *requantization corridor* --- the fraction of INT8 elements that disagree --- bounds the adversary's room. This is an open empirical question. High agreement ($> 99%$) means tight constraint; lower agreement means more room.

+ *Statistical KV anchoring.* The prefix $K$, $V$ values used in attention replay are commitment-verified (they match $R_"KV"$) but only statistically anchored to real computation via sampled shell checks. Unsampled prefix positions are not independently verified, so the replay proves consistency with the _committed_ prefix, not necessarily with the true execution at every earlier token.

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

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, left, left, center),
    [*Approach*], [*Provider overhead*], [*Guarantee*], [*Trust assumption*], [*Transf.*],
    [ZK proofs], [100--1000$times$ prover], [Full computation correct], [None (math)], [Yes],
    [TEEs], [Attestation HW], [Claimed code ran on attested HW], [Hardware vendor], [Yes],
    [Redundant exec.], [2--3$times$ compute], [Outputs agree across replicas], [Honest majority], [No],
    [VeriLM], [100 B receipt + rare audits], [Claimed weights used (exact);\ attention consistent (approx.)], [Verifier key secrecy], [No],
  ),
  caption: [Comparison of verifiable inference approaches.],
)

= Limitations and Extensions

The protocol's practical viability rests on two open empirical questions: the width of the requantization corridor (which bounds the attention gap) and whether storage and bandwidth costs are acceptable to providers at scale. Closing the attention gap entirely would require deterministic attention kernels or stronger proof systems, both of which violate the sidecar design constraint.

Two extensions could tighten the guarantees:

*Score anchoring.* The prover commits pre-softmax attention scores; the verifier spot-checks $"score"[t,j] eq.quest Q_t dot K_j$ as exact INT32 inner products. This constrains individual score entries independently of the output-level replay. The limitation is that the GPU computes softmax on FP16 scores, not the INT32 values the verifier checks, so the check cannot chain exactly to the softmax output.

*Batch Freivalds on prefix.* The verifier checks all prefix positions simultaneously using random linear combinations: pick random scalars $alpha_1 dots alpha_n$ and check $v dot (sum alpha_i x_i) eq.quest r^T dot (sum alpha_i z_i)$. One check covers all $n$ positions with false-accept probability $lt.eq 1\/2^(32)$. This upgrades KV provenance from statistical sampling to exact weight verification at all positions. Cost: $tilde 4 times$ audit bandwidth, making it better suited as an optional deep audit mode.

= Conclusion

VeriLM demonstrates that meaningful verification of LLM inference is possible without modifying the serving path or requiring expensive proof systems. The protocol's honest decomposition into exact, statistical, and approximate layers allows clients to understand precisely what is and is not guaranteed. The 100-byte receipt imposes negligible overhead on normal serving; the full audit is CPU-feasible for the client.

The most important next step is empirical measurement of the requantization corridor, which will determine how tightly the protocol constrains attention manipulation in practice.

#bibliography("refs.bib", title: [References])
