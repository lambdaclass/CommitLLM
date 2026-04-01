#set document(
  title: "CommitLLM: A Commit-and-Audit Protocol for Open-Weight LLM Inference",
  author: ("Federico Carrone", "Diego Kingston", "Manuel Puebla", "Mauro Toscano"),
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
        CommitLLM: A Commit-and-Audit Protocol for\ Open-Weight LLM Inference
      ]
      #v(0.8em)
      #text(size: 10pt)[Federico Carrone, Diego Kingston, Manuel Puebla, Mauro Toscano]
      #v(0.2em)
      #text(size: 9pt)[LambdaClass \ Universidad de Buenos Aires, Centro de Criptografía y Sistemas Distribuidos]
      #v(0.3em)
      #text(size: 9pt, style: "italic")[Preprint]
      #v(0.8em)
    ]

    #pad(x: 1.5em)[
      #text(size: 8.5pt)[
        #text(weight: "bold")[Abstract — ]
        Large language models are increasingly used in settings where integrity matters, but users still lack technical assurance that a provider actually ran the claimed model, decode policy, and output behavior. Fingerprinting and statistical heuristics can provide signals, but not exact per-response verification; zero-knowledge proof systems provide stronger guarantees, but at prover costs that remain impractical for production LLM serving.

        We present CommitLLM, a cryptographic commit-and-audit protocol for open-weight LLM inference. CommitLLM keeps the provider on the normal serving path and keeps verifier work fast and CPU-only (for Llama 70B, about 1.3 ms per challenged token under the measured 10-layer routine audit, and about 10 ms for a 1-token full audit) by combining commitment binding, direct audit, and randomized algebraic fingerprints, including Freivalds-style checks for large matrix products, rather than per-response proof generation or full re-execution. Its main costs are retained-state memory over the audit window and audit bandwidth, not per-response proving. In the current prototype, online tracing adds roughly 12--14% during generation; the larger commitment/finalization cost is measured separately, currently runs synchronously, and is a candidate for asynchronous deferral in production. The protocol is commitment-bound end-to-end. Within that binding, large linear layers are verified by verifier-secret, information-theoretically sound algebraic checks, quantization/dequantization boundaries and supported nonlinear subcomputations are checked by canonical re-execution, attention is verified by bounded approximate replay, and routine prefix-state provenance is statistical unless deep audit is used. Unsupported semantics fail closed. On the corrected replay path, the measured attention corridor is small on Qwen2.5-7B-W8A8 and Llama-3.1-8B-W8A8: at context lengths beyond 1k tokens, worst-case INT8 mismatch is single-digit ($L_"inf" = 8$ and $9$) and more than 99.8% of elements remain within one quantization bucket.
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

CommitLLM takes a different approach: a cryptographically bound sidecar commit-and-audit protocol that runs alongside unmodified GPU inference. The provider does not rewrite the model into a proving circuit, change the serving kernels, or generate a proof for every response; the normal serving path stays intact and the extra work is trace capture plus rare audit openings. CommitLLM borrows the cryptographic verification mindset of proof systems, but replaces full proof generation with commitment binding, direct audit, and cheap algebraic checks. The key insight is that INT8 quantized inference consists of operations that are either deterministic or canonically recomputable, and can be verified exactly. The only non-deterministic component is FP16/BF16 attention, which the protocol constrains from both sides without requiring exact reproduction.

The protocol provides exact, approximate, statistical, and fail-closed guarantees in different parts of the pipeline. The honest decomposition is: exact verification everywhere outside the attention interior, approximate attention replay, and statistical prefix anchoring unless deep audit is used. This is stronger than ordinary operational spot-checking or replica agreement: the provider commits before learning the audit challenge, and opened regions must satisfy cryptographically bound algebraic checks.

= System Model

== Threat Model

The adversary controls the inference server. They may swap model weights, modify attention computation, fabricate intermediate activations, or lie about earlier context. The adversary may cheat adaptively --- choosing different strategies for different responses. The following timing and knowledge constraints apply:

+ *Commitment precedes challenge.* The provider sends the receipt ($R_T$, $R_"KV"$, $M$, $H(s)$, $N$, $dots$) before learning whether the response will be audited.
+ *Unpredictable selection.* The provider cannot predict which response, token position, or layer the verifier will challenge.
+ *Secret key.* The verifier holds $r_j$ and $v_j^((i))$ secret; the provider never observes them or the audit decision rule.
+ *Local verification.* The verifier checks locally --- no third party sees the challenge, the opening, or the conversation.

In the intended deployment, the client's verifier software automatically and randomly audits a small fraction of responses (e.g., 1--5%).

The client holds a verifier key ($tilde 26$ MB for Llama 70B) derived from the public model weights and secret verifier-side randomness. The client audits responses directly --- no trusted third party sees the conversation. The attester / verifier / relying-party terminology here follows standard remote-attestation usage @birkholz2023rats.

CommitLLM provides interactive, client-held verification. The client who holds the verifier key checks the computation directly. The receipt alone is not a succinct SNARK-like transferable proof that third parties can verify offline --- this is a deliberate tradeoff for low overhead. However, the protocol is transferable in a weaker sense: if a full audit transcript is disclosed, third parties can independently re-check that disclosed audit against the public weights and committed receipt. What transfers is a bulky audit bundle rather than a compact proof object. Audit openings contain intermediate activations from which the conversation content can be reconstructed; this is why the client audits themselves rather than delegating to a third party by default.

The audit protocol is interactive, but the Freivalds randomness is not sampled publicly at audit time. It is precomputed into the verifier key and remains verifier-secret, so Freivalds acts here as a verifier-secret randomized algebraic check inside a cryptographically bound protocol rather than as a public non-interactive proof.

== Design Principles

CommitLLM is designed around three constraints:

+ *No serving-path changes.* The provider runs unmodified GPU inference. The only addition is a tracing layer that captures intermediates alongside normal execution; there is no proving circuit, proof generation pass, or required kernel rewrite on the serving path.
+ *Modest online overhead.* Each response carries a compact receipt binding trace commitments, deployment specs, transcript randomness, and token count. The measured online cost of tracing and hooks is currently about 12--14% across the tested 7B/8B/70B runs; heavier receipt finalization occurs after generation.
+ *Client-side verification.* The client performs all verification using CPU-feasible operations (dot products, hash checks, and --- for attention replay --- matrix arithmetic scaling as $O(n^2)$ in sequence length).

#block(
  width: 100%,
  inset: 8pt,
  stroke: 0.5pt + luma(120),
  radius: 2pt,
)[
  *Protocol guarantees at a glance.*
  + *Exact.* Preprocessing, model identity, shell matmuls, bridge operations, the final-token tail, and decode/output replay.
  + *Statistical.* Prefix KV correctness under routine audit sampling ($k$ prefix positions checked).
  + *Approximate.* Attention replay under the FP16$arrow.l.r$FP64 requantization corridor (@sec-attention-gap).
  + *Outside protocol scope.* Standard cryptographic assumptions, verifier-secret secrecy, no side-channel leakage of verifier secrets, and correct verifier execution.
]

== Protocol Boundary

The final protocol uses four guarantee classes:

+ *Exact.* A property is cryptographically bound and then checked by information-theoretically sound algebraic verification or canonical recomputation with fully specified semantics.
+ *Approximate.* The verifier independently replays the computation and constrains it tightly, but native FP16/BF16 execution is not bit-reproducible across hardware.
+ *Statistical.* Commitment binding is exact, but correctness of unopened positions depends on challenge sampling unless deep audit is used. This is a coverage-probability distinction, not a floating-point approximation.
+ *Fail-closed.* A feature is either replayed exactly or rejected explicitly; the verifier never silently accepts unsupported semantics.

CommitLLM does _not_ assume honest provider hardware or honest provider runtime behavior. Those are what the protocol is designed to remove. The explicit assumptions outside protocol scope are standard cryptographic assumptions, secrecy of verifier-only material, resistance to side-channel leakage of verifier secrets, and correct execution of the verifier itself.

The final protocol targets autoregressive decoder-only transformers with a committed capture layout and replay specification. Unsupported architectures must fail closed rather than be silently interpreted as if they shared decoder-only semantics.

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    [*Symbol*], [*Meaning*],
    [$R_W$], [Merkle root over model weights (public identity)],
    [$R_T$], [Merkle root over trace intermediates],
    [$R_"KV"$], [Merkle root over per-token $K, V$ history],
    [$M$], [Deployment commitment over input/model/decode/output specs],
    [$H(s)$], [Transcript-seed commitment],
    [$N$], [Token count in response],
    [$c, ell$], [Number of challenged tokens; number of opened layers per token],
    [$m$], [Output dimension of the checked weight matrix (length of $r_j$)],
    [$k$], [Sampled prefix positions for KV provenance],
    [$delta_"attn"$], [Maximum allowed INT8 absolute difference for attention replay],
    [$p$], [Prime modulus for Freivalds checks ($p = 2^(32) - 5$ in the current implementation); arithmetic in $bb(F)_p$],
  ),
  caption: [Notation],
)

= Verification Layers

CommitLLM's verification is structured in six layers. This section defines the taxonomy and key terms; the full procedure is specified in @sec-protocol.

The *shell* is the non-attention path through each transformer layer: seven INT8 weight matrix multiplications ($W_q$, $W_k$, $W_v$, $W_o$, $W_"gate"$, $W_"up"$, $W_"down"$), the exact INT8 bridge tensors that feed later INT8 stages ($x_"attn"$ and $x_"ffn"$), the committed post-attention output $"attn_out_i8"$ that feeds $W_o$, and canonically recomputable operations such as Q/K/V dequantization, RoPE, RMSNorm, and SiLU. The shell includes nonlinear operations (RMSNorm, SiLU), but all exact shell operations are deterministic or canonically recomputable. Its exactness is a composition: information-theoretically sound algebraic checks (Freivalds @freivalds1979) on the weight matmuls, plus deterministic or canonical recomputation on the exact bridge operations.

Attention ($Q K^T$, softmax, $alpha V$) is computed in FP16/BF16, which is not bit-reproducible across hardware --- this is the only non-exact component. The final exact tail starts from a captured pre-final-norm residual after the last layer, applies final RMSNorm exactly, binds the LM head with Freivalds, computes logits exactly, and replays the decode/output policy.

The six layers are:

+ *Input/model binding (exact).* Preprocessing policy, model identity, quantization scheme, architecture parameters, and deployment specs are committed and checked against known-good values.
+ *Shell verification (exact).* Information-theoretically sound Freivalds checks ($lt.eq 1\/p$ false-accept in the current field) on all shell weight matmuls, plus deterministic or canonical recomputation of the exact INT8 bridge tensors, Q/K/V dequantization, bias handling, RoPE, RMSNorm, and SiLU.
+ *KV provenance (statistical by default).* Merkle-committed per-token K,V history; sampled positions are shell-verified. Binding is exact; correctness of unsampled positions depends on sampling rate unless deep audit is used.
+ *Cross-layer consistency (structural).* Opening multiple layers on the same token creates algebraic coupling through the residual stream --- fake attention must stay consistent across all opened layers.
+ *Attention replay (approximate).* The verifier recomputes attention from shell-verified Q and committed prefix K,V in FP64, quantizes to INT8, and compares. Limited by FP16$arrow.l.r$FP64 mismatch.
+ *Final-token replay (exact / fail-closed).* The verifier starts from the captured pre-final-norm residual, checks the LM head with Freivalds, computes logits exactly, and replays the decode/output policy; unsupported semantics are rejected explicitly rather than silently accepted.

@fig-forward-pass traces the full forward pass for one output token. Each operation is labeled with its verification type: Freivalds-checked weight matmuls, canonically recomputable bridge operations, and the single non-exact point --- FP16 attention. The only non-replayable provider-side state is the per-layer `attn_out_i8` and its quantization scale, plus the captured pre-final-norm residual that anchors the exact final-token tail.

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
      raw("  q = RoPE(add_bias(dequant(q_i32, ...)), pos)"), t[canonical],
      raw("  k = RoPE(add_bias(dequant(k_i32, ...)), pos)"), t[canonical],
      raw("  v = add_bias(dequant(v_i32, ...))"), t[canonical],
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
      raw("final_residual = residual"), t[*STORED*],
      raw("x_norm = RMSNorm(final_residual)"), t[canonical],
      raw("x_i8, s = quantize(x_norm)"), t[exact],
      raw("logits_i32 = LM_head @ x_i8"), t[Freivalds + exact logits],
      raw("token = canonical_decode(logits_i32, seed, policy)"), t[exact / fail-closed],
    )
  ],
  caption: [Annotated forward pass for one output token. Each operation is labeled with its verification type. The only non-exact stage is FP16 attention; the final-token tail is exact because it starts from the captured pre-final-norm residual.],
) <fig-forward-pass>

#figure(
  block(width: 100%, inset: (x: 3pt, y: 4pt))[
    #set text(size: 7.3pt)
    #table(
      columns: (1.35fr, 1.7fr, auto, 1.35fr),
      align: (left, left, center, left),
      [*Component*], [*Verification method*], [*Guarantee*], [*What is bound*],
      [Input preprocessing], [Committed input spec], [Exact], [Tokenizer, chat template, BOS/EOS, truncation, special-token handling, system prompt],
      [Embedding], [Merkle proof to committed embedding root], [Exact], [Token-to-row binding],
      [Shell matmuls], [Freivalds], [Exact], [Seven shell matrix families],
      [Bridge operations], [Canonical recomputation], [Exact], [Requantization, residual, RMSNorm, RoPE, SiLU],
      [Prefix / KV provenance], [Merkle binding + sampled shell checks or deep audit], [Statistical / Exact], [Committed prefix state],
      [Attention], [Independent replay against committed prefix/output], [Approximate], [Consistency with committed shell values],
      [Final boundary], [Captured pre-final-norm residual], [Exact], [Start of the exact final-token tail],
      [LM head], [Freivalds + exact logits replay], [Exact], [Final linear map and logits],
      [Decode policy], [Canonical replay or explicit rejection], [Exact / Fail-closed], [Sampler semantics and randomness],
      [Output policy], [Exact replay or explicit rejection], [Exact / Fail-closed], [Stopping and text-cleanup semantics],
    )
  ],
  caption: [Verification coverage matrix for the final protocol. "Fail-closed" means a feature is either replayed exactly or rejected explicitly; the verifier never silently accepts unsupported semantics.],
) <tab-coverage-matrix>

= The Protocol <sec-protocol>

== Phase 0: Setup

*Public identity.* A Merkle root @merkle1987 $R_W$ is computed over all weight matrices of the published checkpoint. This root is the model's public identity --- anyone can recompute it from the published weights.

*Verifier key generation.* The verifier fixes a prime field modulus $p$; in the current implementation $p = 2^(32) - 5$. All Freivalds checks operate in $bb(F)_p = ZZ \/ p ZZ$: INT8 inputs and INT32 accumulators are lifted into this field. For the 7 shell matrix types ($W_q$, $W_k$, $W_v$, $W_o$, $W_"gate"$, $W_"up"$, $W_"down"$) and the final LM head, the verifier samples secret random vectors $r_j$ uniformly from $bb(F)_p^m$. For each per-layer matrix, the verifier precomputes $v_j^((i)) = r_j^T W_j^((i)) mod p$; for the LM head, the verifier precomputes the corresponding verifier-side check vector once for the global unembedding matrix. The verifier key also binds the embedding commitment, final RMSNorm weights, model configuration, quantization metadata, RoPE/scaling configuration, and RMSNorm epsilon needed for canonical replay. In the current benchmark projection this verifier key is roughly $tilde 48$ MB for Llama 70B, still small enough for client-held verification. After precomputation, the verifier deletes the weights. The $r_j$ vectors are verifier-secret; if the provider learns them, it can forge passing checks. Freivalds is therefore used here as verifier-secret randomized algebraic verification inside the broader cryptographic protocol, not as a standalone public proof system.

*Specification binding.* The deployment is described by four committed specs: $H_"input"$, $H_"model"$, $H_"decode"$, and $H_"output"$. These bind preprocessing semantics, model configuration, decode policy, and output policy respectively. Unsupported semantics are not left ambiguous: they must either be replayed exactly or rejected explicitly.

== Phase 1: Commitment

The provider runs inference normally. The serving path is unchanged except for a tracing layer that captures intermediates alongside execution. In batched serving (e.g., vLLM with paged attention), multiple requests share GPU resources; the tracing layer captures per-request intermediates, extracting per-request activations from the batched computation without modifying the batched kernels.

For every token at every layer (both prefill and decode), the provider records:

- The INT8 input to each of the 7 weight matrices
- The INT32 accumulator output ($W_"i8" times x_"i8"$, exact)
- The requantized INT8 values at each bridge
- The post-attention INT8 output
- The per-tensor quantization scales at each requantization bridge
- The captured pre-final-norm residual needed to anchor the exact final-token tail

After generation completes, the provider builds two Merkle trees @merkle1987. The trees are separate because they serve different access patterns: shell verification (Step 2) opens a few positions from $R_T$, while KV provenance (Step 3) opens the full prefix from $R_"KV"$. A single tree would force opening all intermediates at every prefix position just to extract the KV values.

- $R_T$ (trace tree): over all intermediates at all tokens. The provider cannot change any activation after committing.
- $R_"KV"$ (KV tree): over per-token $K, V$ state across all layers. The provider cannot retroactively rewrite earlier tokens' context. This tree allows efficient opening of the full prefix KV without opening the complete trace at every position.

A deployment commitment binds everything outside the forward pass:

$ M = H(H_"input" || H_"model" || H_"decode" || H_"output") $

The provider also samples a fresh per-request transcript seed $s$, commits $H(s)$ in the receipt, and derives per-token randomness deterministically from $s$ and the token index.

The provider returns the response plus a compact receipt $(R_T, R_"KV", M, H(s), N, dots)$. Most responses are never audited. The only normal-path additions are the tracing layer and receipt machinery; the heavier Merkle/packing finalization can run after generation completes.

== Phase 2: Verification

The client decides to audit a response. The provider does not know which responses will be challenged or which tokens within them will be opened.

=== Step 1: Challenge

The verifier selects $c$ random token positions from the $N$-token response and, for each, chooses $ell$ layers to open. Opening multiple layers on the same token is preferred --- this enables cross-layer consistency checks (Step 4).

=== Step 2: Shell Verification

For each challenged token $t$ at each opened layer $i$, the provider opens the INT8 inputs $x_"i8"$, INT32 accumulators $z_"i32"$, per-tensor quantization scales, the residual stream values at the layer boundaries, and the post-attention output $"attn_out_i8"$ --- all with Merkle proofs against $R_T$. The verifier performs four checks:

+ *Freivalds on each shell weight matrix.* For each of the 7 shell matrices, the verifier checks $v_j^((i)) dot x equiv r_j^T dot z space (mod p)$. Each check is two dot products in $bb(F)_p$ --- $O(n)$. If the provider used wrong weights, false-accept probability is $lt.eq 1\/p$ per matrix. The bound follows from the Schwartz--Zippel lemma: a nonzero linear form over $bb(F)_p$ vanishes on at most a $1\/p$ fraction of inputs.

+ *Bridge tensors (exact).* Wherever an exact INT8 tensor feeds a later INT8 stage --- for example the committed $x_"attn"$ boundary and the FFN bridge tensors --- the verifier recomputes the canonical bridge from the opened accumulators, quantization scales, and residual-stream state. This prevents the provider from passing Freivalds on the INT32 accumulators while feeding fabricated INT8 bridge values into downstream checks. Q/K/V themselves are not separately bridged through committed $q_"i8"$/$k_"i8"$/$v_"i8"$ values in the canonical path; they are reconstructed by dequantization, bias application, and RoPE from the opened accumulators. The retained post-attention output $"attn_out_i8"$ is instead checked by the attention-replay layer below.

+ *SiLU (exact / canonical).* In the toy path, INT8 inputs permit a 256-entry lookup table. In the production W8A8 path, the verifier canonically recomputes the scaled SiLU bridge from dequantized gate/up accumulators and checks the requantized result.

+ *RoPE and RMSNorm.* RoPE is canonically recomputed from the position index and model configuration in deterministic f64 arithmetic. RMSNorm is canonically recomputed from the opened residual stream values; the result is verified in the quantized output space (the recomputed value must requantize to the same $"i8"$ as the committed value).

After this step, the verifier has trusted $Q_t$, $K_t$, $V_t$ for the challenged token as shell-verified accumulator outputs together with their canonical dequantization, bias handling, and RoPE reconstruction.

=== Step 3: KV Provenance

Attention at token $t$ requires the full prefix $K_(1..t)$, $V_(1..t)$. Shell verification gives trusted values for the challenged token but not the prefix.

For each opened layer $i$, the provider opens the $K$, $V$ values at all prefix positions $j < t$ with Merkle proofs against $R_"KV"$. The verifier then:

+ Verifies the Merkle proofs --- confirming these are the values the provider committed in Phase 1.
+ Randomly samples $k$ earlier token positions $j_1, dots, j_k < t$.
+ For each sampled position $j_s$, runs full shell verification at layer $i$ (Freivalds on all 7 matrices, requantization, RoPE, RMSNorm, SiLU), producing independently verified $K_(j_s)^((i))$, $V_(j_s)^((i))$.
+ Compares these shell-verified values against the corresponding entries opened from $R_"KV"$.

If the provider tampered with $f$ out of $n$ prefix positions and the verifier samples $k$:

$ P("catch") = 1 - (1 - f\/n)^k $

Commitment binding is exact (hash collision resistance). Correctness of unsampled positions is statistical --- the provider risks the verifier sampling any corrupted position.

=== Step 4: Cross-Layer Consistency

There is no separate proof object here; the consistency check comes from opening multiple layers and verifying the committed bridges between them. When the verifier opens layers $L$ and $L+1$ on the same token, fake attention at layer $L$ must produce a post-attention output that, after requantization and $W_o$, feeds into layer $L+1$'s RMSNorm consistently with the committed trace. Both sides of this boundary are shell-verified, so the adversary cannot fabricate a consistent bridge without matching the committed values exactly. The more layers opened, the tighter the constraint. In full-audit mode (all layers opened), fake attention at every layer must stay mutually consistent through the entire residual stream.

=== Step 5: Attention Replay

This step combines the trusted $Q_t$ reconstructed in Step 2 with the commitment-verified prefix $K_(1..t)$, $V_(1..t)$ from Step 3. The verifier recomputes attention independently in FP64. Since the KV cache stores post-RoPE values, the verifier dequantizes directly and computes:

+ Scores: $Q_t dot K_j \/ sqrt(d)$ for all prefix positions $j$
+ Softmax over scores
+ Weighted sum: $sum alpha_j times V_j$

The verifier quantizes the result to INT8 using the quantization scales opened from the trace (committed in $R_T$) and compares element-wise against the committed post-attention output. A response passes if the maximum INT8 absolute difference is at most $delta_"attn"$; this tolerance must be calibrated empirically by measuring the FP16$arrow.l.r$FP64 requantization corridor on real model activations (see @sec-attention-gap). In the current implementation, this corridor is measured rather than assumed: on the corrected replay path it remains single-digit in INT8 space on both Qwen2.5-7B-W8A8 and Llama-3.1-8B-W8A8, with more than 99.8% of elements remaining within one quantization bucket.

This pins the attention output to a specific computation over the committed values. Without it, the adversary could fabricate a plausible-looking but wrong attention output --- a local-window approximation, suppressed attention to specific positions, or a cheaper model's attention pattern --- as long as it satisfied cross-layer consistency.

*Limitations.* The replay proves consistency with the _committed_ prefix, not necessarily with true execution at every earlier token. Prefix KV values are commitment-verified (they match $R_"KV"$) but only statistically anchored to real computation via the sampled shell checks in Step 3. The replay also cannot match the GPU's FP16 attention exactly --- the verifier's FP64 reference will differ slightly near quantization bucket boundaries.

=== Step 6: Final-Token Replay

The verifier now enters the exact final-token tail. The provider opens the captured pre-final-norm residual for the challenged token, together with the committed decode/output policy and the revealed transcript seed. The verifier then:

+ Applies the final RMSNorm exactly using the committed model spec.
+ Quantizes into the LM-head input space canonically.
+ Checks the LM head with Freivalds and computes logits exactly.
+ Replays the canonical decode policy using the committed decode spec and the per-token seed derived from the revealed transcript seed.
+ Verifies the chosen token and the output-policy behavior (EOS handling, stop strings, max/min stopping rules, and claimed text cleanup) exactly, or rejects the feature explicitly if the deployment claims an unsupported policy.

This step is what makes the final-token boundary exact. The verifier must not derive the token from a hidden state reconstructed through many layers of approximate attention replay and then call the result exact.

== Summary

@tab-verification-methods summarizes the verification method applied to each operation in the forward pass.

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    [*Operation*], [*Verification*],
    [Input / deployment specs], [Committed four-spec manifest],
    [Input embedding], [Table lookup (exact)],
    [$W_q, W_k, W_v$ (INT8)], [Freivalds],
    [INT8 bridge tensors ($x_"attn"$, $x_"ffn"$)], [Exact recomputation],
    [`attn_out_i8` (committed post-attention output)], [Compared against attention replay],
    [$Q$, $K$, $V$ dequant + bias + RoPE], [Canonical recomputation],
    [Attention], [Replay + cross-layer],
    [$W_o$ (INT8)], [Freivalds],
    [RMSNorm], [Canonical recomputation],
    [$W_"gate"$, $W_"up"$ (INT8)], [Freivalds],
    [SiLU $dot.o$ up], [Canonical recomputation + exact requantized check],
    [$W_"down"$ (INT8)], [Freivalds],
    [Final boundary], [Captured pre-final-norm residual],
    [LM head], [Freivalds + exact logits replay],
    [Decode policy], [Canonical replay or fail-closed rejection],
    [Output policy], [Exact replay or fail-closed rejection],
  ),
  caption: [Per-layer verification methods],
) <tab-verification-methods>

== Audit Walkthrough: One Token, Two Layers

Audit token $t = 47$ at layers 12 and 13 of a 70B model, sampling $k = 8$ prefix positions.

+ *Challenge.* Verifier sends $(t = 47, {12, 13})$.
+ *Shell.* Provider opens trace at both layers from $R_T$. Verifier runs Freivalds on all 7 matrices per layer (28 dot products total), recomputes all bridges. Yields trusted $Q_(47)$, $K_(47)$, $V_(47)$ at both layers.
+ *KV provenance.* Provider opens $K_(1..46)$, $V_(1..46)$ at both layers from $R_"KV"$. Verifier picks 8 random earlier positions, runs full shell verification at each, confirms committed $K$, $V$ match.
+ *Cross-layer.* Post-attention output at layer 12 feeds into layer 13's RMSNorm --- both sides shell-verified, so boundary must match exactly.
+ *Replay.* At each opened layer, the verifier recomputes attention in FP64 from shell-verified $Q_(47)$ and the commitment-verified prefix. Quantizes to INT8, compares against committed `attn_out_i8`. Passes if the maximum INT8 absolute difference is $lt.eq delta_"attn"$ at both layers.
+ *Final token.* Provider opens the captured pre-final-norm residual and revealed transcript seed. Verifier applies final RMSNorm exactly, checks the LM head with Freivalds, recomputes logits exactly, replays the canonical decode policy, and confirms the chosen token and output policy.

== Data Lifecycle

@tab-data-lifecycle shows what data exists at each stage of the protocol.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    [*Receipt fields (compact)*], [*Provider retains (ring buffer)*], [*Revealed on audit*],
    [$R_T$ (trace Merkle root)], [INT8 inputs to all 7 matrices], [Merkle openings from $R_T$],
    [$R_"KV"$ (KV Merkle root)], [INT32 accumulator outputs], [Merkle openings from $R_"KV"$],
    [Spec commitment $M$], [Requantized INT8 at each bridge], [Opened intermediates at challenged positions],
    [Seed commitment $H(s)$], [`attn_out_i8` + quantization scales], [Full prefix $K$, $V$ at opened layers],
    [Token count $N$], [Captured pre-final-norm residual], [Opened final boundary state + revealed seed],
    [], [Per-tensor quantization scales], [],
  ),
  caption: [Data lifecycle: receipt (every response), retained state (RAM ring buffer, discarded after audit window), and audit openings (revealed only when challenged).],
) <tab-data-lifecycle>

= Security Analysis

== Security Game

We formalize model-identity soundness via an interactive game between a challenger $cal(C)$ (verifier) and an adversary $cal(A)$ (provider).

$bold("Game")^("id")_(cal(A))(lambda)$:

+ *Setup.* $cal(C)$ publishes weights $W$ and Merkle root $R_W$. $cal(C)$ samples verifier-secret Freivalds vectors for every shell matrix family and for the LM head, precomputes the corresponding verifier-side checks, and stores the secret verifier key $"sk"$. $cal(A)$ receives $W$, $R_W$ but never observes $"sk"$.
+ *Commit.* $cal(A)$ receives a prompt, runs inference (using any strategy), and returns a response plus receipt $(R_T, R_"KV", M, H(s), N, dots)$.
+ *Challenge.* $cal(C)$ selects $c$ token positions and $ell$ layers per token uniformly at random. $cal(A)$ learns the challenge only now.
+ *Open.* $cal(A)$ opens the trace at challenged positions with Merkle proofs against $R_T$.
+ *Verify.* $cal(C)$ runs Freivalds and bridge checks at every opened position. Outputs accept or reject.

*Definition (Model-identity soundness).* $"Adv"^("id")_(cal(A)) = Pr["Verify accepts" and cal(A) "used" W'_j eq.not W_j "at any Freivalds-checked matrix"]$. The protocol is $epsilon$-sound if $"Adv"^("id")_(cal(A)) lt.eq epsilon$ for all adversaries $cal(A)$.

== Model Swap or Downgrade

*Proposition 1 (Model identity).* _For each Freivalds-checked matrix, if the provider used $W'_j^((i)) eq.not W_j^((i))$, the Freivalds check rejects with probability $gt.eq 1 - 1\/p$._

_Proof sketch._ The adversary commits $(x, z)$ via $R_T$ before learning $r_j$. If $z eq.not W_j^((i)) x$ (wrong weights), let $d = W_j^((i)) x - z eq.not 0$. The check $v_j^((i)) dot x equiv r_j^T dot z space (mod p)$ reduces to $r_j^T dot d equiv 0 space (mod p)$. Since $d eq.not 0$ is fixed before $r_j$ is revealed, and $r_j$ is uniform over $bb(F)_p^m$, exactly $p^(m-1)$ of $p^m$ vectors satisfy this linear equation. Thus $Pr["accept"] = 1\/p$ (about $1\/(2^(32)-5)$ in the current implementation). $square$

A single audit of a single token suffices. The guarantee is unconditional --- it does not depend on statistical sampling or threshold calibration.

== Attention Manipulation

*Proposition 2 (Attention replay bound).* _If attention replay with tolerance $delta_"attn"$ accepts, the committed post-attention output differs from the verifier's FP64 recomputation by at most $delta_"attn"$ in INT8 max absolute difference. The remaining gap is bounded by the empirically measured FP16$arrow.l.r$FP64 requantization corridor together with the KV sampling boundary._

The shell locks the inputs ($Q$, $K$, $V$) and outputs (post-$W_o$) of every attention block. Manipulation that alters the INT8 output beyond this max-absolute-difference corridor --- local-window approximations, suppressed context, substituted patterns --- fails the replay. The remaining adversarial freedom is bounded by the corridor width and the KV sampling rate.

== Fabricated Prefix Context

*Proposition 3 (KV tampering detection).* _If the provider tampered with $f$ out of $n$ prefix positions and the verifier samples $k$, detection probability is $P("catch") = 1 - (1 - f\/n)^k$._

Committed KV values cannot change after $R_"KV"$ is sent (hash collision resistance). The verifier anchors the commitment to real computation by sampling earlier positions and running full shell verification. Detection probability increases monotonically with the tampering fraction $f\/n$.

== Deployment Configuration Tampering

The deployment commitment $M$ binds four specs into the receipt: preprocessing, model configuration, decode policy, and output policy. This includes the tokenizer, chat template, BOS/EOS preprocessing policy, truncation semantics, weight identity, quantization scheme, RoPE/scaling configuration, RMSNorm epsilon, adapter identity, sampler version, temperature/top-k/top-p, penalties, grammar constraints, stop policy, and detokenization semantics. The verifier checks these against known-good values for the intended deployment. Any change to the deployment configuration produces a different committed spec surface.

== Assumptions Outside Protocol Scope

The protocol does not assume honest provider hardware or honest provider runtime behavior. Those are the target of verification. The explicit assumptions outside protocol scope are:

+ standard cryptographic assumptions for the hash functions
+ information-theoretic soundness of the finite-field Freivalds checks
+ secrecy of verifier-only material such as the Freivalds vectors
+ no side-channel leakage of verifier-secret material
+ correct execution of the verifier itself

= The Attention Gap <sec-attention-gap>

The shell is exactly verifiable because its operations are deterministic or canonically recomputable: INT8 matmuls are checked with information-theoretically sound Freivalds checks, while requantization, RoPE, SiLU, and RMSNorm are verified by exact or canonical recomputation. Attention ($Q K^T$, softmax, $alpha V$) is computed in FP16/BF16, which is not bit-reproducible across hardware.

The protocol constrains the attention interior from both sides: inputs ($Q$, $K$, $V$) and outputs (post-$W_o$) are exactly verified by the shell. Attention replay directly checks consistency. Cross-layer constraints force fake attention to survive the residual stream.

The remaining adversarial freedom comes from two sources:

+ *FP16$arrow.l.r$FP64 replay mismatch.* The verifier's FP64 reference and the GPU's FP16/BF16 computation produce slightly different results. The *requantization corridor* is therefore characterized empirically in INT8 space by worst-case absolute-difference bounds together with diagnostic agreement statistics. On the corrected replay path, this corridor is now measured rather than hypothetical: for Qwen2.5-7B-W8A8 and Llama-3.1-8B-W8A8 we observe worst-case INT8 $L_"inf"$ of $8$ and $9$ respectively across six workloads at context lengths beyond 1k tokens, with first generated token max $= 5$ on both models and more than 99.8% of elements within one bucket. Small $L_"inf"$ and high agreement mean tight constraint; larger gaps mean more room.

+ *Statistical KV anchoring.* The prefix $K$, $V$ values used in attention replay are commitment-verified (they match $R_"KV"$) but only statistically anchored to real computation via sampled shell checks. Unsampled prefix positions are not independently verified, so the replay proves consistency with the _committed_ prefix, not necessarily with the true execution at every earlier token.

#figure(
  text(size: 8pt)[
    #table(
      columns: (1.6fr, 0.7fr, 0.85fr, 0.95fr, 0.95fr, 1.0fr),
      align: (left, center, center, center, center, center),
      inset: 4pt,
      [*Model*], [*Wklds*], [*Max pos.*], [*Max $L_"inf"$*], [*1st-gen*], [*Frac. ≤ 1*],
      [Qwen2.5-7B#linebreak()W8A8], [6], [1164], [8], [5], [$gt 99.8%$],
      [Llama-3.1-8B#linebreak()W8A8], [6], [1165], [9], [5], [$gt 99.9%$],
    )
  ],
  caption: [Measured attention requantization corridor on the corrected replay path. These numbers assume architecture-correct replay semantics, including Q/K/V bias handling and model-specific RoPE conventions/scaling. Measurements were collected on A100-80GB GPUs under vLLM eager mode.],
)

= Provider Costs

== Latency

We separate two costs. *Online generation overhead* is the user-visible slowdown from trace capture, hooks, and synchronization while tokens are being generated and streamed. *Post-generation finalization overhead* is the additional cost to build and pack the receipt after generation has completed. The former is the main serving-path metric; the latter matters for provider throughput and infrastructure sizing, but not for time-to-first-token or live streaming latency if receipt finalization is asynchronous.

Across the currently measured configurations, online overhead is stable at roughly 12--14% (about 1.3--1.8 ms/token) from Qwen2.5-7B-W8A8 through Llama-3.1-70B-W8A8. These serving-overhead measurements were collected on A100-80GB GPUs under vLLM eager mode. Post-generation finalization is currently much larger, but it occurs after generation and is therefore off the critical user-facing path. Profiling indicates that the dominant post-generation cost in the current prototype is the Rust-side commitment routine, especially trace hashing, Merkle construction, and eager KV transcript derivation; Python-side preparation is negligible by comparison.

#figure(
  text(size: 8pt)[
    #table(
      columns: (1.7fr, 0.9fr, 0.9fr, 1.1fr, 1.0fr),
      align: (left, center, center, center, center),
      inset: 4pt,
      [*Model*], [*Cfgs*], [*Tokens*], [*Online OH*], [*Full packed OH*],
      [Qwen2.5-7B#linebreak()W8A8], [4], [105--297], [12.7--13.8%], [225--338%],
      [Llama-3.1-8B#linebreak()W8A8], [2], [104--168], [13.5--13.7%], [422--554%],
      [Llama-3.1-70B#linebreak()W8A8], [1], [296], [12.1%], [390%],
    )
  ],
  caption: [Measured serving overhead on the current implementation. "Online OH" is the user-visible generation-path overhead from capture, hooks, and synchronization. "Full packed OH" includes post-generation receipt finalization and packing. The current matrix is complete for Qwen2.5-7B-W8A8 and partial for the two Llama models because the longer 70B finalization runs hit the benchmark timeout.],
)

== Verifier Cost

Client-side verification runs entirely on CPU. We benchmarked the per-token verification cost at real model dimensions using random data (the verifier needs only the precomputed key and the opened trace, not the actual weights). Measurements on a commodity laptop CPU (Apple M4):

#figure(
  text(size: 8pt)[
    #table(
      columns: (1.6fr, 0.8fr, 1.0fr, 1.0fr, 1.0fr),
      align: (left, center, center, center, center),
      inset: 4pt,
      [*Model*], [*Layers*], [*10 layers*], [*All layers*], [*Per layer*],
      [Llama 3.1 8B], [32], [0.66 ms], [1.98 ms], [62 µs],
      [Llama 3.1 70B], [80], [1.27 ms], [10.1 ms], [126 µs],
      [Llama 3.1 405B], [126], [2.33 ms], [29.8 ms], [236 µs],
    )
  ],
  caption: [Measured per-token verification cost (Freivalds checks on all 7 matrices per layer + SiLU recomputation). "10 layers" is the routine stratified audit; "All layers" is the full audit. Merkle and IO-chain proof checks add $<$1 µs and are omitted. For $S$ challenged tokens, multiply by $S$.],
)

For the default audit mode ($S=1$ token, 10 routine layers), verifying one Llama 70B response takes about 1.3 ms --- well within CPU-feasible bounds. Even a full-audit pass over all 80 layers at 5 challenged tokens costs roughly 50 ms.

== Storage

Only the non-derivable retained state requires storage: post-attention INT8 outputs, the associated per-tensor quantization scales, and the captured pre-final-norm residual that anchors the exact final-token tail (everything else is re-derivable):

This storage accounting is only for retained per-token trace state. The deployment commitment $M$ is computed and bound into the receipt at commit time; it is configuration metadata, not non-derivable trace state.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    [*Model*], [*Layers*], [*hidden_dim*], [*Per token*],
    [Llama 8B], [32], [4096], [$tilde 144$ KB],
    [Llama 70B], [80], [8192], [$tilde 672$ KB],
    [Llama 405B], [126], [16384], [$tilde 2.1$ MB],
  ),
  caption: [Per-token storage for non-derivable intermediates],
)

== Audit Window

With a short audit window (1--2 minutes), traces fit in a RAM ring buffer with no disk I/O. For Llama 70B on 4$times$ H100, assuming an aggregate retained-state write rate of $tilde 1.3$ GB/s, a 2-minute window requires $tilde 156$ GB ($tilde 8%$ of 2 TB system RAM). Longer windows spill to NVMe or networked storage.

Short audit windows require automated auditing --- the client's audit decision must be programmatic.

= Related Work

The most direct alternative to CommitLLM is proving inference in zero knowledge. Earlier verifiable-inference systems such as SafetyNets @ghodsi2017safetynets and later zk approaches such as zkCNN @liu2021zkcnn show that inference correctness can be proved by reducing neural networks to arithmetic-circuit-style verification. However, current ZK systems still impose large prover overheads. For a model that costs dollars per response to serve, this translates to hundreds or thousands of dollars per proved response --- prohibitive for production inference. The core design difference is economic: CommitLLM keeps the provider on the normal serving path and keeps the verifier CPU-feasible, while a SNARK-style design pushes large extra cost into per-response proof generation in exchange for a compact transferable proof.

At a more foundational level, CommitLLM also sits in the lineage of interactive proofs and delegated computation. Classical interactive-proof formulations @goldwasser1989knowledge and later delegated-computation protocols @goldwasser2008delegating study how a verifier can check outsourced computation without performing the full work locally. CommitLLM does not instantiate those general-purpose protocols directly; instead, it specializes the verification structure to open-weight INT8 LLM inference, using public weights, commitment binding, and verifier-secret algebraic checks to obtain a much cheaper normal serving path.

Related cryptographic ML systems such as Delphi @mishra2020delphi address a different problem: privacy-preserving neural inference, where neither party should reveal its inputs or model. CommitLLM instead targets model-provenance auditing for open weights, where the model is public and the goal is to verify that the claimed weights were actually used.

CommitLLM makes the opposite tradeoff: verification is interactive (the provider must respond to challenges) and the receipt alone is not a succinct SNARK-like transferable proof. In exchange, the normal serving path remains close to ordinary serving: the measured online overhead from tracing is currently about 12--14%, while heavier receipt finalization is shifted after generation. Transferability exists only in a weaker disclosed-audit sense: a fully disclosed audit transcript can be re-checked by third parties, but that is a bulky audit bundle rather than a compact proof. The expensive part --- verification --- only occurs on the small fraction of responses that are audited.

Trusted execution environments (TEEs) and remote attestation provide another alternative @birkholz2023rats @menetrey2022attestation. TEEs can attest that specific code ran on specific hardware, avoiding the overhead of proof systems, but they introduce hardware trust assumptions, limit deployment flexibility, and tie verification to a specific vendor's attestation chain. CommitLLM requires no hardware trust beyond standard computation.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, left, left, center),
    [*Approach*], [*Provider overhead*], [*Guarantee*], [*Trust assumption*], [*Transf.*],
    [ZK proofs], [100--1000$times$ prover], [Full computation correct], [None (math)], [Yes],
    [TEEs], [Attestation HW], [Claimed code ran on attested HW], [Hardware vendor], [Yes],
    [Redundant exec.], [2--3$times$ compute], [Outputs agree across replicas], [Honest majority], [No],
    [CommitLLM], [$tilde$12--14% online + post-gen finalization + rare audits], [Claimed weights used (exact);\ attention consistent (approx.)], [Verifier key secrecy + verifier integrity], [Audit#linebreak()bundle],
  ),
  caption: [Comparison of verifiable inference approaches.],
)

= Limitations and Extensions

The protocol's practical viability no longer depends on whether the attention corridor exists at all: on the corrected Qwen and Llama paths it is small and stable. The remaining open questions are how broadly this corridor generalizes across additional model families and larger scales, how much it can be tightened by deeper audit objects, and what audit/storage policy providers will accept at production scale. Closing the attention gap entirely would require deterministic attention kernels or stronger proof systems, both of which violate the sidecar design constraint. The final protocol also assumes an autoregressive decoder-only architecture with the committed capture layout and architecture-correct replay semantics; broader architecture support requires additional schema and replay work but does not change the guarantee taxonomy.

A full composed soundness theorem remains future work. The current protocol is commitment-bound end-to-end: large linear components are verified by verifier-secret, information-theoretically sound algebraic checks, supported nonlinear components by canonical replay, attention by bounded approximate replay, and routine KV provenance is statistical unless deep audit is used.

Routine KV provenance is correspondingly weakest against sparse tampering: if an adversary corrupts only a few prefix positions, the per-audit detection probability under small-$k$ sampling can be low. The short audit window also creates a denial-of-audit surface if a provider can force responses past the retention horizon before a challenge arrives. In practice, these risks push deployments toward automated audits, explicit response-deadline policies, longer or durable retention for high-value responses, and deeper audit modes when sparse prefix manipulation is a concern.

Two extensions could tighten the guarantees:

*Score witnessing.* In deep audit, the provider returns pre-softmax score witnesses for the opened layers and token, and the verifier compares them against the canonical $Q K^T \/ sqrt(d)$ reconstruction before continuing softmax from the witnessed scores. This aims to stop error propagation at the main nonlinear boundary without changing the routine path. We are actively implementing and measuring this extension.

*Output-conditioning analysis.* The shell already verifies $W_o$ exactly, but the current paper does not yet quantify how much adversarial freedom can remain after an attention perturbation survives the replay corridor and then flows through $W_o$. A layer-wise conditioning analysis would translate corridor width into an explicit downstream bound.

*Stronger KV provenance.* The verifier can check all prefix positions simultaneously using random linear combinations: pick random scalars $alpha_1 dots alpha_n$ and check $v dot (sum alpha_i x_i) eq.quest r^T dot (sum alpha_i z_i)$. One check covers all $n$ positions with false-accept probability $lt.eq 1\/p$ (about $1\/(2^(32)-5)$ in the current field). This upgrades KV provenance from statistical sampling to exact weight verification at all positions. Cost: $tilde 4 times$ audit bandwidth, making it better suited as an optional deep audit mode.

*Broader empirical matrix.* The current corridor measurements cover Qwen2.5-7B-W8A8 and Llama-3.1-8B-W8A8. Immediate next validation targets are one additional family (e.g., Mistral or Gemma) and one materially larger model, so that the tolerance and audit policy story is supported beyond the first two families.

*Formalization and broader coverage.* Ongoing follow-on work also includes Lean formalization of the core verification claims, validation on additional model families and larger checkpoints, stronger KV provenance, and $W_o$ conditioning to translate corridor width into tighter downstream bounds.

= Conclusion

CommitLLM demonstrates that meaningful verification of LLM inference is possible without modifying the serving path or requiring expensive proof systems. The protocol's honest decomposition into exact, statistical, approximate, and fail-closed layers allows clients to understand precisely what is and is not guaranteed. The measured online serving overhead from tracing is currently about 12--14% across the tested 7B/8B/70B runs; heavier receipt finalization remains off the critical user-facing path, and on Llama 70B verifier cost is about 1.3 ms per challenged token under the measured 10-layer routine audit, rising to about 10 ms for a 1-token full audit over all 80 layers on a commodity CPU. On the corrected replay path, the measured attention corridor is already small on both Qwen and Llama, which means the central protocol architecture is no longer resting on a speculative empirical assumption.

The immediate next steps are to tighten an already strong attention story even further with score witnessing, continue the Lean formalization, quantify downstream freedom through $W_o$ conditioning, strengthen KV provenance, and extend the empirical matrix to additional families and larger models. Those are important for the strongest final security claim, but they are now follow-on work on top of a functioning protocol rather than rescue work for a broken design.

#bibliography("refs.bib", title: [References])
