# Roadmap

All remaining work for CommitLLM, organized by priority. Items numbered sequentially within each tier.

Mark a task done only when it is true in code, tests, docs, and operational behavior. Completed milestones live in [CHANGELOG.md](./CHANGELOG.md).

The shell/tail/binding protocol is structurally complete. The kept path preserves the full exact 7-matrix Freivalds shell (`Wq/Wk/Wv/Wo/Wg/Wu/Wd`) across all audit tiers; tiers differ in attention evidence and payload cost, not in whether the exact linear shell is checked.

---

## Tier 0 — Critical path

Highest-priority work. Do in this order.

| # | Item | Status |
|---|------|--------|
| 1 | **Persistent GPU server** — dedicated A100/H100 instance for continuous testing. Removes Modal/RunPod bottleneck. Every item below depends on GPU access. | open |
| 2 | **Unify attention path** — one `x_attn` boundary, one QKV replay story across prover/verifier/corridor. No silent mixing of bridge-derived Q with committed K/V. | open |
| 3 | **Adversarial testing** — cheating-provider test suite: receipt forgery, transcript splicing, intermediate tampering, model substitution, selective layer cheating, KV injection. Every attack must fail verification or be documented as accepted gap. Includes real-model constrained corridor attacks (see #5). | **in progress** |
| 4 | **Formal security argument** — define the cheating game, state Freivalds soundness bound (1/p), state detection probability as f(sampling rate, cheating fraction), state what commitment binding prevents. | open |
| 5 | **Real-model corridor attacks** — run corridor amplification on Qwen-7B and Llama-8B with production weights. Determine whether ±τ perturbations can flip tokens on trained models (not just random-weight toy). Separate item from broad adversarial testing because the current attention issue needs focused GPU confirmation. | open |
| 6 | **KV provenance** — sampled/batched Freivalds checks verify committed K/V are consistent with committed Wk/Wv weights and bridge inputs. Tightening `a` alone is insufficient if the adversary can manipulate prefix state; KV provenance closes the upstream path. | open |
| 7 | **Score witnessing** — commit pre-softmax scores S=Q·K^T/√d per audited token. Verifier checks S against canonical QK^T from shell-verified Q and committed K, then recomputes softmax(S)·V independently. Fallback for deployments using nondeterministic fast kernels (FlashAttention) where τ>0. Not needed if deterministic replay (#68) achieves τ=0 with a defined arithmetic spec. Cost: O(seq_len × n_heads) per audited token per layer, deep audit only. | open |
| 8 | **W_o conditioning** — compute σ_min(W_o) and ‖W_o‖₂ per layer on real models. Quantifies adversary freedom when attn_out_i8 is commitment-bound but not independently replayed. Directly informs how tight the corridor needs to be before score witnessing is implemented. | open |
| 9 | **Deep-audit payload sizing** — budget the per-token cost of score witnessing and KV provenance at production context lengths (4K, 32K, 128K). Determine storage, bandwidth, and verification-time scaling. Informs audit tier design and whether long-context deployments need different policies. | open |
| 10 | **RMSNorm contraction on real models** — measure ρ_j = γ·‖W_o‖₂/RMS(x^(j)) at every layer of Qwen-7B and Llama-8B. Supporting analysis: if ρ<1 uniformly, corridor perturbations decay exponentially. Determines whether the toy-model corridor amplification result transfers to production. | open |
| 11 | **Rename to CommitLLM** — all docs, APIs, CLI, scripts, packages use `CommitLLM`. | open |
| 12 | **Remove V4/V5 naming** — public surface uses one canonical release name, not internal version numbers | open |
| 13 | **Clean repo** — delete stale corridor/debug paths, one-off hooks, duplicate helpers, obsolete protocol-era code. Repo should reflect the kept path, not an archaeology of experiments. | open |
| 14 | **GPU smoke tests** — maintained GPU tests exercise token-0 attention replay and deep-prefix on the canonical verifier | open |
| 15 | **Freeze canonical verifier** — canonical verifier is the frozen trusted path; legacy verifier no longer needed. Fresh real-GPU confirmation via `test_e2e_v4.py` and `test_adversarial.py`. | partial |
| 16 | **Adversarial methodology research** — move from defender testing (flip bits, check rejection) to attacker simulation (white-box forgery, adaptive adversary, composition attacks, probabilistic security curves). See [`research/adversarial-methodology.md`](./research/adversarial-methodology.md). | open |
| 17 | **Fuzz binary parsers** — coverage-guided fuzzing (cargo-fuzz) on verifier's binary deserialization. Malformed receipts fail closed, never panic. | open |
| 18 | **Stable benchmark protocol** — fix workload corpus, model/settings, warmup, hardware class, reporting format | open |
| 19 | **Benchmark routine-audit path** — baseline, online overhead, commit time, audit-open time, verifier time, retained-state size, payload size | open |
| 20 | **Paper/README/article consistency** — claims match code match measurements. See [Docs detail](#docs--publication-detail) below. | open |
| 21 | **v1.0 release** — cut clean tagged release only when code, benchmarks, docs, and claims all line up | open |

---

## Tier 1 — Security hardening

Blocks credibility with serious reviewers.

| # | Item |
|---|------|
| 22 | **Trust assumptions review** — enumerate exact, approximate, statistical, fail-closed, and out-of-scope assumptions; check against code and claims |
| 23 | **Freshness / temporal binding** — verifier-issued nonce or timestamp in receipts; prevent replay of cached honest responses |
| 24 | **Tolerance bounds** — analytical per-step bound for the attention corridor. Empirical data exists (Qwen L∞=8, Llama L∞=9); formal bound does not. Theorem target: `max_abs_diff ≤ 1` in i8 space given `‖Δa_f‖∞ < scale_a`. |
| 25 | **Audit tiers** — formalize receipt-only / routine / deep / full with explicit coverage and cost per tier |
| 26 | **Guarantee language** — docs clearly separate exact (Freivalds/INT8), bounded approximate (attention), statistical (sampled provenance), and fail-closed |
| 27 | **Verifier-key distribution** — canonical procedure for trusted key provenance, hash pinning, historical lookup, fail-closed on unknown keys |
| 28 | **Key rotation** — key versions tracked; old receipts remain auditable with historical key |
| 29 | **Verifier-secret randomness** — deep-audit batching uses verifier-only randomness the prover cannot predict before commitment |

---

## Tier 2 — Model breadth

Without broader model support, CommitLLM is a Qwen+Llama demo, not a product.

| # | Item |
|---|------|
| 30 | **MoE support** — verify expert routing decisions and per-expert shell matmuls for Mixtral, DeepSeek-V2/V3, Qwen-MoE. Commit router logits/top-k, Freivalds on selected expert weights, fail-closed on unsupported routing. |
| 31 | **FP8 quantization** — validated capture/replay/verification for FP8 (E4M3/E5M2) on Hopper/Blackwell. Capture layer needs FP8 `scaled_mm` wrappers; verifier needs FP8 dequant and bridge replay. |
| 32 | **Third family** — Mistral, Gemma, or strongest alternative measured on the kept path. Proves not overfit to two architectures. |
| 33 | **Larger model** — at least one 30B/70B-class datapoint on the corrected path |
| 34 | **Long-context 128K+** — validate corridor at production context lengths where attention numerics are most stressed |
| 35 | **GPTQ/AWQ/grouped quant** — at least one non-W8A8 family with a fully validated verifier replay path |
| 36 | **Verification profiles** — per-family configuration (tolerance table, context lengths, audit policy) under the same core protocol |

---

## Tier 3 — Performance & serving compatibility

Determines whether CommitLLM is deployable at production throughput.

| # | Item |
|---|------|
| 37 | **Native capture backend** — C++/CUDA/Triton hot path replacing Python interception. Milestones: (a) profile dominant cost, (b) prototype native `cutlass_scaled_mm` hook with ring buffer, (c) benchmark vs Python path, (d) integrate. Also enables CUDA graph compat (#45). |
| 38 | **Cross-request prefix caching** — committed prefix receipt referenced by subsequent requests. Highest-impact unsupported optimization (~2-5x TTFT). Prefix cache becomes first-class committed object. |
| 39 | **Test continuous batching** — tracer splits batched prefill into per-token traces; commit+audit+verify on real GPU with concurrent requests |
| 40 | **Test paged attention** — paged KV doesn't interfere with committed KV transcript integrity |
| 41 | **Test tensor parallelism** — TP=2 and TP=4 produce valid receipts. Startup hook patches TP workers but needs real multi-GPU confirmation. |
| 42 | **Test fused kernels** — fused QKV and gate-up projections produce correct shell accumulators, pass Freivalds |
| 43 | **Test FlashAttention** — attention outputs within established corridor bounds with FA enabled |
| 44 | **Test more W8A8 families** — overlaps with #32 and #33 |
| 45 | **CUDA graph compatibility** — capture hooks survive graph replay. Largely solved by #37. Impact: ~10-30% decode latency. |
| 46 | **Speculative decoding** — commit draft model identity, draft tokens, acceptance masks. New sub-protocol for accept/reject transcript. Impact: ~2-3x generation latency. |
| 47 | **LoRA/adapter verification** — verifier confirms served weights match `base + adapter`. Model spec already hashes adapter identity. |
| 48 | **Pipeline parallelism** — PP configs produce valid receipts. Capture layer handles cross-GPU activation transfers. |
| 49 | **Lower online overhead** — complete remaining cost work or consciously stop if returns are too small |
| 50 | **Reduce audit payload** — lossless shell compression: width-packed accumulators, tighter binary layouts. Don't drop shell matrices. |

---

## Tier 4 — Production infrastructure

Needed for real multi-tenant deployment.

| # | Item |
|---|------|
| 51 | **Streaming / SSE** — incremental commitments during token-by-token streaming; partial receipts before generation completes |
| 52 | **Multi-turn binding** — chain receipts across conversation turns (hash of prior receipt in next commitment) |
| 53 | **Monitoring / observability** — export metrics: failure rate, corridor histograms, audit latency, buffer utilization, receipt throughput (Prometheus/OTel) |
| 54 | **Receipt storage** — indexing by request ID, time, model, verification status. Append-only log, object store, or blockchain anchoring. |
| 55 | **Verifier deployment** — hosted service vs client-side library vs hybrid. Address key distribution, receipt transport, offline vs real-time. |
| 56 | **Receipt compression** — compact aggregate receipts or compressed streams at thousands of req/sec |
| 57 | **Audit delegation** — verifier delegates authority to trusted third-party auditor. Define trust model, revocation. |
| 58 | **OpenAI-compatible proxy** — standard API with receipt metadata, challenge endpoint, verifier CLI |
| 59 | **Client SDKs** — Python and TypeScript wrappers for receipt handling and verification |
| 60 | **Reference client flow** — one canonical CLI: trusted key → receipt → served output → challenge → audit → coverage interpretation |
| 61 | **Startup self-checks** — fail closed on model-identity mismatch, bad capture settings, unsupported config |
| 62 | **Architecture detection** — unsupported families detected explicitly, not silently treated as Llama-style |
| 63 | **Buffer-pressure behavior** — capture and audit-state buffer overflow is explicit and tested |
| 64 | **Retention behavior** — audit-state TTL, cleanup, and purge are explicit and tested |
| 65 | **Health checks** — server reports capture hook, audit buffer, model identity, verified-mode health |
| 66 | **Abuse protection** — audit requests rate-limited, bounds-checked, no amplification |
| 67 | **EOS-trim evidence** — real-GPU test: EOS-before-max_tokens triggers trim, commit+audit+verify passes |

---

## Tier 5 — Strengthening & research

Important for the long-term story but not blocking anything above.

| # | Item |
|---|------|
| 68 | **Deterministic inference mode** — eager attention (`attn_implementation="eager"`), `torch.use_deterministic_algorithms(True)`, fixed CUDA seeds. If the forward pass is bit-exact and the verifier replays with the same arithmetic spec, τ=0 closes the attention gap for opened tokens without score witnessing. Requires a defined arithmetic/kernel spec for cross-hardware determinism — "deterministic on one GPU" is not enough. Note: τ=0 alone does not solve prefix anchoring; replay only proves consistency with committed prefix, not true earlier execution. KV provenance (#6) or deep audit still needed for upstream trust. Measure throughput cost vs FlashAttention. |
| 69 | **Deterministic attention kernel** — custom Triton/CUDA attention with fixed accumulation order, deterministic reductions. Measure corridor=0. May be slower than FlashAttention. |
| 70 | **Cheating-incentive analysis** — quantify cost/benefit of model substitution, detection probability vs audit sampling rate, equilibrium conditions |
| 71 | **Lean formalization** — machine-checked core verification claims |
| 72 | **Non-Rust verifier** — independent implementation consuming golden vectors |
| 73 | **llama.cpp plugin** — capture retained state from llama.cpp serving path |
| 74 | **Marketplace integration** — providers attach receipts, clients challenge through market surface |
| 75 | **Receipt encryption** — encrypt receipts/audit payloads to verifier's public key |
| 76 | **Generalized retained state** — schema-driven format beyond constant-width decoder models |
| 77 | **Full shell recomputation** — bandwidth-light, verifier-heavy mode. Future experiment requiring local weights. |
| 78 | **Receipt-format spec** — binary format documented independently of implementation |
| 79 | **API extensions** — serving API carries receipt metadata in stable, documented way |
| 80 | **Remove legacy verifier** — delete `verify_v4_legacy` after soak period |

---

## Benchmarks checklist

All benchmark items in one place for tracking.

| # | Item |
|---|------|
| 81 | Stable benchmark protocol |
| 82 | Record baselines before/after each runtime-affecting milestone |
| 83 | Periodic remote-GPU benchmark checkpoints |
| 84 | Treat unexplained regressions as blockers |
| 85 | Benchmark deep-audit open cost |
| 86 | Benchmark routine-audit path |
| 87 | Benchmark exact-prefix path |
| 88 | Inspect binary payload by field |
| 89 | Measure retained-state memory and audit bandwidth |
| 90 | Measure audit-window storage costs |
| 91 | Calibrate routine-audit detection probabilities |
| 92 | Benchmark batch verification |
| 93 | Rebenchmark after each online-path change |
| 94 | Run final kept-path benchmark suite |
| 95 | Explain benchmark methodology in paper and docs |

---

## Docs / publication detail

| # | Item |
|---|------|
| 96 | Make implementation-status split explicit |
| 97 | Production verification surface is binary-only |
| 98 | Freeze verifier-facing report contract |
| 99 | Paper/README/article normative to final protocol |
| 100 | Update with landed corridor evidence (Qwen L∞=8, Llama L∞=9) |
| 101 | Full protocol documentation in README |
| 102 | Full protocol specification in paper |
| 103 | Explicit input-verification procedure |
| 104 | Transcript-chain and anti-splice procedure |
| 105 | Canonical deep-audit procedure |
| 106 | Decode/output support matrix in paper |
| 107 | One canonical randomness story |
| 108 | Document verified-mode deviations from stock vLLM |
| 109 | Document supported/unsupported architectures |
| 110 | Document canonical semantics vs trust assumptions |
| 111 | Explicit trusted-assumptions section |
| 112 | Fail-on-unknown versioning rules |
| 113 | Document privacy implications of receipts and audits |
| 114 | Pipeline / boundary figure |
| 115 | Update article to match full protocol |

---

## Completed

| # | Item |
|---|------|
| — | ✓ Make quantization semantics first-class protocol data |
| — | ✓ Fix bridge trust boundary for attention (`x_attn_i8`, `scale_x_attn`, check-and-gate, `vi-retained-v3`) |
| — | ✓ Add committed KV transcript with `kv_root` covering full causal prefix |
| — | ✓ Clean old attention-gap measurement strategy (no confounded L∞=127/255, fail-closed on missing data, clean naming) |
