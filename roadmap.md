# Roadmap

All remaining work for CommitLLM, organized by priority. Item numbers are stable identifiers — they don't change when priority changes.

Mark a task done only when it is true in code, tests, docs, and operational behavior. Completed milestones live in [CHANGELOG.md](./CHANGELOG.md).

The shell/tail/binding protocol is structurally complete. The kept path preserves the full exact 7-matrix Freivalds shell (`Wq/Wk/Wv/Wo/Wg/Wu/Wd`) across all audit tiers; tiers differ in attention evidence and payload cost, not in whether the exact linear shell is checked.

---

## Tier 0 — Publication blockers

These gate the first public release. Do them in this order.

| # | Item | Status |
|---|------|--------|
| 114 | **Adversarial methodology research** — move from defender testing (flip bits, check rejection) to attacker simulation (white-box forgery, adaptive adversary, composition attacks, probabilistic security curves). See [`research/adversarial-methodology.md`](./research/adversarial-methodology.md). | open |
| 49 | **Rename to CommitLLM** — all docs, APIs, CLI, scripts, packages use `CommitLLM`. Do first. | open |
| 50 | **Remove V4/V5 naming** — public surface uses one canonical release name, not internal version numbers | open |
| 6 | **Unify attention path** — one `x_attn` boundary, one QKV replay story across prover/verifier/corridor. No silent mixing of bridge-derived Q with committed K/V. | open |
| 11 | **Clean repo** — delete stale corridor/debug paths, one-off hooks, duplicate helpers, obsolete protocol-era code. Repo should reflect the kept path, not an archaeology of experiments. | open |
| 14 | **GPU smoke tests** — maintained GPU tests exercise token-0 attention replay and deep-prefix on the canonical verifier | open |
| 13 | **Freeze canonical verifier** — canonical verifier is the frozen trusted path; legacy verifier no longer needed. Fresh real-GPU confirmation via `test_e2e_v4.py` and `test_adversarial.py`. | partial |
| 97 | **Adversarial testing** — cheating-provider test suite: receipt forgery, transcript splicing, intermediate tampering, model substitution, selective layer cheating, KV injection. Every attack must fail verification or be documented as accepted gap. | **in progress** |
| 99 | **Fuzz binary parsers** — coverage-guided fuzzing (cargo-fuzz) on verifier's binary deserialization. Malformed receipts fail closed, never panic. | open |
| 33 | **Stable benchmark protocol** — fix workload corpus, model/settings, warmup, hardware class, reporting format | open |
| 38 | **Benchmark routine-audit path** — baseline, online overhead, commit time, audit-open time, verifier time, retained-state size, payload size | open |
| 55–71 | **Paper/README/article consistency** — claims match code match measurements. See [Docs detail](#docs--publication-detail) below. | open |
| 74 | **Publish** — cut clean release only when code, benchmarks, docs, and claims all line up | open |

---

## Tier 1 — Security foundation

Don't block release but block credibility with serious reviewers. Do immediately after publication.

| # | Item |
|---|------|
| 98 | **Formal security argument** — define the cheating game, state Freivalds soundness bound (1/p), state detection probability as f(sampling rate, cheating fraction), state what commitment binding prevents |
| 18 | **Trust assumptions review** — enumerate exact, approximate, statistical, fail-closed, and out-of-scope assumptions; check against code and claims |
| 106 | **Freshness / temporal binding** — verifier-issued nonce or timestamp in receipts; prevent replay of cached honest responses |
| 5 | **Tolerance bounds** — analytical per-step bound for the attention corridor. Empirical data exists (Qwen L∞=8, Llama L∞=9); formal bound does not. Theorem target: `max_abs_diff ≤ 1` in i8 space given `‖Δa_f‖∞ < scale_a` |
| 12 | **KV provenance** — sampled/batched Freivalds checks verify committed K/V are consistent with committed Wk/Wv weights and bridge inputs |
| 19 | **Audit tiers** — formalize receipt-only / routine / deep / full with explicit coverage and cost per tier |
| 20 | **Guarantee language** — docs clearly separate exact (Freivalds/INT8), bounded approximate (attention), statistical (sampled provenance), and fail-closed |
| 15 | **Verifier-key distribution** — canonical procedure for trusted key provenance, hash pinning, historical lookup, fail-closed on unknown keys |
| 16 | **Key rotation** — key versions tracked; old receipts remain auditable with historical key |
| 17 | **Verifier-secret randomness** — deep-audit batching uses verifier-only randomness the prover cannot predict before commitment |

---

## Tier 2 — Model breadth

Without broader model support, CommitLLM is a Qwen+Llama demo, not a product.

| # | Item |
|---|------|
| 100 | **MoE support** — verify expert routing decisions and per-expert shell matmuls for Mixtral, DeepSeek-V2/V3, Qwen-MoE. Commit router logits/top-k, Freivalds on selected expert weights, fail-closed on unsupported routing. |
| 101 | **FP8 quantization** — validated capture/replay/verification for FP8 (E4M3/E5M2) on Hopper/Blackwell. Capture layer needs FP8 `scaled_mm` wrappers; verifier needs FP8 dequant and bridge replay. |
| 8 | **Third family** — Mistral, Gemma, or strongest alternative measured on the kept path. Proves not overfit to two architectures. |
| 7 | **Larger model** — at least one 30B/70B-class datapoint on the corrected path |
| 103 | **Long-context 128K+** — validate corridor at production context lengths where attention numerics are most stressed |
| 95 | **GPTQ/AWQ/grouped quant** — at least one non-W8A8 family with a fully validated verifier replay path |
| 21 | **Verification profiles** — per-family configuration (tolerance table, context lengths, audit policy) under the same core protocol |

---

## Tier 3 — Performance & serving compatibility

Determines whether CommitLLM is deployable at production throughput.

| # | Item |
|---|------|
| 113 | **Native capture backend** — C++/CUDA/Triton hot path replacing Python interception. Milestones: (a) profile dominant cost, (b) prototype native `cutlass_scaled_mm` hook with ring buffer, (c) benchmark vs Python path, (d) integrate. Also enables CUDA graph compat (#93). Distinct from deterministic attention kernel (#22) which targets exactness. |
| 92 | **Cross-request prefix caching** — committed prefix receipt referenced by subsequent requests. Highest-impact unsupported optimization (~2-5x TTFT). Prefix cache becomes first-class committed object. |
| 86 | **Test continuous batching** — tracer splits batched prefill into per-token traces; commit+audit+verify on real GPU with concurrent requests |
| 87 | **Test paged attention** — paged KV doesn't interfere with committed KV transcript integrity |
| 88 | **Test tensor parallelism** — TP=2 and TP=4 produce valid receipts. Startup hook patches TP workers but needs real multi-GPU confirmation. |
| 89 | **Test fused kernels** — fused QKV and gate-up projections produce correct shell accumulators, pass Freivalds |
| 90 | **Test FlashAttention** — attention outputs within established corridor bounds with FA enabled |
| 91 | **Test more W8A8 families** — overlaps with #7 and #8 |
| 93 | **CUDA graph compatibility** — capture hooks survive graph replay. Largely solved by #113. Impact: ~10-30% decode latency. |
| 94 | **Speculative decoding** — commit draft model identity, draft tokens, acceptance masks. New sub-protocol for accept/reject transcript. Impact: ~2-3x generation latency. |
| 96 | **LoRA/adapter verification** — verifier confirms served weights match `base + adapter`. Model spec already hashes adapter identity. |
| 102 | **Pipeline parallelism** — PP configs produce valid receipts. Capture layer handles cross-GPU activation transfers. |
| 47 | **Lower online overhead** — complete remaining cost work or consciously stop if returns are too small |
| 45 | **Reduce audit payload** — lossless shell compression: width-packed accumulators, tighter binary layouts. Don't drop shell matrices. |

---

## Tier 4 — Production infrastructure

Needed for real multi-tenant deployment.

| # | Item |
|---|------|
| 104 | **Streaming / SSE** — incremental commitments during token-by-token streaming; partial receipts before generation completes |
| 105 | **Multi-turn binding** — chain receipts across conversation turns (hash of prior receipt in next commitment) |
| 107 | **Monitoring / observability** — export metrics: failure rate, corridor histograms, audit latency, buffer utilization, receipt throughput (Prometheus/OTel) |
| 108 | **Receipt storage** — indexing by request ID, time, model, verification status. Append-only log, object store, or blockchain anchoring. |
| 109 | **Verifier deployment** — hosted service vs client-side library vs hybrid. Address key distribution, receipt transport, offline vs real-time. |
| 110 | **Receipt compression** — compact aggregate receipts or compressed streams at thousands of req/sec |
| 111 | **Audit delegation** — verifier delegates authority to trusted third-party auditor. Define trust model, revocation. |
| 78 | **OpenAI-compatible proxy** — standard API with receipt metadata, challenge endpoint, verifier CLI |
| 77 | **Client SDKs** — Python and TypeScript wrappers for receipt handling and verification |
| 54 | **Reference client flow** — one canonical CLI: trusted key → receipt → served output → challenge → audit → coverage interpretation |
| 26 | **Startup self-checks** — fail closed on model-identity mismatch, bad capture settings, unsupported config |
| 27 | **Architecture detection** — unsupported families detected explicitly, not silently treated as Llama-style |
| 28 | **Buffer-pressure behavior** — capture and audit-state buffer overflow is explicit and tested |
| 29 | **Retention behavior** — audit-state TTL, cleanup, and purge are explicit and tested |
| 30 | **Health checks** — server reports capture hook, audit buffer, model identity, verified-mode health |
| 31 | **Abuse protection** — audit requests rate-limited, bounds-checked, no amplification |
| 32 | **EOS-trim evidence** — real-GPU test: EOS-before-max_tokens triggers trim, commit+audit+verify passes |

---

## Tier 5 — Strengthening & research

Important for the long-term story but not blocking anything above.

| # | Item |
|---|------|
| 9 | **Score witnessing** — validate pre-softmax scores for challenged tokens against shell-Q and Merkle-bound K. Catches attention pattern manipulation without requiring FP softmax agreement. Cost: O(seq_len × n_heads) per audited token per layer. |
| 10 | **W_o conditioning** — compute σ_min(W_o) per layer; quantifies adversary freedom when attn_out_i8 is commitment-bound but not independently replayed |
| 22 | **Deterministic attention arithmetic** — canonical kernel or arithmetic spec making attention replay bit-exact. Only way to eliminate empirical τ entirely. Side experiment, not critical path. |
| 23 | **Prototype deterministic kernel** — custom Triton/CUDA attention with fixed accumulation order, deterministic reductions. Measure corridor=0. May be slower than FlashAttention. |
| 112 | **Cheating-incentive analysis** — quantify cost/benefit of model substitution, detection probability vs audit sampling rate, equilibrium conditions |
| 84 | **Lean formalization** — machine-checked core verification claims |
| 25 | **Non-Rust verifier** — independent implementation consuming golden vectors |
| 79 | **llama.cpp plugin** — capture retained state from llama.cpp serving path |
| 83 | **Marketplace integration** — providers attach receipts, clients challenge through market surface |
| 81 | **Receipt encryption** — encrypt receipts/audit payloads to verifier's public key |
| 82 | **Generalized retained state** — schema-driven format beyond constant-width decoder models |
| 85 | **Full shell recomputation** — bandwidth-light, verifier-heavy mode. Future experiment requiring local weights. |
| 75 | **Receipt-format spec** — binary format documented independently of implementation |
| 76 | **API extensions** — serving API carries receipt metadata in stable, documented way |
| 24 | **Remove legacy verifier** — delete `verify_v4_legacy` after soak period |

---

## Benchmarks checklist

All benchmark items in one place for tracking.

| # | Item |
|---|------|
| 33 | Stable benchmark protocol |
| 34 | Record baselines before/after each runtime-affecting milestone |
| 35 | Periodic remote-GPU benchmark checkpoints |
| 36 | Treat unexplained regressions as blockers |
| 37 | Benchmark deep-audit open cost |
| 38 | Benchmark routine-audit path |
| 39 | Benchmark exact-prefix path |
| 40 | Inspect binary payload by field |
| 41 | Measure retained-state memory and audit bandwidth |
| 42 | Measure audit-window storage costs |
| 43 | Calibrate routine-audit detection probabilities |
| 44 | Benchmark batch verification |
| 48 | Rebenchmark after each online-path change |
| 72 | Run final kept-path benchmark suite |
| 73 | Explain benchmark methodology in paper and docs |

---

## Docs / publication detail

| # | Item |
|---|------|
| 51 | Make implementation-status split explicit |
| 52 | Production verification surface is binary-only |
| 53 | Freeze verifier-facing report contract |
| 55 | Paper/README/article normative to final protocol |
| 56 | Update with landed corridor evidence (Qwen L∞=8, Llama L∞=9) |
| 57 | Full protocol documentation in README |
| 58 | Full protocol specification in paper |
| 59 | Explicit input-verification procedure |
| 60 | Transcript-chain and anti-splice procedure |
| 61 | Canonical deep-audit procedure |
| 62 | Decode/output support matrix in paper |
| 63 | One canonical randomness story |
| 64 | Document verified-mode deviations from stock vLLM |
| 65 | Document supported/unsupported architectures |
| 66 | Document canonical semantics vs trust assumptions |
| 67 | Explicit trusted-assumptions section |
| 68 | Fail-on-unknown versioning rules |
| 69 | Document privacy implications of receipts and audits |
| 70 | Pipeline / boundary figure |
| 71 | Update article to match full protocol |

---

## Completed

| # | Item |
|---|------|
| 1 | ✓ Make quantization semantics first-class protocol data |
| 2 | ✓ Fix bridge trust boundary for attention (`x_attn_i8`, `scale_x_attn`, check-and-gate, `vi-retained-v3`) |
| 3 | ✓ Add committed KV transcript with `kv_root` covering full causal prefix |
| 4 | ✓ Clean old attention-gap measurement strategy (no confounded L∞=127/255, fail-closed on missing data, clean naming) |
