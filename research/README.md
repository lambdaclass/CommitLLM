# Research

Research documents for CommitLLM. These define open questions, methodology, and analysis plans that feed into the roadmap.

Unlike `redteam/` (which contains executable attack campaigns), this directory contains written analysis and research agendas.

## Documents

- [`decode-boundary.md`](./decode-boundary.md) — Decode-side research history. Shows why quantized replay failed, why LP-hidden was the right intermediate boundary, and why `CapturedLogits` is now the kept exact sampled-decode path.
- [`attention-gap.md`](./attention-gap.md) — Attention-corridor measurements and why arbitrary-position stock-kernel attention replay remains open.
- [`exact-arithmetic.md`](./exact-arithmetic.md) — Exact integer accumulation as a fix for the GPU-vs-CPU seam: integer / exact-accumulation attention accuracy (Qwen2.5-7B, Llama-3.1-8B), CPU vs Apple-GPU bit-exactness, and a full exact forward pass where decode and one-pass recompute are bit-identical. Scripts in [`exact-arithmetic/`](./exact-arithmetic/).
- [`adversarial-methodology.md`](./adversarial-methodology.md) — How to move from defender testing to attacker simulation. Defines goal-oriented attacks, adaptive adversary research, composition attacks, probabilistic security analysis, and protocol-level attacks. Roadmap #114.
