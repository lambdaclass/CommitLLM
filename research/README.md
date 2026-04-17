# Research

Research documents for CommitLLM. These define open questions, methodology, and analysis plans that feed into the roadmap.

Unlike `redteam/` (which contains executable attack campaigns), this directory contains written analysis and research agendas.

## Documents

- [`decode-boundary.md`](./decode-boundary.md) — Decode-side research history. Shows why quantized replay failed, why LP-hidden was the right intermediate boundary, and why `CapturedLogits` is now the kept exact sampled-decode path.
- [`attention-gap.md`](./attention-gap.md) — Attention-corridor measurements and why arbitrary-position stock-kernel attention replay remains open.
- [`adversarial-methodology.md`](./adversarial-methodology.md) — How to move from defender testing to attacker simulation. Defines goal-oriented attacks, adaptive adversary research, composition attacks, probabilistic security analysis, and protocol-level attacks. Roadmap #114.
