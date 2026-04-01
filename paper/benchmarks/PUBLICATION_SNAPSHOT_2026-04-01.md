# CommitLLM Paper Snapshot (2026-04-01)

This file freezes the benchmark/result artifacts that back the current public paper snapshot.

Release tag:
- `paper-2026-04-01`

Primary paper artifacts:
- [main.typ](/Users/unbalancedparen/projects/verishell/paper/main.typ)
- [main.pdf](/Users/unbalancedparen/projects/verishell/paper/main.pdf)

## Committed Raw Benchmark Artifacts

These files are committed directly in the repository and are the authoritative raw artifacts behind the cited benchmark tables.

- [verifier_bench.txt](/Users/unbalancedparen/projects/verishell/paper/benchmarks/verifier_bench.txt)
  - backs the verifier-cost table in the paper
  - key cited numbers:
    - Llama 3.1 70B, 10 layers: `1.27 ms`
    - Llama 3.1 70B, all 80 layers: `10.09 ms`
    - hardware: Apple M4 CPU

- [protocol_bench_toy.txt](/Users/unbalancedparen/projects/verishell/paper/benchmarks/protocol_bench_toy.txt)
  - backs size/key projections cited in the paper
  - key cited number:
    - projected verifier key size for Llama 70B: `48.14 MB`

## Script-Backed Paper Claims

The following paper claims are frozen by the committed measurement scripts plus the values reported in the paper. The raw run logs for these GPU measurements are not currently committed in this repository snapshot.

- Attention corridor:
  - script: [measure_corridor.py](/Users/unbalancedparen/projects/verishell/scripts/modal/measure_corridor.py)
  - environment: `A100-80GB`, vLLM eager mode
  - models:
    - `neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8`
    - `neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8`
  - frozen paper claims:
    - Qwen2.5-7B-W8A8: global max `L_inf = 8`, first-generated-token max `= 5`, decode max `= 8`, `frac_eq > 92%`, `frac<=1 > 99.8%`
    - Llama-3.1-8B-W8A8: global max `L_inf = 9`, first-generated-token max `= 5`, decode max `= 9`, `frac_eq ≈ 94–96%`, `frac<=1 > 99.9%`

- Provider serving overhead:
  - script: [bench_ab_overhead.py](/Users/unbalancedparen/projects/verishell/scripts/modal/bench_ab_overhead.py)
  - environment: `A100-80GB`, vLLM eager mode
  - frozen paper claims:
    - online generation overhead: `~12–14%`
    - post-generation finalization overhead is measured separately and larger

## Rerun Commands

- Paper build:
  - `make paper`

- Corridor measurements:
  - `modal run --detach scripts/modal/measure_corridor.py`

- A/B serving-overhead benchmark:
  - `modal run --detach scripts/modal/bench_ab_overhead.py`
  - or `make gpu-bench-ab-modal`

## Scope Note

This snapshot is intended to freeze the exact paper-backed repository state for publication. Follow-on work such as additional model families, larger checkpoints, stronger KV provenance, `W_o` conditioning, and score witnessing is intentionally out of scope for this snapshot.
