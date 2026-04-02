.PHONY: build test check clean hardening-gate gpu-test-adversarial redteam-gpu \
       redteam-model-substitution redteam-freshness-gap \
       paper paper-watch paper-clean paper-stamp lean-build lean-clean \
       gpu-test gpu-test-e2e gpu-test-sampled gpu-test-stability gpu-test-modal gpu-terminate \
       gpu-bench-hooks gpu-bench-hooks-modal gpu-bench-ab gpu-bench-ab-parallel gpu-bench-ab-modal gpu-bench-ab-terminate

# Rust
build:
	cargo build --workspace --exclude verilm-py

test:
	cargo test --workspace --exclude verilm-py

check:
	cargo check --workspace --exclude verilm-py

clean:
	cargo clean

# Adversarial hardening gate (roadmap #4)
# Must pass before any strong claim or final benchmark lands.
hardening-gate:
	@echo "=== Hardening gate: local Rust suites ==="
	cargo test -p verilm-verify --test hardening_gate
	cargo test -p verilm-verify --test boundary_fuzz
	cargo test -p verilm-verify --test cross_version
	cargo test -p verilm-verify --test v4_e2e
	cargo test -p verilm-test-vectors --test golden_conformance
	cargo test -p verilm-test-vectors --test weight_chain_adversarial
	cargo test -p verilm-test-vectors --test fiat_shamir_soundness
	cargo test -p verilm-core --test quantization_parity
	@echo "=== Hardening gate: PASSED ==="

# GPU adversarial suite (requires Modal or RunPod)
redteam-gpu:
	modal run --detach redteam/modal/test_adversarial.py

redteam-model-substitution:
	modal run --detach redteam/modal/test_model_substitution.py

redteam-freshness-gap:
	modal run --detach redteam/modal/test_freshness_gap.py

gpu-test-adversarial:
	$(MAKE) redteam-gpu

# GPU tests — RunPod (persistent pod, fast iteration)
# Requires: export RUNPOD_API_KEY=...
gpu-test: gpu-test-sampled

gpu-test-sampled:
	python scripts/runpod/test.py --script scripts/modal/test_sampled_decoding.py

gpu-test-e2e:
	python scripts/runpod/test.py --script scripts/modal/test_e2e_v4.py

gpu-test-stability:
	python scripts/runpod/test.py --script scripts/modal/test_capture_stability.py

gpu-terminate:
	python scripts/runpod/test.py --terminate

# GPU tests — Modal (ephemeral, no pod management)
gpu-test-modal:
	modal run --detach scripts/modal/test_sampled_decoding.py

gpu-test-e2e-modal:
	modal run --detach scripts/modal/test_e2e_v4.py

# Benchmarks
gpu-bench-hooks:
	python scripts/runpod/test.py --script scripts/modal/bench_hooks_overhead.py

gpu-bench-hooks-modal:
	modal run --detach scripts/modal/bench_hooks_overhead.py

gpu-bench-commit:
	python scripts/runpod/test.py --script scripts/modal/bench_commit_phases.py

gpu-bench-commit-modal:
	modal run --detach scripts/modal/bench_commit_phases.py

gpu-bench-ab:
	python scripts/runpod/test.py --script scripts/modal/bench_ab_overhead.py

gpu-bench-ab-parallel:
	python scripts/runpod/bench_parallel.py

gpu-bench-ab-modal:
	modal run --detach scripts/modal/bench_ab_overhead.py

gpu-bench-ab-terminate:
	python scripts/runpod/bench_parallel.py --terminate

# Paper
paper:
	typst compile paper/main.typ paper/main.pdf

paper-watch:
	typst watch paper/main.typ paper/main.pdf

paper-stamp: paper
	ots stamp paper/main.pdf

paper-clean:
	rm -f paper/main.pdf paper/main.pdf.ots

# Lean
lean-build:
	cd lean && lake build

lean-clean:
	rm -rf lean/.lake/build
