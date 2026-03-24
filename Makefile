.PHONY: build test check clean paper paper-watch paper-clean paper-stamp lean-build lean-clean \
       gpu-test gpu-test-e2e gpu-test-sampled gpu-test-stability gpu-test-modal gpu-terminate

# Rust
build:
	cargo build --workspace --exclude verilm-py

test:
	cargo test --workspace --exclude verilm-py

check:
	cargo check --workspace --exclude verilm-py

clean:
	cargo clean

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
