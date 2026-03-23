.PHONY: build test check clean paper paper-watch paper-clean paper-stamp lean-build lean-clean

# Rust
build:
	cargo build --workspace --exclude verilm-py

test:
	cargo test --workspace --exclude verilm-py

check:
	cargo check --workspace --exclude verilm-py

clean:
	cargo clean

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
