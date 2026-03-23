.PHONY: paper paper-watch paper-clean paper-stamp lean-build lean-clean

paper:
	typst compile paper/main.typ paper/main.pdf

paper-watch:
	typst watch paper/main.typ paper/main.pdf

paper-stamp: paper
	ots stamp paper/main.pdf

paper-clean:
	rm -f paper/main.pdf paper/main.pdf.ots

lean-build:
	cd lean && lake build

lean-clean:
	rm -rf lean/.lake/build
