NIX := nix --extra-experimental-features 'nix-command flakes'

.PHONY: paper paper-watch paper-clean

paper:
	$(NIX) develop -c typst compile paper/main.typ paper/main.pdf

paper-watch:
	$(NIX) develop -c typst watch paper/main.typ paper/main.pdf

paper-clean:
	rm -f paper/main.pdf
