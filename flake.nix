{
  description = "VeriLM — provenance for open-weight LLM inference";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Rust
            pkgs.cargo
            pkgs.rustc
            pkgs.rust-analyzer

            # Paper
            pkgs.typst
            pkgs.opentimestamps-client

            # Lean
            pkgs.elan

            pkgs.git
          ];

          NIX_SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
          SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
        };
      });
}
