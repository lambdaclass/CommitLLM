# Fuzzing verilm-core

Install cargo-fuzz: `cargo install cargo-fuzz`

Run a target:
```
cd crates/verilm-core
cargo fuzz run fuzz_deserialize_key
cargo fuzz run fuzz_decompress
```
