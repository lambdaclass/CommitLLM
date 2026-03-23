# Fuzzing vi-core

Install cargo-fuzz: `cargo install cargo-fuzz`

Run a target:
```
cd crates/vi-core
cargo fuzz run fuzz_deserialize_compact_batch
cargo fuzz run fuzz_deserialize_key
cargo fuzz run fuzz_deserialize_trace
cargo fuzz run fuzz_decompress
```
