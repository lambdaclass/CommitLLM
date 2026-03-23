#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = verilm_core::serialize::deserialize_compact_batch(data);
});
