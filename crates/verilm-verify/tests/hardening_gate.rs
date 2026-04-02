//! Adversarial hardening gate (roadmap #4).
//!
//! This meta-test defines the hardening gate: the set of test suites that
//! must pass before any strong protocol claim or final benchmark can land.
//!
//! The gate is NOT a test suite itself — it verifies that the required
//! hardening suites exist and are not accidentally empty or removed.
//!
//! # Gate definition
//!
//! The hardening gate consists of:
//!
//! ## Local (Rust, `make hardening-gate`)
//!
//! | Suite                           | Crate              | What it covers                              |
//! |---------------------------------|--------------------|---------------------------------------------|
//! | `boundary_fuzz.rs`              | verilm-verify      | Malformed inputs, edge cases, EOS, coverage |
//! | `cross_version.rs`              | verilm-verify      | Frozen fixtures, format stability, compat   |
//! | `v4_e2e.rs`                     | verilm-verify      | Protocol correctness, tamper detection      |
//! | `golden_conformance.rs`         | verilm-test-vectors| Hash/challenge/commitment pinning           |
//! | `weight_chain_adversarial.rs`   | verilm-test-vectors| Weight substitution attacks                 |
//! | `fiat_shamir_soundness.rs`      | verilm-test-vectors| Challenge binding, domain separation        |
//! | `quantization_parity.rs`        | verilm-core        | Requantize/SiLU boundary correctness        |
//!
//! ## Remote (GPU, `make redteam-gpu` or `make gpu-test-adversarial`)
//!
//! | Suite                           | Location           | What it covers                              |
//! |---------------------------------|--------------------|---------------------------------------------|
//! | `test_adversarial.py`           | redteam/modal/     | 36 tamper/splice/cross-proof GPU scenarios  |
//!
//! # Gate rule
//!
//! No strong claims (paper assertions, README guarantees) or final benchmarks
//! may land unless:
//!
//! 1. `make hardening-gate` passes (all local Rust suites)
//! 2. `make redteam-gpu` passes (GPU adversarial suite) — when
//!    the change touches the verification path, capture path, or protocol
//!    semantics
//!
//! This is enforced by convention, not CI. The gate exists so that the
//! check is explicit, named, and runnable rather than implicit.

/// Verify the dedicated red-team surface exists.
#[test]
fn gate_redteam_surface_exists() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap();
    assert!(workspace.join("redteam/README.md").exists(), "redteam README must exist");
    assert!(workspace.join("redteam/attack_matrix.md").exists(), "redteam attack matrix must exist");
    assert!(workspace.join("redteam/modal/test_model_substitution.py").exists(),
        "redteam model-substitution runner must exist");
    assert!(workspace.join("redteam/modal/test_freshness_gap.py").exists(),
        "redteam freshness-gap runner must exist");
}

/// Verify the boundary_fuzz suite exists and has the expected test count.
#[test]
fn gate_boundary_fuzz_exists() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/boundary_fuzz.rs");
    let content = std::fs::read_to_string(path).expect("boundary_fuzz.rs must exist");
    let test_count = content.matches("#[test]").count();
    assert!(
        test_count >= 50,
        "boundary_fuzz.rs has only {} tests — expected >= 50",
        test_count
    );
}

/// Verify the cross_version suite exists with frozen fixtures.
#[test]
fn gate_cross_version_exists() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cross_version.rs");
    let content = std::fs::read_to_string(path).expect("cross_version.rs must exist");
    let test_count = content.matches("#[test]").count();
    assert!(
        test_count >= 15,
        "cross_version.rs has only {} tests — expected >= 15",
        test_count
    );

    // Frozen fixtures must be present.
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    assert!(
        std::path::Path::new(fixtures).join("v4_audit_canonical.bin").exists(),
        "frozen audit fixture missing"
    );
    assert!(
        std::path::Path::new(fixtures).join("v4_key_canonical.bin").exists(),
        "frozen key fixture missing"
    );
}

/// Verify the v4_e2e suite exists and has substantial coverage.
#[test]
fn gate_v4_e2e_exists() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/v4_e2e.rs");
    let content = std::fs::read_to_string(path).expect("v4_e2e.rs must exist");
    let test_count = content.matches("#[test]").count();
    assert!(
        test_count >= 100,
        "v4_e2e.rs has only {} tests — expected >= 100",
        test_count
    );
}

/// Verify the GPU adversarial suite exists and has substantial assertions.
#[test]
fn gate_gpu_adversarial_exists() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap();
    let path = workspace.join("redteam/modal/test_adversarial.py");
    assert!(path.exists(), "redteam/modal/test_adversarial.py must exist");

    let content = std::fs::read_to_string(&path).unwrap();
    // The adversarial suite is a single-file script with inline assertions,
    // not pytest. Count assert statements as a proxy for scenario coverage.
    let assert_count = content.matches("assert ").count()
        + content.matches("assert(").count();
    assert!(
        assert_count >= 10,
        "redteam/modal/test_adversarial.py has only {} assertions — expected >= 10",
        assert_count
    );
}

/// Verify the FailureCode enum has sufficient coverage for gate assertions.
#[test]
fn gate_failure_taxonomy_exists() {
    // The taxonomy must have codes for all major failure categories.
    // This is a compile-time guarantee via the enum, but we verify the
    // categories are all represented at runtime.
    use verilm_verify::{FailureCategory, FailureCode};

    let categories: Vec<FailureCategory> = [
        FailureCode::WrongCommitmentVersion,  // Structural
        FailureCode::FreivaldsFailed,          // CryptographicBinding
        FailureCode::SpecFieldMismatch,        // SpecMismatch
        FailureCode::UnsupportedSamplerVersion,// Unsupported
        FailureCode::TokenSelectionMismatch,   // SemanticViolation
        FailureCode::TokenizerError,           // Operational
    ]
    .iter()
    .map(|c| c.category())
    .collect();

    // All 6 categories must be represented.
    assert!(categories.contains(&FailureCategory::Structural));
    assert!(categories.contains(&FailureCategory::CryptographicBinding));
    assert!(categories.contains(&FailureCategory::SpecMismatch));
    assert!(categories.contains(&FailureCategory::Unsupported));
    assert!(categories.contains(&FailureCategory::SemanticViolation));
    assert!(categories.contains(&FailureCategory::Operational));
}

/// Verify AuditCoverage distinguishes routine from full.
#[test]
fn gate_coverage_semantics_exist() {
    let full = verilm_verify::AuditCoverage::Full { layers_checked: 32 };
    let routine = verilm_verify::AuditCoverage::Routine {
        layers_checked: 4,
        layers_total: 32,
    };
    assert_ne!(full, routine, "coverage levels must be distinguishable");
}
