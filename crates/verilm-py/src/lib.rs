//! PyO3 bridge: expose the Rust commitment engine to Python.
//!
//! This replaces the HTTP path for Python → Rust communication.
//! The Python `verilm` package captures data, then calls these functions
//! to build commitments and generate proofs directly in-process.
//!
//! Usage from Python:
//! ```python
//! import verilm_rs
//!
//! # Build traces from captured data
//! traces = [...]  # list of list of LayerTrace dicts
//!
//! # Commit
//! state = verilm_rs.commit(
//!     traces=traces,
//!     token_ids=[42, 99],
//!     prompt=b"Hello",
//!     sampling_seed=bytes(32),
//!     manifest={"tokenizer_hash": "aa" * 32, "temperature": 0.0, ...},
//! )
//! commitment_json = state.commitment_json()
//!
//! # Open for audit
//! proof_json = state.open([0, 1])
//! proof_bytes = state.open_compact([0, 1])  # binary + zstd
//! ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use verilm_core::serialize;
use verilm_core::types::{
    AuditChallenge, AuditTier, BatchCommitment, BatchProof, CommitmentVersion,
    DeploymentManifest, LayerTrace, TokenTrace, VerificationPolicy, VerifierKey,
};
use verilm_prover::{commit_with_full_binding, open, FullBindingParams};

/// Extract a Vec<i8> from a Python list of ints.
fn extract_i8_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i8>> {
    if let Ok(bytes) = obj.extract::<Vec<i8>>() {
        return Ok(bytes);
    }
    // Try extracting from bytes object
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(b.as_bytes().iter().map(|&v| v as i8).collect());
    }
    Err(PyValueError::new_err("expected list of i8 or bytes"))
}

/// Extract a Vec<i32> from a Python list of ints.
fn extract_i32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
    obj.extract::<Vec<i32>>()
}

/// Extract a Vec<Vec<i8>> from a Python list of lists of ints.
fn extract_nested_i8(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<i8>>> {
    let list = obj.cast::<PyList>()?;
    let mut result = Vec::with_capacity(list.len());
    for item in list.iter() {
        result.push(extract_i8_vec(&item)?);
    }
    Ok(result)
}

/// Convert a Python dict to a LayerTrace.
fn dict_to_layer_trace(d: &Bound<'_, PyDict>) -> PyResult<LayerTrace> {
    Ok(LayerTrace {
        x_attn: extract_i8_vec(&d.get_item("x_attn")?.ok_or_else(|| PyValueError::new_err("missing x_attn"))?)?,
        q: extract_i32_vec(&d.get_item("q")?.ok_or_else(|| PyValueError::new_err("missing q"))?)?,
        k: extract_i32_vec(&d.get_item("k")?.ok_or_else(|| PyValueError::new_err("missing k"))?)?,
        v: extract_i32_vec(&d.get_item("v")?.ok_or_else(|| PyValueError::new_err("missing v"))?)?,
        a: extract_i8_vec(&d.get_item("a")?.ok_or_else(|| PyValueError::new_err("missing a"))?)?,
        attn_out: extract_i32_vec(&d.get_item("attn_out")?.ok_or_else(|| PyValueError::new_err("missing attn_out"))?)?,
        x_ffn: extract_i8_vec(&d.get_item("x_ffn")?.ok_or_else(|| PyValueError::new_err("missing x_ffn"))?)?,
        g: extract_i32_vec(&d.get_item("g")?.ok_or_else(|| PyValueError::new_err("missing g"))?)?,
        u: extract_i32_vec(&d.get_item("u")?.ok_or_else(|| PyValueError::new_err("missing u"))?)?,
        h: extract_i8_vec(&d.get_item("h")?.ok_or_else(|| PyValueError::new_err("missing h"))?)?,
        ffn_out: extract_i32_vec(&d.get_item("ffn_out")?.ok_or_else(|| PyValueError::new_err("missing ffn_out"))?)?,
        kv_cache_k: d.get_item("kv_cache_k")?
            .map(|v| extract_nested_i8(&v))
            .transpose()?
            .unwrap_or_default(),
        kv_cache_v: d.get_item("kv_cache_v")?
            .map(|v| extract_nested_i8(&v))
            .transpose()?
            .unwrap_or_default(),
        scale_x_attn: d.get_item("scale_x_attn")?
            .map(|v| v.extract())
            .transpose()?,
        scale_a: d.get_item("scale_a")?
            .map(|v| v.extract())
            .transpose()?,
        scale_x_ffn: d.get_item("scale_x_ffn")?
            .map(|v| v.extract())
            .transpose()?,
        scale_h: d.get_item("scale_h")?
            .map(|v| v.extract())
            .transpose()?,
        residual: d.get_item("residual")?
            .map(|v| v.extract::<Vec<f32>>())
            .transpose()?,
    })
}

/// Convert Python traces (list[list[dict]]) to Vec<Vec<LayerTrace>>.
fn extract_traces(traces: &Bound<'_, PyList>) -> PyResult<Vec<Vec<LayerTrace>>> {
    let mut result = Vec::with_capacity(traces.len());
    for token_layers in traces.iter() {
        let layers_list = token_layers.cast::<PyList>()?;
        let mut layers = Vec::with_capacity(layers_list.len());
        for layer_dict in layers_list.iter() {
            let d = layer_dict.cast::<PyDict>()?;
            layers.push(dict_to_layer_trace(d)?);
        }
        result.push(layers);
    }
    Ok(result)
}

/// Parse a manifest dict from Python.
fn extract_manifest(d: &Bound<'_, PyDict>) -> PyResult<DeploymentManifest> {
    let tokenizer_hash_hex: String = d
        .get_item("tokenizer_hash")?
        .ok_or_else(|| PyValueError::new_err("manifest missing tokenizer_hash"))?
        .extract()?;
    let bytes = hex::decode(&tokenizer_hash_hex)
        .map_err(|e| PyValueError::new_err(format!("invalid tokenizer_hash hex: {}", e)))?;
    if bytes.len() != 32 {
        return Err(PyValueError::new_err("tokenizer_hash must be 32 bytes (64 hex chars)"));
    }
    let mut tokenizer_hash = [0u8; 32];
    tokenizer_hash.copy_from_slice(&bytes);

    Ok(DeploymentManifest {
        tokenizer_hash,
        temperature: d.get_item("temperature")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0.0),
        top_k: d.get_item("top_k")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0),
        top_p: d.get_item("top_p")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(1.0),
        eos_policy: d.get_item("eos_policy")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_else(|| "stop".to_string()),
        weight_hash: d.get_item("weight_hash")?
            .map(|v| {
                let hex_str: String = v.extract()?;
                decode_hex32(&hex_str, "weight_hash")
            })
            .transpose()?,
        quant_hash: d.get_item("quant_hash")?
            .map(|v| {
                let hex_str: String = v.extract()?;
                decode_hex32(&hex_str, "quant_hash")
            })
            .transpose()?,
        system_prompt_hash: d.get_item("system_prompt_hash")?
            .map(|v| {
                let hex_str: String = v.extract()?;
                decode_hex32(&hex_str, "system_prompt_hash")
            })
            .transpose()?,
    })
}

/// Opaque handle to Rust BatchState + BatchCommitment.
///
/// Holds the full prover state in Rust memory. Python uses this to:
/// - Read the commitment (as JSON)
/// - Open proofs for challenged tokens
/// - Get KV roots for streaming verification
#[pyclass]
struct BatchState {
    inner: verilm_prover::BatchState,
    commitment: BatchCommitment,
}

#[pymethods]
impl BatchState {
    /// Get the commitment as a JSON string.
    fn commitment_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.commitment)
            .map_err(|e| PyValueError::new_err(format!("serialization error: {}", e)))
    }

    /// Get the Merkle root as hex.
    fn merkle_root_hex(&self) -> String {
        hex::encode(self.commitment.merkle_root)
    }

    /// Get the IO root as hex.
    fn io_root_hex(&self) -> String {
        hex::encode(self.commitment.io_root)
    }

    /// Get the manifest hash as hex (or None).
    fn manifest_hash_hex(&self) -> Option<String> {
        self.commitment.manifest_hash.map(hex::encode)
    }

    /// Number of tokens in the commitment.
    fn n_tokens(&self) -> u32 {
        self.commitment.n_tokens
    }

    /// Number of layers per token.
    fn n_layers(&self) -> usize {
        self.inner.all_layers.first().map_or(0, |layers| layers.len())
    }

    /// Per-token KV Merkle roots as hex strings.
    fn kv_roots_hex(&self) -> Vec<String> {
        self.inner
            .kv_merkle_roots
            .as_ref()
            .map(|roots| roots.iter().map(hex::encode).collect())
            .unwrap_or_default()
    }

    /// Open proofs for challenged token indices. Returns JSON string.
    fn open_json(&self, challenge_indices: Vec<u32>) -> PyResult<String> {
        let proof = open(&self.inner, &challenge_indices);
        serde_json::to_string(&proof)
            .map_err(|e| PyValueError::new_err(format!("serialization error: {}", e)))
    }

    /// Open proofs and return compact binary (zstd compressed).
    fn open_compact<'py>(&self, py: Python<'py>, challenge_indices: Vec<u32>) -> PyResult<Bound<'py, PyBytes>> {
        let proof = open(&self.inner, &challenge_indices);
        let compact_bytes = serialize::serialize_compact_batch(&proof);
        let compressed = serialize::compress(&compact_bytes);
        Ok(PyBytes::new(py, &compressed))
    }

    /// Open a stratified audit proof (compact binary, zstd-compressed).
    ///
    /// Args:
    ///     token_index: int — which token to audit
    ///     layer_indices: list[int] — which layers to open
    ///     tier: str — "routine" or "full"
    ///
    /// Returns:
    ///     bytes — zstd-compressed compact audit response.
    fn audit_stratified<'py>(&self, py: Python<'py>, token_index: u32, layer_indices: Vec<usize>, tier: String) -> PyResult<Bound<'py, PyBytes>> {
        let tier = match tier.as_str() {
            "routine" => AuditTier::Routine,
            "full" => AuditTier::Full,
            _ => return Err(PyValueError::new_err(format!("invalid tier: {}", tier))),
        };

        let challenge = AuditChallenge {
            token_index,
            layer_indices,
            tier,
        };

        let response = verilm_prover::build_audit_response_from_state(&self.inner, &challenge);
        let compact_bytes = serialize::serialize_compact_audit(&response);
        let compressed = serialize::compress(&compact_bytes);
        Ok(PyBytes::new(py, &compressed))
    }
}

/// Build a commitment from per-token per-layer traces.
///
/// Args:
///     traces: list[list[dict]] — per-token, per-layer trace dicts.
///         Each dict must have keys: x_attn, q, k, v, a, attn_out,
///         x_ffn, g, u, h, ffn_out. Optional: kv_cache_k, kv_cache_v.
///     token_ids: list[int] — emitted token IDs.
///     prompt: bytes — prompt text.
///     sampling_seed: bytes — 32-byte sampling seed.
///     manifest: dict — deployment manifest with keys:
///         tokenizer_hash (hex str), temperature, top_k, top_p, eos_policy.
///         Pass None to omit manifest binding.
///
/// Returns:
///     BatchState — opaque handle for reading commitment and opening proofs.
#[pyfunction]
#[pyo3(signature = (traces, token_ids, prompt, sampling_seed, manifest=None))]
fn commit(
    traces: &Bound<'_, PyList>,
    token_ids: Vec<u32>,
    prompt: Vec<u8>,
    sampling_seed: Vec<u8>,
    manifest: Option<&Bound<'_, PyDict>>,
) -> PyResult<BatchState> {
    let all_layers = extract_traces(traces)?;

    if sampling_seed.len() != 32 {
        return Err(PyValueError::new_err("sampling_seed must be exactly 32 bytes"));
    }
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&sampling_seed);

    let manifest_obj = manifest.map(extract_manifest).transpose()?;

    let (commitment, inner) = commit_with_full_binding(
        all_layers,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: &prompt,
            sampling_seed: seed,
            manifest: manifest_obj.as_ref(),
        },
    );

    Ok(BatchState { inner, commitment })
}

/// Build an audit challenge from a seed.
///
/// Args:
///     seed: bytes (32) — challenge derivation seed
///     n_tokens: int — number of tokens in the batch
///     n_layers: int — number of layers in the model
///     tier: str — "routine" or "full"
///
/// Returns:
///     dict with token_index (int), layer_indices (list[int]), tier (str)
#[pyfunction]
fn build_audit_challenge(
    py: Python<'_>,
    seed: Vec<u8>,
    n_tokens: u32,
    n_layers: usize,
    tier: String,
) -> PyResult<Bound<'_, PyDict>> {
    if seed.len() != 32 {
        return Err(PyValueError::new_err("seed must be exactly 32 bytes"));
    }
    let mut seed_arr = [0u8; 32];
    seed_arr.copy_from_slice(&seed);

    let tier_enum = match tier.as_str() {
        "routine" => AuditTier::Routine,
        "full" => AuditTier::Full,
        _ => return Err(PyValueError::new_err(format!("invalid tier: {}", tier))),
    };

    let challenge = verilm_verify::build_audit_challenge(&seed_arr, n_tokens, n_layers, tier_enum);

    let d = PyDict::new(py);
    d.set_item("token_index", challenge.token_index)?;
    d.set_item("layer_indices", challenge.layer_indices)?;
    d.set_item("tier", tier)?;
    Ok(d)
}

/// Helper: decode a 32-byte hex string into `[u8; 32]`.
fn decode_hex32(hex_str: &str, field_name: &str) -> PyResult<[u8; 32]> {
    let bytes = hex::decode(hex_str)
        .map_err(|e| PyValueError::new_err(format!("invalid {} hex: {}", field_name, e)))?;
    if bytes.len() != 32 {
        return Err(PyValueError::new_err(format!(
            "{} must be 32 bytes (64 hex chars), got {}",
            field_name,
            bytes.len()
        )));
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

/// Extract a `VerificationPolicy` from a Python dict.
///
/// Expected keys (all optional):
///   - `min_version`: int (1, 2, or 3)
///   - `expected_prompt_hash`: hex string (64 chars)
///   - `expected_manifest_hash`: hex string (64 chars)
fn extract_policy(d: &Bound<'_, PyDict>) -> PyResult<VerificationPolicy> {
    let min_version = d
        .get_item("min_version")?
        .map(|v| {
            let n: u32 = v.extract()?;
            match n {
                1 => Ok(CommitmentVersion::V1),
                2 => Ok(CommitmentVersion::V2),
                3 => Ok(CommitmentVersion::V3),
                _ => Err(PyValueError::new_err(format!(
                    "invalid min_version {}: expected 1, 2, or 3",
                    n
                ))),
            }
        })
        .transpose()?;

    let expected_prompt_hash = d
        .get_item("expected_prompt_hash")?
        .map(|v| {
            let hex_str: String = v.extract()?;
            decode_hex32(&hex_str, "expected_prompt_hash")
        })
        .transpose()?;

    let expected_manifest_hash = d
        .get_item("expected_manifest_hash")?
        .map(|v| {
            let hex_str: String = v.extract()?;
            decode_hex32(&hex_str, "expected_manifest_hash")
        })
        .transpose()?;

    Ok(VerificationPolicy {
        min_version,
        expected_prompt_hash,
        expected_manifest_hash,
    })
}

/// Format a list of failure strings into Python dicts with `description` field.
///
/// We parse failure strings that follow the `"layer N MatrixType: ..."` pattern
/// to extract structured `layer`, `matrix`, `kind` fields. For failures that
/// don't match that pattern, `layer` and `matrix` are `None`.
fn failures_to_py_list<'py>(
    py: Python<'py>,
    failures: &[String],
) -> PyResult<Bound<'py, PyList>> {
    let items = PyList::empty(py);
    for f in failures {
        let d = PyDict::new(py);
        d.set_item("description", f)?;

        // Try to parse structured failure: "layer N MatrixType: ..."
        if f.starts_with("layer ") {
            let rest = &f["layer ".len()..];
            if let Some(space_pos) = rest.find(' ') {
                if let Ok(layer_idx) = rest[..space_pos].parse::<usize>() {
                    d.set_item("layer", layer_idx)?;
                    let after_layer = &rest[space_pos + 1..];
                    if let Some(colon_pos) = after_layer.find(':') {
                        d.set_item("matrix", &after_layer[..colon_pos])?;
                        d.set_item("kind", after_layer[colon_pos + 1..].trim())?;
                    }
                }
            }
        }

        items.append(d)?;
    }
    Ok(items)
}

/// Verify a batch proof against a verifier key.
///
/// Args:
///     proof_json: JSON-serialized `BatchProof`.
///     key_json: JSON-serialized `VerifierKey`.
///     seed: hex string (64 chars) — challenge derivation seed.
///     challenge_k: number of tokens to challenge.
///     policy: optional dict with `min_version`, `expected_prompt_hash`,
///         `expected_manifest_hash`.
///
/// Returns:
///     dict with `passed` (bool), `failures` (list of dicts).
#[pyfunction]
#[pyo3(signature = (proof_json, key_json, seed, challenge_k, policy=None))]
fn verify_batch<'py>(
    py: Python<'py>,
    proof_json: &str,
    key_json: &str,
    seed: &str,
    challenge_k: u32,
    policy: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let proof: BatchProof = serde_json::from_str(proof_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize BatchProof: {}", e)))?;
    let key: VerifierKey = serde_json::from_str(key_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize VerifierKey: {}", e)))?;
    let seed_bytes = decode_hex32(seed, "seed")?;

    let policy_obj = policy.map(extract_policy).transpose()?.unwrap_or_default();

    let report =
        verilm_verify::verify_batch_with_policy(&key, &proof, seed_bytes, challenge_k, &policy_obj);

    let result = PyDict::new(py);
    result.set_item("passed", report.verdict == verilm_verify::Verdict::Pass)?;
    result.set_item("failures", failures_to_py_list(py, &report.failures)?)?;
    result.set_item("n_tokens", report.n_tokens)?;
    result.set_item("n_challenged", report.n_challenged)?;
    result.set_item("tokens_passed", report.tokens_passed)?;
    Ok(result)
}

/// Verify a single token trace against a verifier key.
///
/// Args:
///     trace_json: JSON-serialized `TokenTrace`.
///     key_json: JSON-serialized `VerifierKey`.
///
/// Returns:
///     dict with `passed` (bool), `failures` (list of dicts).
#[pyfunction]
fn verify_single<'py>(
    py: Python<'py>,
    trace_json: &str,
    key_json: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let trace: TokenTrace = serde_json::from_str(trace_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize TokenTrace: {}", e)))?;
    let key: VerifierKey = serde_json::from_str(key_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize VerifierKey: {}", e)))?;

    let report = verilm_verify::verify_trace(&key, &trace);

    let result = PyDict::new(py);
    result.set_item("passed", report.verdict == verilm_verify::Verdict::Pass)?;
    result.set_item("failures", failures_to_py_list(py, &report.failures)?)?;
    result.set_item("checks_run", report.checks_run)?;
    result.set_item("checks_passed", report.checks_passed)?;
    Ok(result)
}

#[pymodule]
fn verilm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(commit, m)?)?;
    m.add_function(wrap_pyfunction!(build_audit_challenge, m)?)?;
    m.add_function(wrap_pyfunction!(verify_batch, m)?)?;
    m.add_function(wrap_pyfunction!(verify_single, m)?)?;
    m.add_class::<BatchState>()?;
    Ok(())
}
