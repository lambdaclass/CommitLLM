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
use verilm_prover::{commit_with_full_binding, FullBindingParams};

/// Extract a Vec<i8> from bytes (fast path) or a Python list of ints (fallback).
fn extract_i8_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i8>> {
    // Fast path: bytes object — zero-alloc reinterpret u8 → i8.
    if let Ok(b) = obj.cast::<PyBytes>() {
        let src = b.as_bytes();
        let mut dst = Vec::with_capacity(src.len());
        // SAFETY: i8 and u8 have identical layout; this is a bitwise copy.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr() as *const i8, dst.as_mut_ptr(), src.len());
            dst.set_len(src.len());
        }
        return Ok(dst);
    }
    // Fallback: Python list of ints.
    if let Ok(v) = obj.extract::<Vec<i8>>() {
        return Ok(v);
    }
    Err(PyValueError::new_err("expected bytes or list of i8"))
}

/// Extract a Vec<i32> from bytes (fast path) or a Python list of ints (fallback).
fn extract_i32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
    // Fast path: bytes object — bulk reinterpret as native-endian i32.
    if let Ok(b) = obj.cast::<PyBytes>() {
        let src = b.as_bytes();
        if src.len() % 4 != 0 {
            return Err(PyValueError::new_err(
                format!("i32 bytes length {} not a multiple of 4", src.len()),
            ));
        }
        let n = src.len() / 4;
        let mut dst = Vec::with_capacity(n);
        // SAFETY: src.len() is a multiple of 4; native-endian bytes from .tobytes()
        // can be bulk-reinterpreted as i32. Single memcpy instead of per-element loop.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr() as *const i32, dst.as_mut_ptr(), n);
            dst.set_len(n);
        }
        return Ok(dst);
    }
    // Fallback: Python list of ints.
    obj.extract::<Vec<i32>>()
}

/// Extract a Vec<f32> from bytes (fast path) or a Python list of floats (fallback).
fn extract_f32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(b) = obj.cast::<PyBytes>() {
        let src = b.as_bytes();
        if src.len() % 4 != 0 {
            return Err(PyValueError::new_err(
                format!("f32 bytes length {} not a multiple of 4", src.len()),
            ));
        }
        let n = src.len() / 4;
        let mut dst = Vec::with_capacity(n);
        // SAFETY: same as extract_i32_vec — bulk reinterpret native-endian bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr() as *const f32, dst.as_mut_ptr(), n);
            dst.set_len(n);
        }
        return Ok(dst);
    }
    obj.extract::<Vec<f32>>()
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
            .map(|v| extract_f32_vec(&v))
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
        None, // dict path: no pre-computed token KVs
    );

    Ok(BatchState { inner, commitment })
}

/// Build a commitment directly from raw captures (bypasses Python trace building).
///
/// Takes the flat capture buffer from the Python hook and does trace
/// construction + commitment entirely in Rust. Eliminates the Python
/// build_layer_traces and serialize phases.
///
/// Args:
///     capture_inputs: list[bytes] — x_i8 tensors as bytes, in call order.
///     capture_accumulators: list[bytes] — acc_i32 tensors as bytes, in call order.
///     capture_scales: list[float] — per-capture activation scales.
///     fwd_batch_sizes: list[int] — batch size for each forward pass.
///     n_layers: int — number of transformer layers.
///     q_dim: int — Q dimension (num_heads * head_dim).
///     kv_dim: int — KV dimension (num_kv_heads * head_dim).
///     intermediate_size: int — FFN intermediate size (gate_up half).
///     level_c: bool — whether to accumulate KV cache snapshots.
///     token_ids: list[int] — emitted token IDs.
///     prompt: bytes — prompt text.
///     sampling_seed: bytes — 32-byte sampling seed.
///     manifest: dict — deployment manifest (optional).
///     residuals: list[bytes] — f32 residual tensors, per fwd_pass per layer (optional).
///
/// Returns:
///     BatchState — opaque handle for reading commitment and opening proofs.
#[pyfunction]
#[pyo3(signature = (
    capture_inputs,
    capture_accumulators,
    capture_scales,
    fwd_batch_sizes,
    n_layers,
    q_dim,
    kv_dim,
    intermediate_size,
    level_c,
    token_ids,
    prompt,
    sampling_seed,
    manifest = None,
    residuals = None,
))]
#[allow(clippy::too_many_arguments)]
fn commit_from_captures(
    capture_inputs: &Bound<'_, PyList>,
    capture_accumulators: &Bound<'_, PyList>,
    capture_scales: Vec<f32>,
    fwd_batch_sizes: Vec<usize>,
    n_layers: usize,
    q_dim: usize,
    kv_dim: usize,
    intermediate_size: usize,
    level_c: bool,
    token_ids: Vec<u32>,
    prompt: Vec<u8>,
    sampling_seed: Vec<u8>,
    manifest: Option<&Bound<'_, PyDict>>,
    residuals: Option<&Bound<'_, PyList>>,
) -> PyResult<BatchState> {
    let n_captures = capture_inputs.len();
    if capture_accumulators.len() != n_captures || capture_scales.len() != n_captures {
        return Err(PyValueError::new_err(
            "capture_inputs, capture_accumulators, and capture_scales must have the same length",
        ));
    }

    // Parse captures from Python bytes.
    let mut captures = Vec::with_capacity(n_captures);
    for i in 0..n_captures {
        let x_i8 = extract_i8_vec(&capture_inputs.get_item(i)?)?;
        let acc_i32 = extract_i32_vec(&capture_accumulators.get_item(i)?)?;
        captures.push(verilm_prover::CaptureEntry {
            x_i8,
            acc_i32,
            scale_a: capture_scales[i],
        });
    }

    // Parse residuals if provided.
    let residual_vecs: Option<Vec<Vec<f32>>> = residuals.map(|res_list| {
        let mut vecs = Vec::with_capacity(res_list.len());
        for i in 0..res_list.len() {
            let item = res_list.get_item(i).expect("residual item");
            vecs.push(extract_f32_vec(&item).expect("residual f32 bytes"));
        }
        vecs
    });
    let residual_refs: Option<&[Vec<f32>]> = residual_vecs.as_deref();

    // Build traces in Rust.
    let geom = verilm_prover::ModelGeometry {
        n_layers,
        q_dim,
        kv_dim,
        intermediate_size,
    };
    let (all_layers, token_kvs) = verilm_prover::build_traces_from_captures(
        &captures,
        &geom,
        &fwd_batch_sizes,
        level_c,
        residual_refs,
    );

    // Commit.
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
        token_kvs, // pre-computed: no re-requantize in commit
    );

    Ok(BatchState { inner, commitment })
}

/// Build an audit challenge from a verifier-generated challenge seed.
///
/// Args:
///     challenge_seed: bytes (32) — verifier-generated challenge seed
///     n_tokens: int — number of tokens in the batch
///     n_layers: int — number of layers in the model
///     tier: str — "routine" or "full"
///
/// Returns:
///     dict with token_index (int), layer_indices (list[int]), tier (str)
#[pyfunction]
fn build_audit_challenge(
    py: Python<'_>,
    challenge_seed: Vec<u8>,
    n_tokens: u32,
    n_layers: usize,
    tier: String,
) -> PyResult<Bound<'_, PyDict>> {
    if challenge_seed.len() != 32 {
        return Err(PyValueError::new_err("challenge_seed must be exactly 32 bytes"));
    }
    let mut challenge_seed_arr = [0u8; 32];
    challenge_seed_arr.copy_from_slice(&challenge_seed);

    let tier_enum = match tier.as_str() {
        "routine" => AuditTier::Routine,
        "full" => AuditTier::Full,
        _ => return Err(PyValueError::new_err(format!("invalid tier: {}", tier))),
    };

    let challenge = verilm_verify::build_audit_challenge(
        &challenge_seed_arr,
        n_tokens,
        n_layers,
        tier_enum,
    );

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
///     challenge_seed: hex string (64 chars) — verifier-generated challenge seed.
///     challenge_k: number of tokens to challenge.
///     policy: optional dict with `min_version`, `expected_prompt_hash`,
///         `expected_manifest_hash`.
///
/// Returns:
///     dict with `passed` (bool), `failures` (list of dicts).
#[pyfunction]
#[pyo3(signature = (proof_json, key_json, challenge_seed, challenge_k, policy=None))]
fn verify_batch<'py>(
    py: Python<'py>,
    proof_json: &str,
    key_json: &str,
    challenge_seed: &str,
    challenge_k: u32,
    policy: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let proof: BatchProof = serde_json::from_str(proof_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize BatchProof: {}", e)))?;
    let key: VerifierKey = serde_json::from_str(key_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize VerifierKey: {}", e)))?;
    let challenge_seed_bytes = decode_hex32(challenge_seed, "challenge_seed")?;

    let policy_obj = policy.map(extract_policy).transpose()?.unwrap_or_default();

    let report = verilm_verify::verify_batch_with_policy(
        &key,
        &proof,
        challenge_seed_bytes,
        challenge_k,
        &policy_obj,
    );

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

/// Compute the paper's R_W (weight-chain hash) from safetensors files.
///
/// Args:
///     model_dir: str — path to directory containing .safetensors files.
///
/// Returns:
///     str — hex-encoded 32-byte SHA-256 weight hash.
#[pyfunction]
fn compute_weight_hash(model_dir: String) -> PyResult<String> {
    let path = std::path::Path::new(&model_dir);
    let hash = verilm_keygen::compute_weight_hash(path)
        .map_err(|e| PyValueError::new_err(format!("weight hash computation failed: {}", e)))?;
    Ok(hex::encode(hash))
}

// ===========================================================================
// V4: Minimal retained-state commitment (no _int_mm, no full LayerTrace)
// ===========================================================================

use std::sync::Arc;

/// Cached weight provider loaded once at server startup.
///
/// Wraps `SafetensorsWeightProvider` in an `Arc` so it can be shared
/// across multiple `MinimalBatchStateHandle` instances without reloading.
#[pyclass]
struct WeightProvider {
    inner: Arc<verilm_keygen::SafetensorsWeightProvider>,
}

#[pymethods]
impl WeightProvider {
    #[new]
    fn new(model_dir: String) -> PyResult<Self> {
        let provider = verilm_keygen::SafetensorsWeightProvider::load(
            std::path::Path::new(&model_dir),
        ).map_err(|e| PyValueError::new_err(format!(
            "failed to load weights from {}: {}", model_dir, e
        )))?;
        Ok(WeightProvider { inner: Arc::new(provider) })
    }
}

/// Opaque handle to V4 minimal retained-state commitment.
///
/// Same interface as BatchState for commitment inspection, but audit
/// uses replay from retained state rather than pre-computed full traces.
#[pyclass]
struct MinimalBatchStateHandle {
    inner: verilm_prover::MinimalBatchState,
    commitment: BatchCommitment,
    /// Shared weight provider for shell opening computation at audit time.
    provider: Option<Arc<verilm_keygen::SafetensorsWeightProvider>>,
}

#[pymethods]
impl MinimalBatchStateHandle {
    fn commitment_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.commitment)
            .map_err(|e| PyValueError::new_err(format!("serialization error: {}", e)))
    }

    fn merkle_root_hex(&self) -> String {
        hex::encode(self.commitment.merkle_root)
    }

    fn io_root_hex(&self) -> String {
        hex::encode(self.commitment.io_root)
    }

    fn manifest_hash_hex(&self) -> Option<String> {
        self.commitment.manifest_hash.map(hex::encode)
    }

    fn n_tokens(&self) -> u32 {
        self.commitment.n_tokens
    }

    fn kv_roots_hex(&self) -> Vec<String> {
        Vec::new() // V4: prefix binding via retained Merkle tree
    }

    /// Open a V4 audit response for a challenged token.
    ///
    /// Returns JSON-serialized V4AuditResponse containing the challenged
    /// token's retained state, Merkle/IO proofs, all prefix tokens' retained
    /// states + proofs, and prover-computed shell openings for the challenged
    /// token (so the verifier can check with key-only Freivalds).
    fn audit_v4(&self, token_index: u32, layer_indices: Option<Vec<usize>>) -> PyResult<String> {
        let response = self.build_v4_response(token_index, layer_indices.as_deref())?;
        serde_json::to_string(&response)
            .map_err(|e| PyValueError::new_err(format!("serialization error: {}", e)))
    }

    /// Binary V4 audit: bincode + zstd. Returns bytes.
    fn audit_v4_binary<'py>(&self, py: Python<'py>, token_index: u32, layer_indices: Option<Vec<usize>>) -> PyResult<Bound<'py, PyBytes>> {
        let response = self.build_v4_response(token_index, layer_indices.as_deref())?;
        let data = verilm_core::serialize::serialize_v4_audit(&response);
        Ok(PyBytes::new(py, &data))
    }
}

impl MinimalBatchStateHandle {
    fn build_v4_response(&self, token_index: u32, layer_filter: Option<&[usize]>) -> PyResult<verilm_core::types::V4AuditResponse> {
        if token_index >= self.inner.all_retained.len() as u32 {
            return Err(PyValueError::new_err(format!(
                "token_index {} out of range (n_tokens={})",
                token_index,
                self.inner.all_retained.len()
            )));
        }

        let provider = self.provider.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "V4 audit requires WeightProvider; none was provided at commit time"
            )
        })?;
        let token_id = self.inner.token_ids[token_index as usize] as usize;
        let bridge_data: Option<(Vec<f32>, Option<verilm_core::merkle::MerkleProof>)> =
            if !provider.rmsnorm_attn_weights().is_empty() {
                let embedding_row = provider.load_embedding_row(token_id)
                    .map_err(|e| PyValueError::new_err(format!(
                        "failed to load embedding row for token {}: {}", token_id, e
                    )))?;
                let embedding_proof = provider.embedding_proof(token_id);
                Some((embedding_row, embedding_proof))
            } else {
                None
            };
        let bridge = bridge_data.as_ref().map(|(emb_row, emb_proof)| {
            verilm_core::types::BridgeParams {
                rmsnorm_attn_weights: provider.rmsnorm_attn_weights(),
                rmsnorm_ffn_weights: provider.rmsnorm_ffn_weights(),
                rmsnorm_eps: provider.rmsnorm_eps(),
                initial_residual: emb_row,
                embedding_proof: emb_proof.clone(),
            }
        });
        Ok(verilm_prover::open_v4(
            &self.inner, token_index, provider.as_ref(), provider.config(),
            provider.weight_scales(), bridge.as_ref(), layer_filter,
        ))
    }
}

/// V4 commitment from minimal captures (no accumulators, no _int_mm).
///
/// Args:
///     o_proj_inputs: list[bytes] — per-layer o_proj a_i8 tensors (n_fwd * n_layers).
///     scales: list[float] — 4 scales per layer: [scale_x_attn, scale_a, scale_x_ffn, scale_h].
///     n_layers: int — number of transformer layers.
///     fwd_batch_sizes: list[int] — batch size for each forward pass.
///     token_ids: list[int] — emitted token IDs.
///     prompt: bytes — prompt text.
///     sampling_seed: bytes — 32-byte sampling seed.
///     manifest: dict — deployment manifest (optional).
#[pyfunction]
#[pyo3(signature = (
    o_proj_inputs,
    scales,
    n_layers,
    fwd_batch_sizes,
    token_ids,
    prompt,
    sampling_seed,
    manifest = None,
    weight_provider = None,
))]
#[allow(clippy::too_many_arguments)]
fn commit_minimal_from_captures(
    o_proj_inputs: &Bound<'_, PyList>,
    scales: Vec<f32>,
    n_layers: usize,
    fwd_batch_sizes: Vec<usize>,
    token_ids: Vec<u32>,
    prompt: Vec<u8>,
    sampling_seed: Vec<u8>,
    manifest: Option<&Bound<'_, PyDict>>,
    weight_provider: Option<&WeightProvider>,
) -> PyResult<MinimalBatchStateHandle> {
    let n_entries = o_proj_inputs.len();
    let expected_scales = n_entries * 4;
    if scales.len() != expected_scales {
        return Err(PyValueError::new_err(format!(
            "expected {} scales (4 per layer entry), got {}",
            expected_scales,
            scales.len()
        )));
    }

    let mut captures = Vec::with_capacity(n_entries);
    for i in 0..n_entries {
        let a_i8 = extract_i8_vec(&o_proj_inputs.get_item(i)?)?;
        let base = i * 4;
        captures.push(verilm_prover::MinimalCaptureEntry {
            a_i8,
            scale_x_attn: scales[base],
            scale_a: scales[base + 1],
            scale_x_ffn: scales[base + 2],
            scale_h: scales[base + 3],
        });
    }

    let all_retained = verilm_prover::build_retained_from_captures(
        &captures,
        n_layers,
        &fwd_batch_sizes,
    );

    if sampling_seed.len() != 32 {
        return Err(PyValueError::new_err("sampling_seed must be exactly 32 bytes"));
    }
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&sampling_seed);

    let manifest_obj = manifest.map(extract_manifest).transpose()?;

    let (commitment, inner) = verilm_prover::commit_minimal(
        all_retained,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: &prompt,
            sampling_seed: seed,
            manifest: manifest_obj.as_ref(),
        },
    );

    Ok(MinimalBatchStateHandle {
        inner,
        commitment,
        provider: weight_provider.map(|wp| Arc::clone(&wp.inner)),
    })
}

/// Generate a verifier key from safetensors model weights.
///
/// Args:
///     model_dir: str — path to directory containing .safetensors files.
///     seed: bytes (32) — verifier-secret seed for random vector generation.
///
/// Returns:
///     str — JSON-serialized VerifierKey.
#[pyfunction]
fn generate_key(model_dir: String, seed: Vec<u8>) -> PyResult<String> {
    if seed.len() != 32 {
        return Err(PyValueError::new_err("seed must be exactly 32 bytes"));
    }
    let mut seed_arr = [0u8; 32];
    seed_arr.copy_from_slice(&seed);

    let key = verilm_keygen::generate_key(
        std::path::Path::new(&model_dir),
        seed_arr,
    ).map_err(|e| PyValueError::new_err(format!("keygen failed: {}", e)))?;

    serde_json::to_string(&key)
        .map_err(|e| PyValueError::new_err(format!("serialization error: {}", e)))
}

/// Verify a V4 audit response against a verifier key (key-only Freivalds).
///
/// Args:
///     audit_json: str — JSON-serialized V4AuditResponse.
///     key_json: str — JSON-serialized VerifierKey.
///
/// Returns:
///     dict with `passed` (bool), `checks_run` (int), `checks_passed` (int),
///     `failures` (list of str), `duration_us` (int).
#[pyfunction]
fn verify_v4<'py>(
    py: Python<'py>,
    audit_json: &str,
    key_json: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let response: verilm_core::types::V4AuditResponse = serde_json::from_str(audit_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize V4AuditResponse: {}", e)))?;
    let key: VerifierKey = serde_json::from_str(key_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize VerifierKey: {}", e)))?;

    let report = verilm_verify::verify_v4(&key, &response);

    let result = PyDict::new(py);
    result.set_item("passed", report.verdict == verilm_verify::Verdict::Pass)?;
    result.set_item("checks_run", report.checks_run)?;
    result.set_item("checks_passed", report.checks_passed)?;
    result.set_item("failures", &report.failures)?;
    result.set_item("duration_us", report.duration.as_micros() as u64)?;
    Ok(result)
}

/// Verify a V4 audit response from binary (bincode+zstd) format.
///
/// Args:
///     audit_binary: bytes — binary V4AuditResponse (from audit_v4_binary).
///     key_json: str — JSON-serialized VerifierKey.
#[pyfunction]
fn verify_v4_binary<'py>(
    py: Python<'py>,
    audit_binary: &[u8],
    key_json: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let response = verilm_core::serialize::deserialize_v4_audit(audit_binary)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize V4 binary: {}", e)))?;
    let key: VerifierKey = serde_json::from_str(key_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize VerifierKey: {}", e)))?;

    let report = verilm_verify::verify_v4(&key, &response);

    let result = PyDict::new(py);
    result.set_item("passed", report.verdict == verilm_verify::Verdict::Pass)?;
    result.set_item("checks_run", report.checks_run)?;
    result.set_item("checks_passed", report.checks_passed)?;
    result.set_item("failures", &report.failures)?;
    result.set_item("duration_us", report.duration.as_micros() as u64)?;
    Ok(result)
}

#[pymodule]
fn verilm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(commit, m)?)?;
    m.add_function(wrap_pyfunction!(commit_from_captures, m)?)?;
    m.add_function(wrap_pyfunction!(commit_minimal_from_captures, m)?)?;
    m.add_function(wrap_pyfunction!(build_audit_challenge, m)?)?;
    m.add_function(wrap_pyfunction!(verify_batch, m)?)?;
    m.add_function(wrap_pyfunction!(verify_single, m)?)?;
    m.add_function(wrap_pyfunction!(compute_weight_hash, m)?)?;
    m.add_function(wrap_pyfunction!(generate_key, m)?)?;
    m.add_function(wrap_pyfunction!(verify_v4, m)?)?;
    m.add_function(wrap_pyfunction!(verify_v4_binary, m)?)?;
    m.add_class::<WeightProvider>()?;
    m.add_class::<BatchState>()?;
    m.add_class::<MinimalBatchStateHandle>()?;
    Ok(())
}
