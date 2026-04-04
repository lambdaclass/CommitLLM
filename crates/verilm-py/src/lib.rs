//! PyO3 bridge: expose the Rust commitment engine to Python.
//!
//! The Python `verilm` package captures data, then calls these functions
//! to build V4 retained-state commitments and generate proofs in-process.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict, PyList};

use verilm_core::types::{
    AuditTier, BatchCommitment, DeploymentManifest, InputSpec, V4AuditResponse, VerifierKey,
};
use verilm_prover::FullBindingParams;

/// Wraps a Python callable as a `PromptTokenizer` for verify_v4_full.
///
/// The Python callable signature is:
///   `fn(prompt: bytes, input_spec: dict) -> list[int]`
///
/// The input_spec dict contains: tokenizer_hash (hex), system_prompt_hash (hex|None),
/// chat_template_hash (hex|None), bos_eos_policy (str|None), truncation_policy (str|None),
/// special_token_policy (str|None).
struct PyCallableTokenizer<'py> {
    callback: Bound<'py, PyAny>,
}

impl verilm_verify::PromptTokenizer for PyCallableTokenizer<'_> {
    fn tokenize(&self, prompt: &[u8], input_spec: &InputSpec) -> Result<Vec<u32>, String> {
        let py = self.callback.py();
        let prompt_bytes = PyBytes::new(py, prompt);
        let spec_dict = PyDict::new(py);
        spec_dict
            .set_item("tokenizer_hash", hex::encode(input_spec.tokenizer_hash))
            .map_err(|e| e.to_string())?;
        spec_dict
            .set_item(
                "system_prompt_hash",
                input_spec.system_prompt_hash.map(hex::encode),
            )
            .map_err(|e| e.to_string())?;
        spec_dict
            .set_item(
                "chat_template_hash",
                input_spec.chat_template_hash.map(hex::encode),
            )
            .map_err(|e| e.to_string())?;
        spec_dict
            .set_item("bos_eos_policy", &input_spec.bos_eos_policy)
            .map_err(|e| e.to_string())?;
        spec_dict
            .set_item("truncation_policy", &input_spec.truncation_policy)
            .map_err(|e| e.to_string())?;
        spec_dict
            .set_item("special_token_policy", &input_spec.special_token_policy)
            .map_err(|e| e.to_string())?;

        let result = self
            .callback
            .call1((prompt_bytes, spec_dict))
            .map_err(|e| format!("Python tokenizer callback failed: {}", e))?;
        result
            .extract::<Vec<u32>>()
            .map_err(|e| format!("tokenizer callback did not return list[int]: {}", e))
    }
}

/// Wraps a Python callable as a `Detokenizer` for verify_v4_full.
///
/// The Python callable signature is:
///   `fn(token_ids: list[int], policy: str|None) -> str`
struct PyCallableDetokenizer<'py> {
    callback: Bound<'py, PyAny>,
}

impl verilm_verify::Detokenizer for PyCallableDetokenizer<'_> {
    fn decode(&self, token_ids: &[u32], policy: Option<&str>) -> Result<String, String> {
        let py = self.callback.py();
        let ids_list = PyList::new(py, token_ids).map_err(|e| e.to_string())?;
        let policy_obj: Bound<'_, PyAny> = match policy {
            Some(p) => p.into_pyobject(py).map_err(|e| e.to_string())?.into_any(),
            None => py.None().into_bound(py),
        };
        let result = self
            .callback
            .call1((ids_list, policy_obj))
            .map_err(|e| format!("Python detokenizer callback failed: {}", e))?;
        result
            .extract::<String>()
            .map_err(|e| format!("detokenizer callback did not return str: {}", e))
    }
}

/// Read raw bytes from a Python buffer protocol object (numpy arrays, CPU tensors).
/// Returns None if the object doesn't support the buffer protocol.
/// Avoids the Python-side `.tobytes()` allocation + copy.
fn try_buffer_as_bytes(obj: &Bound<'_, PyAny>) -> Option<Vec<u8>> {
    let mut view: pyo3::ffi::Py_buffer = unsafe { std::mem::zeroed() };
    if unsafe { pyo3::ffi::PyObject_GetBuffer(obj.as_ptr(), &mut view, pyo3::ffi::PyBUF_SIMPLE) }
        != 0
    {
        unsafe { pyo3::ffi::PyErr_Clear() };
        return None;
    }
    let len = view.len as usize;
    let mut dst = Vec::with_capacity(len);
    unsafe {
        std::ptr::copy_nonoverlapping(view.buf as *const u8, dst.as_mut_ptr(), len);
        dst.set_len(len);
        pyo3::ffi::PyBuffer_Release(&mut view);
    }
    Some(dst)
}

/// Reinterpret raw bytes as i8 (identical layout, zero-cost transmute).
fn raw_to_i8(raw: Vec<u8>) -> Vec<i8> {
    let len = raw.len();
    let cap = raw.capacity();
    let ptr = std::mem::ManuallyDrop::new(raw).as_mut_ptr();
    // SAFETY: u8 and i8 have identical size, alignment, and valid-value ranges.
    unsafe { Vec::from_raw_parts(ptr as *mut i8, len, cap) }
}

/// Reinterpret raw bytes as f32 (native-endian bulk copy).
fn raw_to_f32(src: &[u8]) -> PyResult<Vec<f32>> {
    if src.len() % 4 != 0 {
        return Err(PyValueError::new_err(format!(
            "f32 bytes length {} not a multiple of 4",
            src.len()
        )));
    }
    let n = src.len() / 4;
    let mut dst = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr() as *const f32, dst.as_mut_ptr(), n);
        dst.set_len(n);
    }
    Ok(dst)
}

/// Extract a Vec<i8> from bytes, buffer protocol, or Python list.
fn extract_i8_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i8>> {
    if let Ok(b) = obj.cast::<PyBytes>() {
        let src = b.as_bytes();
        let mut dst = Vec::with_capacity(src.len());
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr() as *const i8, dst.as_mut_ptr(), src.len());
            dst.set_len(src.len());
        }
        return Ok(dst);
    }
    if let Some(raw) = try_buffer_as_bytes(obj) {
        return Ok(raw_to_i8(raw));
    }
    if let Ok(v) = obj.extract::<Vec<i8>>() {
        return Ok(v);
    }
    Err(PyValueError::new_err(
        "expected bytes, buffer, or list of i8",
    ))
}

/// Extract a Vec<f32> from bytes, buffer protocol, or Python list.
fn extract_f32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(b) = obj.cast::<PyBytes>() {
        return raw_to_f32(b.as_bytes());
    }
    if let Some(raw) = try_buffer_as_bytes(obj) {
        return raw_to_f32(&raw);
    }
    obj.extract::<Vec<f32>>()
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
        return Err(PyValueError::new_err(
            "tokenizer_hash must be 32 bytes (64 hex chars)",
        ));
    }
    let mut tokenizer_hash = [0u8; 32];
    tokenizer_hash.copy_from_slice(&bytes);

    Ok(DeploymentManifest {
        tokenizer_hash,
        temperature: d
            .get_item("temperature")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0.0),
        top_k: d
            .get_item("top_k")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0),
        top_p: d
            .get_item("top_p")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(1.0),
        eos_policy: d
            .get_item("eos_policy")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_else(|| "stop".to_string()),
        weight_hash: d
            .get_item("weight_hash")?
            .map(|v| {
                let hex_str: String = v.extract()?;
                decode_hex32(&hex_str, "weight_hash")
            })
            .transpose()?,
        quant_hash: d
            .get_item("quant_hash")?
            .map(|v| {
                let hex_str: String = v.extract()?;
                decode_hex32(&hex_str, "quant_hash")
            })
            .transpose()?,
        system_prompt_hash: d
            .get_item("system_prompt_hash")?
            .map(|v| {
                let hex_str: String = v.extract()?;
                decode_hex32(&hex_str, "system_prompt_hash")
            })
            .transpose()?,
        repetition_penalty: d
            .get_item("repetition_penalty")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(1.0),
        frequency_penalty: d
            .get_item("frequency_penalty")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0.0),
        presence_penalty: d
            .get_item("presence_penalty")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0.0),
        logit_bias: {
            let lb: Vec<(u32, f32)> = d
                .get_item("logit_bias")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_default();
            let mut sorted = lb;
            sorted.sort_by_key(|&(tid, _)| tid);
            sorted
        },
        bad_word_ids: {
            let bw: Vec<u32> = d
                .get_item("bad_word_ids")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_default();
            let mut sorted = bw;
            sorted.sort();
            sorted.dedup();
            sorted
        },
        guided_decoding: d
            .get_item("guided_decoding")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default(),
        stop_sequences: d
            .get_item("stop_sequences")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default(),
        max_tokens: d
            .get_item("max_tokens")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0),
        chat_template_hash: d
            .get_item("chat_template_hash")?
            .and_then(|v| {
                // Python may pass None for absent hash.
                if v.is_none() {
                    return None;
                }
                Some(v)
            })
            .map(|v| {
                let hex_str: String = v.extract()?;
                decode_hex32(&hex_str, "chat_template_hash")
            })
            .transpose()?,
        rope_config_hash: d
            .get_item("rope_config_hash")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| {
                let hex_str: String = v.extract()?;
                decode_hex32(&hex_str, "rope_config_hash")
            })
            .transpose()?,
        rmsnorm_eps: d
            .get_item("rmsnorm_eps")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract::<f64>())
            .transpose()?,
        sampler_version: d
            .get_item("sampler_version")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract::<String>())
            .transpose()?,
        bos_eos_policy: d
            .get_item("bos_eos_policy")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract::<String>())
            .transpose()?,
        truncation_policy: d
            .get_item("truncation_policy")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract::<String>())
            .transpose()?,
        special_token_policy: d
            .get_item("special_token_policy")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract::<String>())
            .transpose()?,
        adapter_hash: d
            .get_item("adapter_hash")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| {
                let hex_str: String = v.extract()?;
                decode_hex32(&hex_str, "adapter_hash")
            })
            .transpose()?,
        min_tokens: d
            .get_item("min_tokens")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0),
        ignore_eos: d
            .get_item("ignore_eos")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(false),
        detokenization_policy: d
            .get_item("detokenization_policy")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract::<String>())
            .transpose()?,
        eos_token_id: d
            .get_item("eos_token_id")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract())
            .transpose()?,
        n_layers: d
            .get_item("n_layers")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract())
            .transpose()?,
        hidden_dim: d
            .get_item("hidden_dim")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract())
            .transpose()?,
        vocab_size: d
            .get_item("vocab_size")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract())
            .transpose()?,
        embedding_merkle_root: d
            .get_item("embedding_merkle_root")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| {
                let hex_str: String = v.extract()?;
                decode_hex32(&hex_str, "embedding_merkle_root")
            })
            .transpose()?,
        padding_policy: d
            .get_item("padding_policy")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract::<String>())
            .transpose()?,
        decode_mode: d
            .get_item("decode_mode")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract::<String>())
            .transpose()?,
        quant_family: d
            .get_item("quant_family")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract::<String>())
            .transpose()?,
        scale_derivation: d
            .get_item("scale_derivation")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract::<String>())
            .transpose()?,
        quant_block_size: d
            .get_item("quant_block_size")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract())
            .transpose()?,
        attn_backend: d
            .get_item("attn_backend")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract::<String>())
            .transpose()?,
        attn_dtype: d
            .get_item("attn_dtype")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract::<String>())
            .transpose()?,
        kv_dim: d
            .get_item("kv_dim")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract())
            .transpose()?,
        ffn_dim: d
            .get_item("ffn_dim")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract())
            .transpose()?,
        d_head: d
            .get_item("d_head")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract())
            .transpose()?,
        n_q_heads: d
            .get_item("n_q_heads")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract())
            .transpose()?,
        n_kv_heads: d
            .get_item("n_kv_heads")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract())
            .transpose()?,
        rope_theta: d
            .get_item("rope_theta")?
            .and_then(|v| if v.is_none() { None } else { Some(v) })
            .map(|v| v.extract::<f64>())
            .transpose()?,
    })
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
        return Err(PyValueError::new_err(
            "challenge_seed must be exactly 32 bytes",
        ));
    }
    let mut challenge_seed_arr = [0u8; 32];
    challenge_seed_arr.copy_from_slice(&challenge_seed);

    let tier_enum = match tier.as_str() {
        "routine" => AuditTier::Routine,
        "full" => AuditTier::Full,
        _ => return Err(PyValueError::new_err(format!("invalid tier: {}", tier))),
    };

    let challenge =
        verilm_verify::build_audit_challenge(&challenge_seed_arr, n_tokens, n_layers, tier_enum);

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
        let provider =
            verilm_keygen::SafetensorsWeightProvider::load(std::path::Path::new(&model_dir))
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "failed to load weights from {}: {}",
                        model_dir, e
                    ))
                })?;
        Ok(WeightProvider {
            inner: Arc::new(provider),
        })
    }

    /// Return the R_W weight-chain hash as a hex string.
    fn weight_hash_hex(&self) -> String {
        hex::encode(self.inner.weight_hash())
    }

    /// Quantization family matching keygen vocabulary (e.g. "W8A8", "INT8").
    fn quant_family(&self) -> Option<String> {
        self.inner.quant_family()
    }

    /// Scale derivation matching keygen vocabulary (e.g. "per_channel_absmax").
    fn scale_derivation(&self) -> Option<String> {
        self.inner.scale_derivation()
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
    /// Whether to include rich prefix (embedding rows + proofs) in audit openings.
    rich_prefix: bool,
    /// Whether to include full retained state + shell openings for prefix tokens (deep audit).
    deep_prefix: bool,
    /// When true, shell opening QKV uses GPU-captured x_attn instead of bridge-derived.
    /// For corridor measurement only — makes Q match the same boundary as committed K/V.
    use_captured_x_attn: bool,
}

#[pymethods]
impl MinimalBatchStateHandle {
    #[setter]
    fn set_deep_prefix(&mut self, val: bool) {
        self.deep_prefix = val;
    }

    #[getter]
    fn get_deep_prefix(&self) -> bool {
        self.deep_prefix
    }

    #[setter]
    fn set_use_captured_x_attn(&mut self, val: bool) {
        self.use_captured_x_attn = val;
    }

    #[getter]
    fn get_use_captured_x_attn(&self) -> bool {
        self.use_captured_x_attn
    }

    /// Check if captured x_attn data is available for precision corridor.
    fn has_captured_x_attn(&self) -> bool {
        self.inner.captured_x_attn.is_some()
    }

    /// Clear captured x_attn data (switches audit to verifier-replay mode).
    fn clear_captured_x_attn(&mut self) {
        self.inner.captured_x_attn = None;
    }

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
        self.commitment.kv_roots.iter().map(hex::encode).collect()
    }

    /// Open a V4 audit response for a challenged token.
    ///
    /// Returns JSON-serialized V4AuditResponse containing the challenged
    /// token's retained state, Merkle/IO proofs, all prefix tokens' retained
    /// states + proofs, and prover-computed shell openings for the challenged
    /// token (so the verifier can check with key-only Freivalds).
    #[pyo3(signature = (token_index, layer_indices=None, output_text=None))]
    fn audit_v4(
        &self,
        token_index: u32,
        layer_indices: Option<Vec<usize>>,
        output_text: Option<String>,
    ) -> PyResult<String> {
        let mut response = self.build_v4_response(token_index, layer_indices.as_deref())?;
        response.output_text = output_text;
        serde_json::to_string(&response)
            .map_err(|e| PyValueError::new_err(format!("serialization error: {}", e)))
    }

    /// Binary V4 audit: bincode + zstd. Returns bytes.
    #[pyo3(signature = (token_index, layer_indices=None, output_text=None))]
    fn audit_v4_binary<'py>(
        &self,
        py: Python<'py>,
        token_index: u32,
        layer_indices: Option<Vec<usize>>,
        output_text: Option<String>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut response = self.build_v4_response(token_index, layer_indices.as_deref())?;
        response.output_text = output_text;
        let data = verilm_core::serialize::serialize_v4_audit(&response);
        Ok(PyBytes::new(py, &data))
    }
}

impl MinimalBatchStateHandle {
    fn build_v4_response(
        &self,
        token_index: u32,
        layer_filter: Option<&[usize]>,
    ) -> PyResult<verilm_core::types::V4AuditResponse> {
        if token_index >= self.inner.all_retained.len() as u32 {
            return Err(PyValueError::new_err(format!(
                "token_index {} out of range (n_tokens={})",
                token_index,
                self.inner.all_retained.len()
            )));
        }

        let provider = self.provider.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "V4 audit requires WeightProvider; none was provided at commit time",
            )
        })?;
        let token_id = self.inner.token_ids[token_index as usize] as usize;
        let bridge_data: Option<(Vec<f32>, Option<verilm_core::merkle::MerkleProof>)> =
            if !provider.rmsnorm_attn_weights().is_empty() {
                let embedding_row = provider.load_embedding_row(token_id).map_err(|e| {
                    PyValueError::new_err(format!(
                        "failed to load embedding row for token {}: {}",
                        token_id, e
                    ))
                })?;
                let embedding_proof = provider.embedding_proof(token_id);
                Some((embedding_row, embedding_proof))
            } else {
                None
            };
        let bridge =
            bridge_data
                .as_ref()
                .map(|(emb_row, emb_proof)| verilm_core::types::BridgeParams {
                    rmsnorm_attn_weights: provider.rmsnorm_attn_weights(),
                    rmsnorm_ffn_weights: provider.rmsnorm_ffn_weights(),
                    rmsnorm_eps: provider.rmsnorm_eps(),
                    initial_residual: emb_row,
                    embedding_proof: emb_proof.clone(),
                });
        let tail = provider.tail_params();
        let emb_lookup: Option<&dyn verilm_core::types::EmbeddingLookup> =
            if self.rich_prefix || self.deep_prefix {
                Some(provider.as_ref())
            } else {
                None
            };
        Ok(verilm_prover::open_v4(
            &self.inner,
            token_index,
            provider.as_ref(),
            provider.config(),
            provider.weight_scales(),
            provider.per_channel_weight_scales(),
            bridge.as_ref(),
            tail.as_ref(),
            layer_filter,
            emb_lookup,
            self.deep_prefix,
            self.use_captured_x_attn,
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
    final_residuals = None,
    n_prompt_tokens = None,
    x_attn_inputs = None,
))]
#[allow(clippy::too_many_arguments)]
fn commit_minimal_from_captures(
    o_proj_inputs: &Bound<'_, PyList>,
    scales: &Bound<'_, PyAny>,
    n_layers: usize,
    fwd_batch_sizes: Vec<usize>,
    token_ids: Vec<u32>,
    prompt: Vec<u8>,
    sampling_seed: Vec<u8>,
    manifest: Option<&Bound<'_, PyDict>>,
    weight_provider: Option<&WeightProvider>,
    final_residuals: Option<&Bound<'_, PyList>>,
    n_prompt_tokens: Option<u32>,
    x_attn_inputs: Option<&Bound<'_, PyList>>,
) -> PyResult<MinimalBatchStateHandle> {
    let scales = extract_f32_vec(scales)?;
    let n_entries = o_proj_inputs.len();

    // Scales are per-row: for a prefill with batch_size=B, each of the 4
    // projections contributes B scale values. Flat layout in call order:
    //   for each fwd pass:
    //     for each layer:
    //       [B values for qkv_proj, B for o_proj, B for gate_up, B for down]
    let expected_scales: usize = fwd_batch_sizes.iter().map(|&bs| bs * 4 * n_layers).sum();
    if scales.len() != expected_scales {
        return Err(PyValueError::new_err(format!(
            "expected {} scales (4 × batch_size per layer entry), got {}",
            expected_scales,
            scales.len()
        )));
    }

    let mut captures = Vec::with_capacity(n_entries);
    let mut cursor = 0usize;
    let mut entry_idx = 0usize;
    for &batch_size in &fwd_batch_sizes {
        for _l in 0..n_layers {
            let a_i8 = extract_i8_vec(&o_proj_inputs.get_item(entry_idx)?)?;
            let x_attn_i8 = if let Some(ref xa_list) = x_attn_inputs {
                if entry_idx < xa_list.len() {
                    Some(extract_i8_vec(&xa_list.get_item(entry_idx)?)?)
                } else {
                    None
                }
            } else {
                None
            };
            let s_qkv = scales[cursor..cursor + batch_size].to_vec();
            cursor += batch_size;
            let s_o = scales[cursor..cursor + batch_size].to_vec();
            cursor += batch_size;
            let s_gate = scales[cursor..cursor + batch_size].to_vec();
            cursor += batch_size;
            let s_down = scales[cursor..cursor + batch_size].to_vec();
            cursor += batch_size;
            captures.push(verilm_prover::MinimalCaptureEntry {
                a_i8,
                x_attn_i8,
                scale_x_attn: s_qkv,
                scale_a: s_o,
                scale_x_ffn: s_gate,
                scale_h: s_down,
            });
            entry_idx += 1;
        }
    }

    let (all_retained, captured_scales, captured_x_attn) =
        verilm_prover::build_retained_from_captures(&captures, n_layers, &fwd_batch_sizes);

    if sampling_seed.len() != 32 {
        return Err(PyValueError::new_err(
            "sampling_seed must be exactly 32 bytes",
        ));
    }
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&sampling_seed);

    let manifest_obj = manifest.map(extract_manifest).transpose()?;

    // Extract per-token final residuals (pre-final-norm, f32) if provided.
    let final_res = if let Some(fr_list) = final_residuals {
        let mut vecs = Vec::with_capacity(fr_list.len());
        for i in 0..fr_list.len() {
            vecs.push(extract_f32_vec(&fr_list.get_item(i)?)?);
        }
        Some(vecs)
    } else {
        None
    };

    // Compute KV transcript from captured x_attn + weights when both are available.
    // This produces the committed KV entries that the verifier will replay against.
    let kv_transcripts = if captured_x_attn.is_some() && weight_provider.is_some() {
        let wp = weight_provider.unwrap();
        let cfg = wp.inner.config();
        let x_attn_ref = captured_x_attn.as_ref().unwrap();
        Some(verilm_prover::compute_kv_transcript(
            x_attn_ref,
            wp.inner.as_ref(),
            cfg,
            &captured_scales,
            wp.inner.weight_scales(),
            wp.inner.per_channel_weight_scales(),
            wp.inner.qkv_biases(),
        ))
    } else {
        None
    };

    let (commitment, inner) = verilm_prover::commit_minimal(
        all_retained,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: &prompt,
            sampling_seed: seed,
            manifest: manifest_obj.as_ref(),
            n_prompt_tokens,
        },
        final_res,
        captured_scales,
        captured_x_attn,
        kv_transcripts,
    );

    Ok(MinimalBatchStateHandle {
        inner,
        commitment,
        provider: weight_provider.map(|wp| Arc::clone(&wp.inner)),
        rich_prefix: false,
        deep_prefix: false,
        use_captured_x_attn: false,
    })
}

// ── Packed commit path (fewer allocations) ────────────────────────

/// Python handle wrapping PackedBatchState.
///
/// Drop-in replacement for MinimalBatchStateHandle — same Python API,
/// but commit was done from packed buffers without per-token Vecs.
#[pyclass]
struct PackedBatchStateHandle {
    inner: verilm_prover::PackedBatchState,
    commitment: BatchCommitment,
    provider: Option<Arc<verilm_keygen::SafetensorsWeightProvider>>,
    rich_prefix: bool,
    deep_prefix: bool,
}

#[pymethods]
impl PackedBatchStateHandle {
    #[setter]
    fn set_deep_prefix(&mut self, val: bool) {
        self.deep_prefix = val;
    }

    #[getter]
    fn get_deep_prefix(&self) -> bool {
        self.deep_prefix
    }

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
        self.commitment.kv_roots.iter().map(hex::encode).collect()
    }

    #[pyo3(signature = (token_index, layer_indices=None, output_text=None))]
    fn audit_v4(
        &self,
        token_index: u32,
        layer_indices: Option<Vec<usize>>,
        output_text: Option<String>,
    ) -> PyResult<String> {
        let mut response = self.build_v4_response(token_index, layer_indices.as_deref())?;
        response.output_text = output_text;
        serde_json::to_string(&response)
            .map_err(|e| PyValueError::new_err(format!("serialization error: {}", e)))
    }

    #[pyo3(signature = (token_index, layer_indices=None, output_text=None))]
    fn audit_v4_binary<'py>(
        &self,
        py: Python<'py>,
        token_index: u32,
        layer_indices: Option<Vec<usize>>,
        output_text: Option<String>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut response = self.build_v4_response(token_index, layer_indices.as_deref())?;
        response.output_text = output_text;
        let data = verilm_core::serialize::serialize_v4_audit(&response);
        Ok(PyBytes::new(py, &data))
    }
}

impl PackedBatchStateHandle {
    fn build_v4_response(
        &self,
        token_index: u32,
        layer_filter: Option<&[usize]>,
    ) -> PyResult<verilm_core::types::V4AuditResponse> {
        if token_index >= self.inner.n_tokens() as u32 {
            return Err(PyValueError::new_err(format!(
                "token_index {} out of range (n_tokens={})",
                token_index,
                self.inner.n_tokens()
            )));
        }

        let provider = self.provider.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "V4 audit requires WeightProvider; none was provided at commit time",
            )
        })?;
        let token_id = self.inner.token_ids[token_index as usize] as usize;
        let bridge_data: Option<(Vec<f32>, Option<verilm_core::merkle::MerkleProof>)> =
            if !provider.rmsnorm_attn_weights().is_empty() {
                let embedding_row = provider.load_embedding_row(token_id).map_err(|e| {
                    PyValueError::new_err(format!(
                        "failed to load embedding row for token {}: {}",
                        token_id, e
                    ))
                })?;
                let embedding_proof = provider.embedding_proof(token_id);
                Some((embedding_row, embedding_proof))
            } else {
                None
            };
        let bridge =
            bridge_data
                .as_ref()
                .map(|(emb_row, emb_proof)| verilm_core::types::BridgeParams {
                    rmsnorm_attn_weights: provider.rmsnorm_attn_weights(),
                    rmsnorm_ffn_weights: provider.rmsnorm_ffn_weights(),
                    rmsnorm_eps: provider.rmsnorm_eps(),
                    initial_residual: emb_row,
                    embedding_proof: emb_proof.clone(),
                });
        let tail = provider.tail_params();
        let emb_lookup: Option<&dyn verilm_core::types::EmbeddingLookup> =
            if self.rich_prefix || self.deep_prefix {
                Some(provider.as_ref())
            } else {
                None
            };
        Ok(verilm_prover::open_v4_packed(
            &self.inner,
            token_index,
            provider.as_ref(),
            provider.config(),
            provider.weight_scales(),
            provider.per_channel_weight_scales(),
            bridge.as_ref(),
            tail.as_ref(),
            layer_filter,
            emb_lookup,
            self.deep_prefix,
        ))
    }
}

/// V4 commitment from packed contiguous buffers (fewer allocations).
///
/// Accepts capture data as contiguous buffers via the buffer protocol
/// (bytearray, numpy arrays, memoryview), avoiding per-entry Python→Rust
/// crossing overhead and intermediate Vec allocations.
///
/// Args:
///     packed_a: buffer — contiguous i8 bytes, fwd-major × layer-major × batch-row-major.
///     packed_scales: buffer — f32 scales via buffer protocol (numpy array). 4 per (fwd, layer): [scale_x_attn, scale_a, scale_x_ffn, scale_h].
///     n_layers: int — number of transformer layers.
///     hidden_dim: int — hidden dimension (a_i8 row length).
///     fwd_batch_sizes: list[int] — batch size for each forward pass.
///     token_ids: list[int] — emitted token IDs.
///     prompt: bytes — prompt text.
///     sampling_seed: bytes — 32-byte sampling seed.
///     manifest: dict — deployment manifest (optional).
///     weight_provider: WeightProvider — for audit-time shell computation (optional).
///     packed_final_res: buffer — contiguous f32 bytes for final residuals (optional).
///     final_res_dim: int — per-token final residual dimension.
#[pyfunction]
#[pyo3(signature = (
    packed_a,
    packed_scales,
    n_layers,
    hidden_dim,
    fwd_batch_sizes,
    token_ids,
    prompt,
    sampling_seed,
    manifest = None,
    weight_provider = None,
    packed_final_res = None,
    final_res_dim = 0,
    n_prompt_tokens = None,
))]
#[allow(clippy::too_many_arguments)]
fn commit_minimal_packed(
    packed_a: &Bound<'_, PyAny>,
    packed_scales: &Bound<'_, PyAny>,
    n_layers: usize,
    hidden_dim: usize,
    fwd_batch_sizes: Vec<usize>,
    token_ids: Vec<u32>,
    prompt: Vec<u8>,
    sampling_seed: Vec<u8>,
    manifest: Option<&Bound<'_, PyDict>>,
    weight_provider: Option<&WeightProvider>,
    packed_final_res: Option<&Bound<'_, PyAny>>,
    final_res_dim: usize,
    n_prompt_tokens: Option<u32>,
) -> PyResult<PackedBatchStateHandle> {
    // Extract packed_a via buffer protocol (one copy into Rust Vec<u8>).
    let a_bytes: Vec<u8> = if let Ok(b) = packed_a.cast::<PyBytes>() {
        b.as_bytes().to_vec()
    } else if let Some(raw) = try_buffer_as_bytes(packed_a) {
        raw
    } else {
        return Err(PyValueError::new_err(
            "packed_a must support buffer protocol (bytes, bytearray, numpy, memoryview)",
        ));
    };

    // Extract packed final residuals similarly.
    let fr_bytes: Option<Vec<u8>> = packed_final_res
        .map(|obj| {
            if let Ok(b) = obj.cast::<PyBytes>() {
                Ok(b.as_bytes().to_vec())
            } else if let Some(raw) = try_buffer_as_bytes(obj) {
                Ok(raw)
            } else {
                Err(PyValueError::new_err(
                    "packed_final_res must support buffer protocol",
                ))
            }
        })
        .transpose()?;

    // Extract scales via buffer protocol (numpy array → one bulk memcpy).
    // Drain output is in call order: for each (fwd, layer), 4 groups of
    // batch_size values. Rearrange to per-token layout: (token, layer, proj).
    let drain_scales = extract_f32_vec(packed_scales)?;
    let n_tokens: usize = fwd_batch_sizes.iter().sum();
    let mut packed_scales = vec![0f32; n_tokens * n_layers * 4];
    {
        let mut cursor = 0usize;
        let mut token_offset = 0usize;
        for &batch_size in &fwd_batch_sizes {
            for l in 0..n_layers {
                // 4 projections, each with batch_size values
                for proj in 0..4usize {
                    for b in 0..batch_size {
                        let t = token_offset + b;
                        packed_scales[(t * n_layers + l) * 4 + proj] = drain_scales[cursor];
                        cursor += 1;
                    }
                }
            }
            token_offset += batch_size;
        }
        assert_eq!(
            cursor,
            drain_scales.len(),
            "scale rearrangement consumed wrong number of values"
        );
    }

    if sampling_seed.len() != 32 {
        return Err(PyValueError::new_err(
            "sampling_seed must be exactly 32 bytes",
        ));
    }
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&sampling_seed);

    let manifest_obj = manifest.map(extract_manifest).transpose()?;

    let (commitment, inner) = verilm_prover::commit_minimal_packed(
        a_bytes,
        packed_scales,
        n_layers,
        hidden_dim,
        fwd_batch_sizes,
        &FullBindingParams {
            token_ids: &token_ids,
            prompt: &prompt,
            sampling_seed: seed,
            manifest: manifest_obj.as_ref(),
            n_prompt_tokens,
        },
        fr_bytes,
        final_res_dim,
    );

    Ok(PackedBatchStateHandle {
        inner,
        commitment,
        provider: weight_provider.map(|wp| Arc::clone(&wp.inner)),
        rich_prefix: false,
        deep_prefix: false,
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

    let key = verilm_keygen::generate_key(std::path::Path::new(&model_dir), seed_arr)
        .map_err(|e| PyValueError::new_err(format!("keygen failed: {}", e)))?;

    serde_json::to_string(&key)
        .map_err(|e| PyValueError::new_err(format!("serialization error: {}", e)))
}

/// Verify a V4 audit response from JSON (debug/convenience path).
///
/// The canonical production path is [`verify_v4_binary`] which consumes
/// the normative bincode+zstd wire format. This JSON entry point exists
/// for debugging and development.
///
/// Internally delegates to the canonical verifier (`canonical::verify_response`).
///
/// Args:
///     audit_json: str — JSON-serialized V4AuditResponse.
///     key_json: str — JSON-serialized VerifierKey.
///     expected_prompt_token_ids: Optional[list[int]] — **ignored** (accepted for API
///         compatibility only). The canonical verifier requires a tokenizer callback
///         for tokenization checks; caller-supplied IDs are not used.
///     tokenizer_fn: Optional[Callable[[bytes, dict], list[int]]] — tokenizer callback.
///         When provided, the verifier calls `tokenizer_fn(prompt_bytes, input_spec_dict)`
///         to reconstruct prompt token IDs from raw bytes + the committed InputSpec.
///     detokenizer_fn: Optional[Callable[[list[int], str|None], str]] — detokenizer callback.
///
/// Returns:
///     dict with `passed` (bool), `checks_run` (int), `checks_passed` (int),
///     `failures` (list of str), `duration_us` (int).
#[pyfunction]
#[pyo3(signature = (audit_json, key_json, expected_prompt_token_ids=None, tokenizer_fn=None, detokenizer_fn=None))]
fn verify_v4<'py>(
    py: Python<'py>,
    audit_json: &str,
    key_json: &str,
    expected_prompt_token_ids: Option<Vec<u32>>,
    tokenizer_fn: Option<Bound<'py, PyAny>>,
    detokenizer_fn: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let response: verilm_core::types::V4AuditResponse =
        serde_json::from_str(audit_json).map_err(|e| {
            PyValueError::new_err(format!("failed to deserialize V4AuditResponse: {}", e))
        })?;
    let key: VerifierKey = serde_json::from_str(key_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize VerifierKey: {}", e)))?;

    let report = run_verify(
        &key,
        &response,
        expected_prompt_token_ids.as_deref(),
        tokenizer_fn,
        detokenizer_fn,
    );

    let result = PyDict::new(py);
    result.set_item("passed", report.verdict == verilm_verify::Verdict::Pass)?;
    result.set_item("checks_run", report.checks_run)?;
    result.set_item("checks_passed", report.checks_passed)?;
    let failure_msgs: Vec<&str> = report.failures.iter().map(|f| f.message.as_str()).collect();
    result.set_item("failures", &failure_msgs)?;
    result.set_item(
        "classified_failures",
        classified_failures_to_py(py, &report.failures)?,
    )?;
    result.set_item("coverage", coverage_to_py(py, &report.coverage))?;
    result.set_item("duration_us", report.duration.as_micros() as u64)?;
    Ok(result)
}

/// Verify a V4 audit response from binary (bincode+zstd) format — canonical production path.
///
/// Internally delegates to the canonical verifier (`canonical::verify_response`).
///
/// Args:
///     audit_binary: bytes — binary V4AuditResponse (from audit_v4_binary).
///     key_json: str — JSON-serialized VerifierKey.
///     expected_prompt_token_ids: Optional[list[int]] — **ignored** (accepted for API
///         compatibility only). The canonical verifier requires a tokenizer callback
///         for tokenization checks; caller-supplied IDs are not used.
///     tokenizer_fn: Optional[Callable[[bytes, dict], list[int]]] — tokenizer callback.
///     detokenizer_fn: Optional[Callable[[list[int], str|None], str]] — detokenizer callback.
#[pyfunction]
#[pyo3(signature = (audit_binary, key_json, expected_prompt_token_ids=None, tokenizer_fn=None, detokenizer_fn=None))]
fn verify_v4_binary<'py>(
    py: Python<'py>,
    audit_binary: &[u8],
    key_json: &str,
    expected_prompt_token_ids: Option<Vec<u32>>,
    tokenizer_fn: Option<Bound<'py, PyAny>>,
    detokenizer_fn: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let response = verilm_core::serialize::deserialize_v4_audit(audit_binary)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize V4 binary: {}", e)))?;
    let key: VerifierKey = serde_json::from_str(key_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize VerifierKey: {}", e)))?;

    let report = run_verify(
        &key,
        &response,
        expected_prompt_token_ids.as_deref(),
        tokenizer_fn,
        detokenizer_fn,
    );

    let result = PyDict::new(py);
    result.set_item("passed", report.verdict == verilm_verify::Verdict::Pass)?;
    result.set_item("checks_run", report.checks_run)?;
    result.set_item("checks_passed", report.checks_passed)?;
    let failure_msgs: Vec<&str> = report.failures.iter().map(|f| f.message.as_str()).collect();
    result.set_item("failures", &failure_msgs)?;
    result.set_item(
        "classified_failures",
        classified_failures_to_py(py, &report.failures)?,
    )?;
    result.set_item("coverage", coverage_to_py(py, &report.coverage))?;
    result.set_item("duration_us", report.duration.as_micros() as u64)?;
    Ok(result)
}

/// Convert classified failures to a Python list of dicts.
fn classified_failures_to_py<'py>(
    py: Python<'py>,
    failures: &[verilm_verify::VerificationFailure],
) -> PyResult<Bound<'py, PyList>> {
    let items: Vec<Bound<'py, PyDict>> = failures
        .iter()
        .map(|f| {
            let d = PyDict::new(py);
            d.set_item("code", f.code.to_string()).unwrap();
            d.set_item("category", f.category.to_string()).unwrap();
            d.set_item("message", &f.message).unwrap();
            // Context: only include non-None fields
            let ctx = PyDict::new(py);
            if let Some(ti) = f.context.token_index {
                ctx.set_item("token_index", ti).unwrap();
            }
            if let Some(l) = f.context.layer {
                ctx.set_item("layer", l).unwrap();
            }
            if let Some(ref m) = f.context.matrix {
                ctx.set_item("matrix", m).unwrap();
            }
            if let Some(ref fld) = f.context.field {
                ctx.set_item("field", fld).unwrap();
            }
            if let Some(ref s) = f.context.spec {
                ctx.set_item("spec", s).unwrap();
            }
            if let Some(ref e) = f.context.expected {
                ctx.set_item("expected", e).unwrap();
            }
            if let Some(ref a) = f.context.actual {
                ctx.set_item("actual", a).unwrap();
            }
            if ctx.len() > 0 {
                d.set_item("context", ctx).unwrap();
            }
            d
        })
        .collect();
    Ok(PyList::new(py, items)?)
}

/// Convert AuditCoverage to a Python dict.
fn coverage_to_py<'py>(
    py: Python<'py>,
    coverage: &verilm_verify::AuditCoverage,
) -> Bound<'py, PyDict> {
    let d = PyDict::new(py);
    match coverage {
        verilm_verify::AuditCoverage::Full { layers_checked } => {
            d.set_item("level", "full").unwrap();
            d.set_item("layers_checked", *layers_checked).unwrap();
        }
        verilm_verify::AuditCoverage::Routine {
            layers_checked,
            layers_total,
        } => {
            d.set_item("level", "routine").unwrap();
            d.set_item("layers_checked", *layers_checked).unwrap();
            d.set_item("layers_total", *layers_total).unwrap();
        }
        verilm_verify::AuditCoverage::Unknown => {
            d.set_item("level", "unknown").unwrap();
        }
    }
    d
}

/// Shared verification logic: delegates to canonical verifier via verify_v4_full.
/// Note: expected_prompt_token_ids is ignored by the canonical path.
fn run_verify(
    key: &VerifierKey,
    response: &V4AuditResponse,
    expected_prompt_token_ids: Option<&[u32]>,
    tokenizer_fn: Option<Bound<'_, PyAny>>,
    detokenizer_fn: Option<Bound<'_, PyAny>>,
) -> verilm_verify::V4VerifyReport {
    let tok_wrapper = tokenizer_fn.map(|cb| PyCallableTokenizer { callback: cb });
    let detok_wrapper = detokenizer_fn.map(|cb| PyCallableDetokenizer { callback: cb });
    let tok_ref: Option<&dyn verilm_verify::PromptTokenizer> = tok_wrapper
        .as_ref()
        .map(|t| t as &dyn verilm_verify::PromptTokenizer);
    let detok_ref: Option<&dyn verilm_verify::Detokenizer> = detok_wrapper
        .as_ref()
        .map(|d| d as &dyn verilm_verify::Detokenizer);
    verilm_verify::verify_v4_full(key, response, expected_prompt_token_ids, tok_ref, detok_ref)
}

/// Verify that externally-computed prompt token IDs match the committed token chain.
///
/// The caller tokenizes the raw prompt using the committed InputSpec and passes
/// the resulting token IDs. Returns a dict with `passed` (bool) and `failures` (list[str]).
///
/// Args:
///     audit_json: str — JSON-serialized V4AuditResponse.
///     expected_prompt_token_ids: list[int] — token IDs from external tokenizer.
///
/// Returns:
///     dict with `passed` (bool), `failures` (list[str])
#[pyfunction]
fn verify_input_tokenization<'py>(
    py: Python<'py>,
    audit_json: &str,
    expected_prompt_token_ids: Vec<u32>,
) -> PyResult<Bound<'py, PyDict>> {
    let response: V4AuditResponse = serde_json::from_str(audit_json).map_err(|e| {
        PyValueError::new_err(format!("failed to deserialize V4AuditResponse: {}", e))
    })?;

    let failures = verilm_verify::verify_input_tokenization(&response, &expected_prompt_token_ids);

    let result = PyDict::new(py);
    result.set_item("passed", failures.is_empty())?;
    let failure_msgs: Vec<&str> = failures.iter().map(|f| f.message.as_str()).collect();
    result.set_item("failures", &failure_msgs)?;
    Ok(result)
}

/// Derive a per-token PRNG seed from the batch seed and token index.
///
/// Implements `SHA256("vi-sample-v1" || batch_seed || token_index_le32)`.
///
/// Args:
///     batch_seed: bytes — 32-byte batch seed (revealed at audit time).
///     token_index: int — zero-based index of the token within the batch.
///
/// Returns:
///     bytes — 32-byte per-token seed.
#[pyfunction]
fn derive_token_seed<'py>(
    py: Python<'py>,
    batch_seed: &[u8],
    token_index: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    if batch_seed.len() != 32 {
        return Err(PyValueError::new_err("batch_seed must be exactly 32 bytes"));
    }
    let mut seed_arr = [0u8; 32];
    seed_arr.copy_from_slice(batch_seed);
    let token_seed = verilm_core::sampling::derive_token_seed(&seed_arr, token_index);
    Ok(PyBytes::new(py, &token_seed))
}

/// Run the canonical sampler on logits.
///
/// This is the exact same algorithm the verifier uses. Given logits (as f32
/// list), decode parameters, and a per-token seed, returns the selected token
/// index.
///
/// Args:
///     logits: list[float] — raw logits for the full vocabulary.
///     temperature: float — sampling temperature (0.0 = greedy argmax).
///     top_k: int — top-k filtering (0 = disabled).
///     top_p: float — nucleus sampling threshold (1.0 = disabled).
///     token_seed: bytes — 32-byte per-token seed from derive_token_seed.
///
/// Returns:
///     int — the selected token ID.
#[pyfunction]
fn canonical_sample(
    logits: &Bound<'_, PyAny>,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    token_seed: &[u8],
) -> PyResult<u32> {
    let logits = extract_f32_vec(logits)?;
    if logits.is_empty() {
        return Err(PyValueError::new_err("logits must not be empty"));
    }
    if token_seed.len() != 32 {
        return Err(PyValueError::new_err("token_seed must be exactly 32 bytes"));
    }
    let mut seed_arr = [0u8; 32];
    seed_arr.copy_from_slice(token_seed);
    let params = verilm_core::sampling::DecodeParams {
        temperature,
        top_k,
        top_p,
    };
    Ok(verilm_core::sampling::sample(&logits, &params, &seed_arr))
}

// ── CaptureHook: Rust-side bookkeeping for the capture wrapper ──
//
// Moves counter arithmetic, scale storage, and proj identification out of
// Python bytecode. The Python wrapper becomes:
//
//   output = real(a, b, scale_a, scale_b, out_dtype, bias)
//   if hook is not None and buf.enabled:
//       is_o_proj = hook.record(scale_a)
//       if is_o_proj: buf._slab_copy_o_proj(a)
//   return output
//
// Each record() call: ~2-3µs (PyO3 round-trip) vs ~9µs (Python bytecode).
// At 28K calls/request, saves ~170ms.

#[pyclass]
struct CaptureHook {
    calls_per_fwd: usize,
    projs_per_layer: usize,
    call_counter: usize,
    minimal_call_count: usize,
    total_captured: usize,
    scales: Vec<Py<PyAny>>,
}

#[pymethods]
impl CaptureHook {
    #[new]
    fn new(
        calls_per_fwd: usize,
        projs_per_layer: usize,
        #[allow(unused)] o_proj_idx: usize,
    ) -> Self {
        CaptureHook {
            calls_per_fwd,
            projs_per_layer,
            call_counter: 0,
            minimal_call_count: 0,
            total_captured: 0,
            scales: Vec::with_capacity(65536),
        }
    }

    /// Record a scale capture. Returns the projection index within the layer
    /// (0=qkv, 1=o_proj, 2=gate_up, 3=down). Caller uses this to decide
    /// which input tensor to capture (o_proj for a_i8, qkv for x_attn_i8).
    fn record(&mut self, scale_a: Py<PyAny>) -> i32 {
        let idx = self.call_counter % self.calls_per_fwd;
        let proj_idx = idx % self.projs_per_layer;
        self.call_counter += 1;
        self.scales.push(scale_a);
        self.minimal_call_count += 1;
        self.total_captured += 1;
        proj_idx as i32
    }

    /// Drain accumulated scales. Returns (numpy_array, call_count).
    ///
    /// Does torch.cat + .cpu() + .numpy() internally via PyO3 — avoids
    /// returning 28K Python tensor objects that would need to be passed
    /// to C++ for bulk cat (which costs ~30ms in pybind11 conversion).
    /// Doing the cat here saves ~30ms per request.
    fn drain_scales(&mut self, py: Python<'_>) -> PyResult<(Py<PyAny>, usize)> {
        let count = self.minimal_call_count;
        self.minimal_call_count = 0;

        if self.scales.is_empty() {
            let numpy = py.import("numpy")?;
            let empty = numpy.call_method(
                "array",
                (Vec::<f32>::new(),),
                Some(&[("dtype", numpy.getattr("float32")?)].into_py_dict(py)?),
            )?;
            return Ok((empty.unbind(), 0));
        }

        let scales = std::mem::take(&mut self.scales);
        let torch_mod = py.import("torch")?;
        let list = PyList::new(py, &scales)?;

        // Fast path: torch.cat directly (works when all tensors are 1D — 99.6% of calls).
        let cat = match torch_mod.call_method1("cat", (&list,)) {
            Ok(t) => t,
            Err(_) => {
                // Mixed dimensions (prefill 2D + decode 1D): flatten each, re-cat.
                let flat = PyList::empty(py);
                for item in list.iter() {
                    flat.append(item.call_method0("flatten")?)?;
                }
                torch_mod.call_method1("cat", (&flat,))?
            }
        };

        // GPU→CPU + numpy: one bulk transfer.
        // Per-row scales are preserved (no .max() reduction) so that each
        // token in a prefill batch gets its own scale_a.
        let numpy_arr = cat.call_method0("cpu")?.call_method0("numpy")?;

        Ok((numpy_arr.unbind(), count))
    }

    /// Reset the call counter to zero (between requests).
    fn reset_counter(&mut self) {
        self.call_counter = 0;
    }

    #[getter]
    fn get_call_counter(&self) -> usize {
        self.call_counter
    }

    #[getter]
    fn get_total_captured(&self) -> usize {
        self.total_captured
    }

    #[getter]
    fn get_minimal_call_count(&self) -> usize {
        self.minimal_call_count
    }
}

/// Measure the attention corridor for a deep-prefix audit.
///
/// Takes a binary V4 audit and a JSON verifier key, replays attention from
/// the QKV accumulators, and returns a JSON-serialized `CorridorReport`
/// with per-layer/per-position diff statistics.
///
/// When `scale_overrides_json` is provided, uses per-channel weight scales
/// for dequantization. Required for faithful measurement on W8A8 models
/// where per-tensor weight scales are 0.0 (native INT8).
///
/// Args:
///     audit_binary: bytes — V4 audit in canonical binary format.
///     key_json: str — JSON-serialized VerifierKey.
///     scale_overrides_json: Optional[str] — JSON-serialized CorridorScaleOverrides.
///
/// Returns:
///     str — JSON-serialized CorridorReport.
#[pyfunction]
#[pyo3(signature = (audit_binary, key_json, scale_overrides_json=None))]
fn measure_corridor(
    audit_binary: &[u8],
    key_json: &str,
    scale_overrides_json: Option<&str>,
) -> PyResult<String> {
    let key: VerifierKey = serde_json::from_str(key_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize key: {}", e)))?;
    let response = verilm_core::serialize::deserialize_v4_audit(audit_binary)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize audit: {}", e)))?;
    let overrides = scale_overrides_json
        .map(|json| {
            serde_json::from_str::<verilm_verify::corridor::CorridorScaleOverrides>(json).map_err(
                |e| PyValueError::new_err(format!("failed to deserialize scale overrides: {}", e)),
            )
        })
        .transpose()?;
    let report = verilm_verify::corridor::measure_corridor(&key, &response, overrides.as_ref())
        .map_err(|e| PyValueError::new_err(format!("corridor measurement failed: {}", e)))?;
    serde_json::to_string(&report)
        .map_err(|e| PyValueError::new_err(format!("serialization error: {}", e)))
}

/// Measure the attention corridor using committed KV entries.
///
/// Uses `kv_entries` from the audit response (committed under `kv_roots`)
/// directly as the KV cache. This is the production measurement path —
/// does NOT require deep_prefix data.
///
/// Args:
///     audit_binary: bytes — V4 audit in canonical binary format.
///     key_json: str — JSON-serialized VerifierKey.
///     scale_overrides_json: Optional[str] — JSON-serialized CorridorScaleOverrides.
///
/// Returns:
///     str — JSON-serialized CorridorReport.
#[pyfunction]
#[pyo3(signature = (audit_binary, key_json, scale_overrides_json=None))]
fn measure_corridor_committed_kv(
    audit_binary: &[u8],
    key_json: &str,
    scale_overrides_json: Option<&str>,
) -> PyResult<String> {
    let key: VerifierKey = serde_json::from_str(key_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize key: {}", e)))?;
    let response = verilm_core::serialize::deserialize_v4_audit(audit_binary)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize audit: {}", e)))?;
    let overrides = scale_overrides_json
        .map(|json| {
            serde_json::from_str::<verilm_verify::corridor::CorridorScaleOverrides>(json).map_err(
                |e| PyValueError::new_err(format!("failed to deserialize scale overrides: {}", e)),
            )
        })
        .transpose()?;
    let report =
        verilm_verify::corridor::measure_corridor_committed_kv(&key, &response, overrides.as_ref())
            .map_err(|e| PyValueError::new_err(format!("corridor measurement failed: {}", e)))?;
    serde_json::to_string(&report)
        .map_err(|e| PyValueError::new_err(format!("serialization error: {}", e)))
}

/// Measure the attention corridor with selectable replay precision.
///
/// Args:
///     audit_binary: bytes — bincode-serialized V4AuditResponse.
///     key_json: str — JSON-serialized VerifierKey.
///     precision: str — one of "f64", "f32", "fp16_f32", "bf16_f32".
///     scale_overrides_json: Optional[str] — JSON-serialized CorridorScaleOverrides.
///
/// Returns:
///     str — JSON-serialized CorridorReport.
#[pyfunction]
#[pyo3(signature = (audit_binary, key_json, precision, scale_overrides_json=None))]
fn measure_corridor_precision(
    audit_binary: &[u8],
    key_json: &str,
    precision: &str,
    scale_overrides_json: Option<&str>,
) -> PyResult<String> {
    let replay_precision = match precision {
        "f64" => verilm_core::attention::ReplayPrecision::F64,
        "f32" => verilm_core::attention::ReplayPrecision::F32,
        "fp16_f32" => verilm_core::attention::ReplayPrecision::Fp16InputsF32Accum,
        "bf16_f32" => verilm_core::attention::ReplayPrecision::Bf16InputsF32Accum,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown precision '{}': expected f64, f32, fp16_f32, or bf16_f32",
                other
            )));
        }
    };
    let key: VerifierKey = serde_json::from_str(key_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize key: {}", e)))?;
    let response = verilm_core::serialize::deserialize_v4_audit(audit_binary)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize audit: {}", e)))?;
    let overrides = scale_overrides_json
        .map(|json| {
            serde_json::from_str::<verilm_verify::corridor::CorridorScaleOverrides>(json).map_err(
                |e| PyValueError::new_err(format!("failed to deserialize scale overrides: {}", e)),
            )
        })
        .transpose()?;
    let report = verilm_verify::corridor::measure_corridor_committed_kv_precision(
        &key,
        &response,
        overrides.as_ref(),
        replay_precision,
    )
    .map_err(|e| PyValueError::new_err(format!("corridor measurement failed: {}", e)))?;
    serde_json::to_string(&report)
        .map_err(|e| PyValueError::new_err(format!("serialization error: {}", e)))
}

/// Backwards-compatible wrapper: f32 corridor measurement.
#[pyfunction]
#[pyo3(signature = (audit_binary, key_json, scale_overrides_json=None))]
fn measure_corridor_committed_kv_f32(
    audit_binary: &[u8],
    key_json: &str,
    scale_overrides_json: Option<&str>,
) -> PyResult<String> {
    measure_corridor_precision(audit_binary, key_json, "f32", scale_overrides_json)
}

/// Simulate per-head vs per-tensor boundary strategies on audit data.
///
/// Replays attention for each layer and compares requantization under:
///   - per-tensor scale (current protocol)
///   - per-head scale (simulated)
///
/// Reports corridor L-inf under each strategy and float-space adversarial room.
///
/// Args:
///     audit_binary: bytes — bincode-serialized V4AuditResponse.
///     key_json: str — JSON-serialized VerifierKey.
///     scale_overrides_json: Optional[str] — JSON-serialized CorridorScaleOverrides.
///
/// Returns:
///     str — JSON-serialized BoundarySimReport.
#[pyfunction]
#[pyo3(signature = (audit_binary, key_json, scale_overrides_json=None))]
fn simulate_boundary_strategies(
    audit_binary: &[u8],
    key_json: &str,
    scale_overrides_json: Option<&str>,
) -> PyResult<String> {
    let key: VerifierKey = serde_json::from_str(key_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize key: {}", e)))?;
    let response = verilm_core::serialize::deserialize_v4_audit(audit_binary)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize audit: {}", e)))?;
    let overrides = scale_overrides_json
        .map(|json| {
            serde_json::from_str::<verilm_verify::corridor::CorridorScaleOverrides>(json).map_err(
                |e| PyValueError::new_err(format!("failed to deserialize scale overrides: {}", e)),
            )
        })
        .transpose()?;
    let report = verilm_verify::corridor::simulate_boundary_strategies(
        &key,
        &response,
        overrides.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(format!("boundary simulation failed: {}", e)))?;
    serde_json::to_string(&report)
        .map_err(|e| PyValueError::new_err(format!("serialization error: {}", e)))
}

/// Simulate INT16 retained `a` against the current INT8 boundary.
///
/// Uses the same audit payload and committed-KV replay path as the corridor
/// tooling, but evaluates whether an INT16 retained boundary would reduce the
/// honest float-space mismatch.
#[pyfunction]
#[pyo3(signature = (audit_binary, key_json, scale_overrides_json=None))]
fn simulate_int16_boundary(
    audit_binary: &[u8],
    key_json: &str,
    scale_overrides_json: Option<&str>,
) -> PyResult<String> {
    let key: VerifierKey = serde_json::from_str(key_json)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize key: {}", e)))?;
    let response = verilm_core::serialize::deserialize_v4_audit(audit_binary)
        .map_err(|e| PyValueError::new_err(format!("failed to deserialize audit: {}", e)))?;
    let overrides = scale_overrides_json
        .map(|json| {
            serde_json::from_str::<verilm_verify::corridor::CorridorScaleOverrides>(json).map_err(
                |e| PyValueError::new_err(format!("failed to deserialize scale overrides: {}", e)),
            )
        })
        .transpose()?;
    let report = verilm_verify::corridor::simulate_int16_boundary(
        &key,
        &response,
        overrides.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(format!("int16 boundary simulation failed: {}", e)))?;
    serde_json::to_string(&report)
        .map_err(|e| PyValueError::new_err(format!("serialization error: {}", e)))
}

#[pymodule]
fn verilm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(commit_minimal_from_captures, m)?)?;
    m.add_function(wrap_pyfunction!(build_audit_challenge, m)?)?;
    m.add_function(wrap_pyfunction!(compute_weight_hash, m)?)?;
    m.add_function(wrap_pyfunction!(generate_key, m)?)?;
    m.add_function(wrap_pyfunction!(verify_v4, m)?)?;
    m.add_function(wrap_pyfunction!(verify_v4_binary, m)?)?;
    m.add_function(wrap_pyfunction!(derive_token_seed, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_sample, m)?)?;
    m.add_class::<WeightProvider>()?;
    m.add_class::<MinimalBatchStateHandle>()?;
    m.add_function(wrap_pyfunction!(commit_minimal_packed, m)?)?;
    m.add_class::<PackedBatchStateHandle>()?;
    m.add_function(wrap_pyfunction!(measure_corridor, m)?)?;
    m.add_function(wrap_pyfunction!(measure_corridor_committed_kv, m)?)?;
    m.add_function(wrap_pyfunction!(measure_corridor_precision, m)?)?;
    m.add_function(wrap_pyfunction!(measure_corridor_committed_kv_f32, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_boundary_strategies, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_int16_boundary, m)?)?;
    m.add_class::<CaptureHook>()?;
    m.add_function(wrap_pyfunction!(verify_input_tokenization, m)?)?;
    Ok(())
}
