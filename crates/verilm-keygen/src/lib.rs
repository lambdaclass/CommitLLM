//! Verifier-secret key generation from safetensors model weights.
//!
//! Reads weight matrices from safetensors files (INT8, BF16, or FP16),
//! generates random r_j vectors, and precomputes v_j^(i) = r_j^T W_j^(i)
//! for all layers and matrix types.
//!
//! BF16/FP16 weights are quantized to INT8 using absmax per-tensor
//! quantization during keygen. The same quantization must be applied
//! during inference for Freivalds checks to pass.
//!
//! # Security
//!
//! The generated key is **verifier-secret**. It contains random vectors
//! r_j that must never be revealed to the prover. Since the prover knows
//! the model weights W, leaking v_j = r_j^T W_j is equivalent to leaking
//! r_j (solvable linear system). The entire key must stay with the verifier.
//!
//! This tool should be run by the verifier (or a trusted party), not the
//! prover. The prover needs no key material.

use std::fs::File;
use std::path::Path;

use anyhow::{bail, Context, Result};
use memmap2::MmapOptions;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use safetensors::{Dtype, SafeTensors};

use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::field::Fp;
use verilm_core::freivalds;
use verilm_core::types::{ShellWeights, VerifierKey};

/// A memory-mapped safetensors file.
struct MappedShard {
    _file: File,
    mmap: memmap2::Mmap,
}

impl MappedShard {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(MappedShard { _file: file, mmap })
    }
}

/// Open all .safetensors files in a directory.
fn open_shards(dir: &Path) -> Result<Vec<MappedShard>> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    if paths.is_empty() {
        bail!("no .safetensors files found in {}", dir.display());
    }
    paths.sort();
    paths.iter().map(|p| MappedShard::open(p)).collect()
}

/// Look up a tensor by name across shards. Returns (raw bytes, shape, dtype).
fn find_tensor_raw<'a>(
    shards: &'a [(SafeTensors<'a>, &'a MappedShard)],
    name: &str,
) -> Result<(&'a [u8], Vec<usize>, Dtype)> {
    for (st, _) in shards {
        if let Ok(view) = st.tensor(name) {
            return Ok((view.data(), view.shape().to_vec(), view.dtype()));
        }
    }
    bail!("tensor {} not found in any shard", name)
}

/// Convert BF16 bytes to f32.
fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

/// Convert FP16 bytes to f32.
fn fp16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            half_to_f32(bits)
        })
        .collect()
}

/// IEEE 754 half-precision to single-precision.
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    let result = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal
            let mut m = mant;
            let mut e = 0u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3ff;
            (sign << 31) | ((127 - 15 + 1 - e) << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (mant << 13) // Inf/NaN
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(result)
}

/// Absmax quantization: quantize f32 values to INT8.
/// Returns (quantized_weights, scale) where w_f32 ≈ quantized * scale.
fn quantize_absmax(values: &[f32]) -> (Vec<i8>, f32) {
    let absmax = values
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    if absmax == 0.0 {
        return (vec![0i8; values.len()], 1.0);
    }
    let scale = absmax / 127.0;
    let quantized: Vec<i8> = values
        .iter()
        .map(|&v| (v / scale).round().clamp(-128.0, 127.0) as i8)
        .collect();
    (quantized, scale)
}

/// Load a weight tensor as INT8. If the tensor is BF16/FP16, quantize it.
/// Returns (INT8 weights, quantization scale). Scale is 0.0 for native INT8.
fn load_weights_as_i8(
    shards: &[(SafeTensors<'_>, &MappedShard)],
    name: &str,
) -> Result<(Vec<i8>, f32)> {
    let (data, _shape, dtype) = find_tensor_raw(shards, name)?;
    match dtype {
        Dtype::I8 => {
            let weights: Vec<i8> = data.iter().map(|&b| b as i8).collect();
            Ok((weights, 0.0))
        }
        Dtype::BF16 => {
            let f32_vals = bf16_to_f32(data);
            let (quantized, scale) = quantize_absmax(&f32_vals);
            eprintln!("    {} BF16->INT8 (scale={:.6})", name, scale);
            Ok((quantized, scale))
        }
        Dtype::F16 => {
            let f32_vals = fp16_to_f32(data);
            let (quantized, scale) = quantize_absmax(&f32_vals);
            eprintln!("    {} FP16->INT8 (scale={:.6})", name, scale);
            Ok((quantized, scale))
        }
        other => bail!("tensor {} has unsupported dtype {:?}", name, other),
    }
}

/// Load a 1D tensor as f32. Supports F32, BF16, and F16 source dtypes.
fn load_1d_f32(
    shards: &[(SafeTensors<'_>, &MappedShard)],
    name: &str,
) -> Result<Vec<f32>> {
    let (data, shape, dtype) = find_tensor_raw(shards, name)?;
    if shape.len() != 1 {
        bail!("tensor {} has shape {:?}, expected 1D", name, shape);
    }
    match dtype {
        Dtype::F32 => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        Dtype::BF16 => Ok(bf16_to_f32(data)),
        Dtype::F16 => Ok(fp16_to_f32(data)),
        other => bail!("tensor {} has unsupported dtype {:?}", name, other),
    }
}

/// Compute SHA-256 leaf hashes for every row of a 2D embedding tensor.
fn compute_embedding_hashes(
    shards: &[(SafeTensors<'_>, &MappedShard)],
    name: &str,
) -> Result<Vec<[u8; 32]>> {
    let (data, shape, dtype) = find_tensor_raw(shards, name)?;
    if shape.len() != 2 {
        bail!("embedding tensor {} shape {:?}, expected 2D", name, shape);
    }
    let rows = shape[0];
    let cols = shape[1];
    let bytes_per_elem: usize = match dtype {
        Dtype::F32 => 4,
        Dtype::BF16 | Dtype::F16 => 2,
        other => bail!("embedding tensor {} has unsupported dtype {:?}", name, other),
    };
    let mut hashes = Vec::with_capacity(rows);
    for row in 0..rows {
        let offset = row * cols * bytes_per_elem;
        let row_data = &data[offset..offset + cols * bytes_per_elem];
        let f32_row: Vec<f32> = match dtype {
            Dtype::F32 => row_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            Dtype::BF16 => bf16_to_f32(row_data),
            Dtype::F16 => fp16_to_f32(row_data),
            _ => unreachable!(),
        };
        hashes.push(verilm_core::merkle::hash_embedding_row(&f32_row));
    }
    eprintln!("  hashed {} embedding rows ({}D)", rows, cols);
    Ok(hashes)
}

/// Load a single row from a 2D tensor as f32.
fn load_2d_row_f32(
    shards: &[(SafeTensors<'_>, &MappedShard)],
    name: &str,
    row: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    let (data, shape, dtype) = find_tensor_raw(shards, name)?;
    if shape.len() != 2 || shape[1] != cols {
        bail!("tensor {} shape {:?}, expected [_, {}]", name, shape, cols);
    }
    if row >= shape[0] {
        bail!("row {} out of range for {} (rows={})", row, name, shape[0]);
    }
    let bytes_per_elem: usize = match dtype {
        Dtype::F32 => 4,
        Dtype::BF16 | Dtype::F16 => 2,
        other => bail!("tensor {} has unsupported dtype {:?}", name, other),
    };
    let offset = row * cols * bytes_per_elem;
    let row_data = &data[offset..offset + cols * bytes_per_elem];
    match dtype {
        Dtype::F32 => Ok(row_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        Dtype::BF16 => Ok(bf16_to_f32(row_data)),
        Dtype::F16 => Ok(fp16_to_f32(row_data)),
        _ => unreachable!(),
    }
}

/// Detect model config by inspecting tensor shapes.
pub fn detect_config(dir: &Path) -> Result<ModelConfig> {
    let mapped = open_shards(dir)?;
    let parsed: Vec<_> = mapped
        .iter()
        .map(|s| SafeTensors::deserialize(&s.mmap).map(|st| (st, s)))
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to parse safetensors headers")?;

    // Find layer 0 q_proj to get hidden_dim
    let q_name = MatrixType::Wq.weight_name().replace("{}", "0");
    let (_, q_shape, _) = find_tensor_raw(&parsed, &q_name)?;
    let hidden_dim = q_shape[0];

    // Find layer 0 k_proj to get kv_dim
    let k_name = MatrixType::Wk.weight_name().replace("{}", "0");
    let (_, k_shape, _) = find_tensor_raw(&parsed, &k_name)?;
    let kv_dim = k_shape[0];

    // Find layer 0 gate_proj to get ffn_dim
    let g_name = MatrixType::Wg.weight_name().replace("{}", "0");
    let (_, g_shape, _) = find_tensor_raw(&parsed, &g_name)?;
    let ffn_dim = g_shape[0];

    // Count layers
    let mut n_layers = 0;
    loop {
        let name = MatrixType::Wq.weight_name().replace("{}", &n_layers.to_string());
        let found = parsed.iter().any(|(st, _)| st.tensor(&name).is_ok());
        if !found {
            break;
        }
        n_layers += 1;
    }

    if n_layers == 0 {
        bail!("no transformer layers found");
    }

    // Derive head counts.
    let d_head = if hidden_dim % 128 == 0 && kv_dim % 128 == 0 {
        128
    } else {
        gcd(hidden_dim, kv_dim)
    };

    let n_q_heads = hidden_dim / d_head;
    let n_kv_heads = kv_dim / d_head;

    // Llama-3 family uses rope_theta=500000.0; smaller/toy models use 10000.0
    let rope_theta = if hidden_dim >= 4096 { 500000.0 } else { 10000.0 };

    let config = ModelConfig {
        name: format!("detected-{}L-{}d", n_layers, hidden_dim),
        hidden_dim,
        kv_dim,
        ffn_dim,
        d_head,
        n_layers,
        n_q_heads,
        n_kv_heads,
        vocab_size: 0,
        rope_theta,
    };

    eprintln!(
        "Detected: {} layers, hidden_dim={}, kv_dim={}, ffn_dim={}, d_head={}, q_heads={}, kv_heads={}",
        n_layers, hidden_dim, kv_dim, ffn_dim, d_head, n_q_heads, n_kv_heads
    );

    Ok(config)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Generate a verifier key from safetensors model weights.
///
/// Streams one layer at a time. BF16/FP16 weights are quantized to INT8
/// using absmax per-tensor quantization.
pub fn generate_key(dir: &Path, seed: [u8; 32]) -> Result<VerifierKey> {
    let cfg = detect_config(dir)?;
    let mapped = open_shards(dir)?;
    let parsed: Vec<_> = mapped
        .iter()
        .map(|s| SafeTensors::deserialize(&s.mmap).map(|st| (st, s)))
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to parse safetensors")?;

    let mut rng = ChaCha20Rng::from_seed(seed);

    // Generate per-matrix-type r vectors
    let r_vectors: Vec<Vec<Fp>> = MatrixType::ALL
        .iter()
        .map(|mt| {
            let dim = mt.output_dim(&cfg);
            (0..dim).map(|_| Fp::new(rng.gen::<u32>())).collect()
        })
        .collect();

    // Detect source dtype from first tensor.
    // Use fixed protocol literals, not Debug format, so the weight hash
    // is stable across library versions.
    let first_name = MatrixType::Wq.weight_name().replace("{}", "0");
    let (_, _, first_dtype) = find_tensor_raw(&parsed, &first_name)?;
    let source_dtype = match first_dtype {
        Dtype::I8 => "I8",
        Dtype::BF16 => "BF16",
        Dtype::F16 => "F16",
        other => bail!("unsupported source dtype {:?} — expected I8, BF16, or F16", other),
    }
    .to_string();

    // Precompute v_j^(i) for each layer and matrix type.
    // Also collect all INT8 weights for the weight-chain hash.
    let mut v_vectors: Vec<Vec<Vec<Fp>>> = Vec::with_capacity(cfg.n_layers);
    let mut quantization_scales: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_layers);
    let mut all_weights: Vec<Vec<Vec<i8>>> = Vec::with_capacity(cfg.n_layers);
    let mut rmsnorm_attn_weights: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_layers);
    let mut rmsnorm_ffn_weights: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_layers);

    for layer_idx in 0..cfg.n_layers {
        let mut layer_vs = Vec::with_capacity(7);
        let mut layer_scales = Vec::with_capacity(7);
        let mut layer_weights = Vec::with_capacity(7);

        for (j, mt) in MatrixType::ALL.iter().enumerate() {
            let name = mt.weight_name().replace("{}", &layer_idx.to_string());
            let (weights, scale) = load_weights_as_i8(&parsed, &name)?;

            let rows = mt.output_dim(&cfg);
            let cols = mt.input_dim(&cfg);

            if weights.len() != rows * cols {
                bail!(
                    "tensor {} has {} elements, expected {}",
                    name,
                    weights.len(),
                    rows * cols
                );
            }

            let r = &r_vectors[j];
            let v = freivalds::precompute_v(r, &weights, rows, cols);
            layer_vs.push(v);
            layer_scales.push(scale);
            layer_weights.push(weights);
        }

        // Extract RMSNorm weights for this layer (attention + FFN)
        let attn_norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
        match load_1d_f32(&parsed, &attn_norm_name) {
            Ok(w) => rmsnorm_attn_weights.push(w),
            Err(e) => {
                eprintln!("  warning: could not load {}: {}", attn_norm_name, e);
                rmsnorm_attn_weights.push(Vec::new());
            }
        }

        let ffn_norm_name =
            format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
        match load_1d_f32(&parsed, &ffn_norm_name) {
            Ok(w) => rmsnorm_ffn_weights.push(w),
            Err(e) => {
                eprintln!("  warning: could not load {}: {}", ffn_norm_name, e);
                rmsnorm_ffn_weights.push(Vec::new());
            }
        }

        v_vectors.push(layer_vs);
        quantization_scales.push(layer_scales);
        all_weights.push(layer_weights);
        eprintln!("  layer {}/{}", layer_idx + 1, cfg.n_layers);
    }

    // Compute weight-chain hash
    let weight_hash = verilm_core::merkle::hash_weights(
        &source_dtype,
        cfg.n_layers,
        &quantization_scales,
        |layer, mt_idx| all_weights[layer][mt_idx].clone(),
        MatrixType::ALL.len(),
    );

    eprintln!("  weight hash: {}", hex::encode(weight_hash));

    // Compute embedding Merkle root
    let emb_name = "model.embed_tokens.weight";
    let embedding_merkle_root = match compute_embedding_hashes(&parsed, emb_name) {
        Ok(hashes) => {
            let root = verilm_core::merkle::compute_root(&hashes);
            eprintln!("  embedding Merkle root: {}", hex::encode(root));
            Some(root)
        }
        Err(e) => {
            eprintln!("  warning: could not compute embedding Merkle root: {}", e);
            None
        }
    };

    // Load lm_head (unembedding matrix) for logit verification.
    let lm_head = match load_weights_as_i8(&parsed, "lm_head.weight") {
        Ok((weights, scale)) => {
            let (_, lm_shape, _) = find_tensor_raw(&parsed, "lm_head.weight")?;
            let vocab_size = lm_shape[0];
            eprintln!("  lm_head: {}x{} (scale={:.6})", vocab_size, cfg.hidden_dim, scale);
            Some(weights)
        }
        Err(e) => {
            eprintln!("  warning: could not load lm_head.weight: {}", e);
            None
        }
    };

    // Detect vocab_size from lm_head shape (more reliable than config.json).
    let vocab_size = if lm_head.is_some() {
        let (_, lm_shape, _) = find_tensor_raw(&parsed, "lm_head.weight")?;
        lm_shape[0]
    } else {
        0
    };

    // Load final RMSNorm weights (model.norm.weight).
    let final_norm_weights = match load_1d_f32(&parsed, "model.norm.weight") {
        Ok(w) => {
            eprintln!("  final RMSNorm: {} dims", w.len());
            Some(w)
        }
        Err(e) => {
            eprintln!("  warning: could not load model.norm.weight: {}", e);
            None
        }
    };

    let mut config = cfg;
    if vocab_size > 0 {
        config.vocab_size = vocab_size;
    }

    Ok(VerifierKey {
        version: 1,
        config,
        seed,
        source_dtype,
        weight_scales: quantization_scales.clone(),
        quantization_scales,
        r_vectors,
        v_vectors,
        wo_norms: Vec::new(),
        max_v_norm: 0.0,
        lm_head,
        weight_hash: Some(weight_hash),
        rmsnorm_attn_weights,
        rmsnorm_ffn_weights,
        rmsnorm_eps: 1e-5,
        embedding_merkle_root,
        final_norm_weights,
    })
}

/// Compute the paper's R_W (weight-chain hash) without full keygen.
///
/// Loads all safetensors, quantizes to INT8 if needed, and hashes in
/// canonical order. Returns the 32-byte SHA-256 weight hash.
///
/// This is cheaper than `generate_key` — it skips r/v vector generation
/// and RMSNorm weight extraction.
pub fn compute_weight_hash(dir: &Path) -> Result<[u8; 32]> {
    let cfg = detect_config(dir)?;
    let mapped = open_shards(dir)?;
    let parsed: Vec<_> = mapped
        .iter()
        .map(|s| SafeTensors::deserialize(&s.mmap).map(|st| (st, s)))
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to parse safetensors")?;

    // Detect source dtype.
    let first_name = MatrixType::Wq.weight_name().replace("{}", "0");
    let (_, _, first_dtype) = find_tensor_raw(&parsed, &first_name)?;
    let source_dtype = match first_dtype {
        Dtype::I8 => "I8",
        Dtype::BF16 => "BF16",
        Dtype::F16 => "F16",
        other => bail!("unsupported source dtype {:?}", other),
    }
    .to_string();

    // Load weights and scales per layer.
    let mut quantization_scales: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_layers);
    let mut all_weights: Vec<Vec<Vec<i8>>> = Vec::with_capacity(cfg.n_layers);

    for layer_idx in 0..cfg.n_layers {
        let mut layer_scales = Vec::with_capacity(7);
        let mut layer_weights = Vec::with_capacity(7);

        for mt in MatrixType::ALL.iter() {
            let name = mt.weight_name().replace("{}", &layer_idx.to_string());
            let (weights, scale) = load_weights_as_i8(&parsed, &name)?;
            layer_scales.push(scale);
            layer_weights.push(weights);
        }

        quantization_scales.push(layer_scales);
        all_weights.push(layer_weights);
    }

    Ok(verilm_core::merkle::hash_weights(
        &source_dtype,
        cfg.n_layers,
        &quantization_scales,
        |layer, mt_idx| all_weights[layer][mt_idx].clone(),
        MatrixType::ALL.len(),
    ))
}

/// Write a safetensors file from a list of (name, shape, i8 data).
/// Used for testing.
pub fn write_safetensors(
    path: &Path,
    tensors: &[(&str, Vec<usize>, &[i8])],
) -> Result<()> {
    let views: Vec<_> = tensors
        .iter()
        .map(|(name, shape, data)| {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len())
            };
            (name.to_string(), Dtype::I8, shape.clone(), bytes)
        })
        .collect();

    let tensor_data: Vec<(String, safetensors::tensor::TensorView<'_>)> = views
        .iter()
        .map(|(name, dtype, shape, data)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        })
        .collect();

    let bytes = safetensors::tensor::serialize(tensor_data, &None)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Weight provider that loads INT8 matrices from safetensors on disk.
///
/// Used by the prover at audit time to compute shell openings.
/// Loads all weight matrices up front for the requested layers.
pub struct SafetensorsWeightProvider {
    config: ModelConfig,
    /// weights[layer][matrix_type_idx] = flattened i8 data
    weights: Vec<Vec<Vec<i8>>>,
    /// scales[layer][matrix_type_idx] = per-tensor quantization scale.
    /// 0.0 for native INT8 tensors.
    scales: Vec<Vec<f32>>,
    /// Per-layer RMSNorm attention weights.
    rmsnorm_attn_weights: Vec<Vec<f32>>,
    /// Per-layer RMSNorm FFN weights.
    rmsnorm_ffn_weights: Vec<Vec<f32>>,
    /// Embedding Merkle tree (for producing proofs at audit time).
    embedding_tree: Option<verilm_core::merkle::MerkleTree>,
    /// Path to model directory (for loading embedding rows on demand).
    model_dir: std::path::PathBuf,
    /// R_W: SHA-256 weight-chain hash over all INT8 weights and scales.
    weight_hash: [u8; 32],
}

impl SafetensorsWeightProvider {
    /// Load all weight matrices from a safetensors model directory.
    ///
    /// Also loads RMSNorm weights and builds an embedding Merkle tree
    /// for full bridge support at audit time.
    pub fn load(dir: &Path) -> Result<Self> {
        let config = detect_config(dir)?;
        let mapped = open_shards(dir)?;
        let parsed: Vec<_> = mapped
            .iter()
            .map(|s| SafeTensors::deserialize(&s.mmap).map(|st| (st, s)))
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to parse safetensors")?;

        let mut weights = Vec::with_capacity(config.n_layers);
        let mut scales = Vec::with_capacity(config.n_layers);
        let mut rmsnorm_attn_weights = Vec::with_capacity(config.n_layers);
        let mut rmsnorm_ffn_weights = Vec::with_capacity(config.n_layers);

        for layer_idx in 0..config.n_layers {
            let mut layer_weights = Vec::with_capacity(MatrixType::ALL.len());
            let mut layer_scales = Vec::with_capacity(MatrixType::ALL.len());
            for mt in MatrixType::ALL {
                let name = mt.weight_name().replace("{}", &layer_idx.to_string());
                let (w, scale) = load_weights_as_i8(&parsed, &name)?;
                layer_weights.push(w);
                layer_scales.push(scale);
            }
            weights.push(layer_weights);
            scales.push(layer_scales);

            // RMSNorm weights
            let attn_norm = format!("model.layers.{}.input_layernorm.weight", layer_idx);
            rmsnorm_attn_weights.push(load_1d_f32(&parsed, &attn_norm).unwrap_or_default());
            let ffn_norm = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
            rmsnorm_ffn_weights.push(load_1d_f32(&parsed, &ffn_norm).unwrap_or_default());
        }

        // Build embedding Merkle tree for proof generation at audit time
        let embedding_tree = compute_embedding_hashes(&parsed, "model.embed_tokens.weight")
            .map(|hashes| verilm_core::merkle::build_tree(&hashes))
            .ok();

        // Detect source dtype from the first weight tensor and compute R_W.
        let first_name = MatrixType::Wq.weight_name().replace("{}", "0");
        let (_, _, first_dtype) = find_tensor_raw(&parsed, &first_name)?;
        let source_dtype = match first_dtype {
            Dtype::I8 => "I8",
            Dtype::BF16 => "BF16",
            Dtype::F16 => "F16",
            other => bail!("unsupported source dtype {:?}", other),
        };

        let weight_hash = verilm_core::merkle::hash_weights(
            source_dtype,
            config.n_layers,
            &scales,
            |layer, mt_idx| weights[layer][mt_idx].clone(),
            MatrixType::ALL.len(),
        );

        Ok(SafetensorsWeightProvider {
            config,
            weights,
            scales,
            rmsnorm_attn_weights,
            rmsnorm_ffn_weights,
            embedding_tree,
            model_dir: dir.to_path_buf(),
            weight_hash,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Return the R_W weight-chain hash computed at load time.
    pub fn weight_hash(&self) -> [u8; 32] {
        self.weight_hash
    }

    /// Per-layer, per-matrix-type weight quantization scales.
    /// 0.0 for native INT8 tensors. Same layout as `VerifierKey.weight_scales`.
    pub fn weight_scales(&self) -> &[Vec<f32>] {
        &self.scales
    }

    pub fn rmsnorm_attn_weights(&self) -> &[Vec<f32>] {
        &self.rmsnorm_attn_weights
    }

    pub fn rmsnorm_ffn_weights(&self) -> &[Vec<f32>] {
        &self.rmsnorm_ffn_weights
    }

    pub fn rmsnorm_eps(&self) -> f64 {
        1e-5
    }

    /// Load a single embedding row (f32) for the given token ID.
    ///
    /// Re-opens the safetensors files to read one row on demand.
    pub fn load_embedding_row(&self, token_id: usize) -> Result<Vec<f32>> {
        let mapped = open_shards(&self.model_dir)?;
        let parsed: Vec<_> = mapped
            .iter()
            .map(|s| SafeTensors::deserialize(&s.mmap).map(|st| (st, s)))
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to parse safetensors")?;
        load_2d_row_f32(&parsed, "model.embed_tokens.weight", token_id, self.config.hidden_dim)
    }

    /// Generate a Merkle proof for the given token ID's embedding row.
    ///
    /// Returns `None` if the embedding tree was not built (tensor missing).
    pub fn embedding_proof(&self, token_id: usize) -> Option<verilm_core::merkle::MerkleProof> {
        self.embedding_tree.as_ref().map(|tree| verilm_core::merkle::prove(tree, token_id))
    }

    /// Embedding Merkle root, if available.
    pub fn embedding_merkle_root(&self) -> Option<[u8; 32]> {
        self.embedding_tree.as_ref().map(|tree| tree.root)
    }
}

impl ShellWeights for SafetensorsWeightProvider {
    fn weight(&self, layer: usize, mt: MatrixType) -> &[i8] {
        let mt_idx = MatrixType::ALL.iter().position(|&m| m == mt)
            .expect("unknown MatrixType");
        &self.weights[layer][mt_idx]
    }
}
