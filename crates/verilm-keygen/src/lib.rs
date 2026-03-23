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
use verilm_core::types::VerifierKey;

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

    Ok(VerifierKey {
        version: 1,
        config: cfg,
        seed,
        source_dtype,
        quantization_scales,
        r_vectors,
        v_vectors,
        wo_norms: Vec::new(),
        max_v_norm: 0.0,
        lm_head: None,
        weight_hash: Some(weight_hash),
    })
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
