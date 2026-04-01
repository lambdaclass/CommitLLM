//! Model configurations and matrix type definitions.
//!
//! Each model config encodes the dimensions needed for Freivalds checks:
//! hidden dim, KV dim (under GQA), FFN intermediate dim, number of layers,
//! and head counts.

use serde::{Deserialize, Serialize};

/// Q8_0 block size: 32 quantized int8 values per block, with one f16 scale factor.
pub const Q8_0_BLOCK_SIZE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatrixType {
    Wq,
    Wk,
    Wv,
    Wo,
    Wg,
    Wu,
    Wd,
    LmHead,
}

impl MatrixType {
    /// The 7 per-layer matrices (one set per transformer layer).
    pub const PER_LAYER: [MatrixType; 7] = [
        MatrixType::Wq,
        MatrixType::Wk,
        MatrixType::Wv,
        MatrixType::Wo,
        MatrixType::Wg,
        MatrixType::Wu,
        MatrixType::Wd,
    ];

    /// All 8 matrices including the global LM-head.
    pub const ALL: [MatrixType; 8] = [
        MatrixType::Wq,
        MatrixType::Wk,
        MatrixType::Wv,
        MatrixType::Wo,
        MatrixType::Wg,
        MatrixType::Wu,
        MatrixType::Wd,
        MatrixType::LmHead,
    ];

    /// Output dimension m_j (the dimension of r_j).
    pub fn output_dim(&self, cfg: &ModelConfig) -> usize {
        match self {
            MatrixType::Wq => cfg.hidden_dim,
            MatrixType::Wk => cfg.kv_dim,
            MatrixType::Wv => cfg.kv_dim,
            MatrixType::Wo => cfg.hidden_dim,
            MatrixType::Wg => cfg.ffn_dim,
            MatrixType::Wu => cfg.ffn_dim,
            MatrixType::Wd => cfg.hidden_dim,
            MatrixType::LmHead => cfg.vocab_size,
        }
    }

    /// Input dimension n_j (the dimension of v_j = r_j^T W_j).
    pub fn input_dim(&self, cfg: &ModelConfig) -> usize {
        match self {
            MatrixType::Wq => cfg.hidden_dim,
            MatrixType::Wk => cfg.hidden_dim,
            MatrixType::Wv => cfg.hidden_dim,
            MatrixType::Wo => cfg.hidden_dim,
            MatrixType::Wg => cfg.hidden_dim,
            MatrixType::Wu => cfg.hidden_dim,
            MatrixType::Wd => cfg.ffn_dim,
            MatrixType::LmHead => cfg.hidden_dim,
        }
    }

    /// Safetensors weight name pattern for this matrix type.
    /// Layer index is substituted for `{}` (not applicable for LmHead).
    pub fn weight_name(&self) -> &'static str {
        match self {
            MatrixType::Wq => "model.layers.{}.self_attn.q_proj.weight",
            MatrixType::Wk => "model.layers.{}.self_attn.k_proj.weight",
            MatrixType::Wv => "model.layers.{}.self_attn.v_proj.weight",
            MatrixType::Wo => "model.layers.{}.self_attn.o_proj.weight",
            MatrixType::Wg => "model.layers.{}.mlp.gate_proj.weight",
            MatrixType::Wu => "model.layers.{}.mlp.up_proj.weight",
            MatrixType::Wd => "model.layers.{}.mlp.down_proj.weight",
            MatrixType::LmHead => "lm_head.weight",
        }
    }

    /// Safetensors bias name pattern for this projection.
    /// Only Q, K, V projections may have bias terms (model-dependent).
    /// Returns `None` for projections that never have bias.
    pub fn bias_name(&self) -> Option<&'static str> {
        match self {
            MatrixType::Wq => Some("model.layers.{}.self_attn.q_proj.bias"),
            MatrixType::Wk => Some("model.layers.{}.self_attn.k_proj.bias"),
            MatrixType::Wv => Some("model.layers.{}.self_attn.v_proj.bias"),
            _ => None,
        }
    }

    /// Safetensors per-channel weight scale name pattern.
    /// Present in W8A8 models with native INT8 weights. Shape = `[output_dim]`.
    pub fn weight_scale_name(&self) -> &'static str {
        match self {
            MatrixType::Wq => "model.layers.{}.self_attn.q_proj.weight_scale",
            MatrixType::Wk => "model.layers.{}.self_attn.k_proj.weight_scale",
            MatrixType::Wv => "model.layers.{}.self_attn.v_proj.weight_scale",
            MatrixType::Wo => "model.layers.{}.self_attn.o_proj.weight_scale",
            MatrixType::Wg => "model.layers.{}.mlp.gate_proj.weight_scale",
            MatrixType::Wu => "model.layers.{}.mlp.up_proj.weight_scale",
            MatrixType::Wd => "model.layers.{}.mlp.down_proj.weight_scale",
            MatrixType::LmHead => "lm_head.weight_scale",
        }
    }
}

/// RoPE scaling configuration for extended-context models.
///
/// Llama 3.1 uses `rope_type = "llama3"` with frequency-dependent scaling:
/// high-frequency dimensions (short wavelength) are unchanged, low-frequency
/// dimensions (long wavelength) are divided by `factor`, and medium-frequency
/// dimensions are smoothly interpolated between scaled and unscaled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RopeScaling {
    /// Scaling type, e.g. "llama3", "linear", "dynamic".
    pub rope_type: String,
    /// Position scaling factor (e.g. 8.0 for Llama 3.1).
    pub factor: f64,
    /// Low frequency factor for band boundary (default 1.0).
    pub low_freq_factor: f64,
    /// High frequency factor for band boundary (default 4.0).
    pub high_freq_factor: f64,
    /// Original training context length before scaling (e.g. 8192).
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub hidden_dim: usize,    // n
    pub kv_dim: usize,        // n_kv = n_kv_heads * d_head
    pub ffn_dim: usize,       // n_ffn
    pub d_head: usize,
    pub n_layers: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    /// Vocabulary size (number of logit entries).
    pub vocab_size: usize,
    /// RoPE base frequency (theta), e.g. 10000.0 for Llama-family models.
    pub rope_theta: f64,
    /// RoPE scaling for extended-context models (e.g. Llama 3.1).
    /// None for models with no position scaling (e.g. Qwen2.5).
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
}

impl ModelConfig {
    pub fn llama_70b() -> Self {
        ModelConfig {
            name: "Llama-3-70B".into(),
            hidden_dim: 8192,
            kv_dim: 1024,     // 8 KV heads * 128
            ffn_dim: 28672,
            d_head: 128,
            n_layers: 80,
            n_q_heads: 64,
            n_kv_heads: 8,
            vocab_size: 128256,
            rope_theta: 500000.0,
            rope_scaling: None,
        }
    }

    pub fn llama_8b() -> Self {
        ModelConfig {
            name: "Llama-3-8B".into(),
            hidden_dim: 4096,
            kv_dim: 1024,     // 8 KV heads * 128
            ffn_dim: 14336,
            d_head: 128,
            n_layers: 32,
            n_q_heads: 32,
            n_kv_heads: 8,
            vocab_size: 128256,
            rope_theta: 500000.0,
            rope_scaling: None,
        }
    }

    /// Llama 3.1 8B with rope_scaling (extended context to 128K).
    pub fn llama_3_1_8b() -> Self {
        ModelConfig {
            name: "Llama-3.1-8B".into(),
            hidden_dim: 4096,
            kv_dim: 1024,     // 8 KV heads * 128
            ffn_dim: 14336,
            d_head: 128,
            n_layers: 32,
            n_q_heads: 32,
            n_kv_heads: 8,
            vocab_size: 128256,
            rope_theta: 500000.0,
            rope_scaling: Some(RopeScaling {
                rope_type: "llama3".into(),
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192,
            }),
        }
    }

    pub fn llama_405b() -> Self {
        ModelConfig {
            name: "Llama-3-405B".into(),
            hidden_dim: 16384,
            kv_dim: 1024,     // 8 KV heads * 128
            ffn_dim: 53248,
            d_head: 128,
            n_layers: 126,
            n_q_heads: 128,
            n_kv_heads: 8,
            vocab_size: 128256,
            rope_theta: 500000.0,
            rope_scaling: None,
        }
    }

    /// Tiny model for testing. 2 layers, dim 16.
    pub fn toy() -> Self {
        ModelConfig {
            name: "toy".into(),
            hidden_dim: 16,
            kv_dim: 4,        // 2 KV heads * 2
            ffn_dim: 32,
            d_head: 2,
            n_layers: 2,
            n_q_heads: 8,
            n_kv_heads: 2,
            vocab_size: 64,
            rope_theta: 10000.0,
            rope_scaling: None,
        }
    }

    /// Compute scaled inverse frequencies for this config.
    ///
    /// Returns `d_head / 2` inverse frequency values. When `rope_scaling`
    /// is `None`, these are the standard RoPE frequencies. When present,
    /// frequencies are modified according to the scaling type.
    pub fn scaled_inv_freq(&self) -> Vec<f64> {
        let half = self.d_head / 2;
        let base_inv_freq: Vec<f64> = (0..half)
            .map(|k| 1.0 / self.rope_theta.powf((2 * k) as f64 / self.d_head as f64))
            .collect();

        let scaling = match &self.rope_scaling {
            Some(s) => s,
            None => return base_inv_freq,
        };

        match scaling.rope_type.as_str() {
            "llama3" => {
                let old_ctx = scaling.original_max_position_embeddings as f64;
                let low_freq_wavelen = old_ctx / scaling.low_freq_factor;
                let high_freq_wavelen = old_ctx / scaling.high_freq_factor;

                base_inv_freq
                    .iter()
                    .map(|&f| {
                        let wavelen = 2.0 * std::f64::consts::PI / f;
                        if wavelen < high_freq_wavelen {
                            // High frequency: no scaling
                            f
                        } else if wavelen > low_freq_wavelen {
                            // Low frequency: full scaling
                            f / scaling.factor
                        } else {
                            // Medium: smooth interpolation
                            let smooth = (old_ctx / wavelen - scaling.low_freq_factor)
                                / (scaling.high_freq_factor - scaling.low_freq_factor);
                            f * ((1.0 - smooth) / scaling.factor + smooth)
                        }
                    })
                    .collect()
            }
            "linear" => {
                // Linear scaling: divide all frequencies by factor
                base_inv_freq.iter().map(|&f| f / scaling.factor).collect()
            }
            _ => {
                // Unknown scaling type: fall back to unscaled
                base_inv_freq
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_dims_llama70b() {
        let cfg = ModelConfig::llama_70b();
        assert_eq!(MatrixType::Wq.output_dim(&cfg), 8192);
        assert_eq!(MatrixType::Wq.input_dim(&cfg), 8192);
        assert_eq!(MatrixType::Wk.output_dim(&cfg), 1024);
        assert_eq!(MatrixType::Wk.input_dim(&cfg), 8192);
        assert_eq!(MatrixType::Wg.output_dim(&cfg), 28672);
        assert_eq!(MatrixType::Wg.input_dim(&cfg), 8192);
        assert_eq!(MatrixType::Wd.output_dim(&cfg), 8192);
        assert_eq!(MatrixType::Wd.input_dim(&cfg), 28672);
    }

    #[test]
    fn test_all_matrix_types() {
        assert_eq!(MatrixType::ALL.len(), 8);
        assert_eq!(MatrixType::PER_LAYER.len(), 7);
    }
}
