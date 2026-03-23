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
}

impl MatrixType {
    pub const ALL: [MatrixType; 7] = [
        MatrixType::Wq,
        MatrixType::Wk,
        MatrixType::Wv,
        MatrixType::Wo,
        MatrixType::Wg,
        MatrixType::Wu,
        MatrixType::Wd,
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
        }
    }

    /// Safetensors weight name pattern for this matrix type.
    /// Layer index is substituted for `{}`.
    pub fn weight_name(&self) -> &'static str {
        match self {
            MatrixType::Wq => "model.layers.{}.self_attn.q_proj.weight",
            MatrixType::Wk => "model.layers.{}.self_attn.k_proj.weight",
            MatrixType::Wv => "model.layers.{}.self_attn.v_proj.weight",
            MatrixType::Wo => "model.layers.{}.self_attn.o_proj.weight",
            MatrixType::Wg => "model.layers.{}.mlp.gate_proj.weight",
            MatrixType::Wu => "model.layers.{}.mlp.up_proj.weight",
            MatrixType::Wd => "model.layers.{}.mlp.down_proj.weight",
        }
    }
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
    /// Vocabulary size (number of logit entries). 0 if not applicable.
    #[serde(default)]
    pub vocab_size: usize,
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
        assert_eq!(MatrixType::ALL.len(), 7);
    }
}
