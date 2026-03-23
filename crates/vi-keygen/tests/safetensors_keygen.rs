//! Tests for keygen from safetensors files.

use tempfile::TempDir;
use vi_core::constants::{MatrixType, ModelConfig};
use vi_core::freivalds;
use vi_core::serialize;

/// Build a synthetic safetensors model directory with the toy config.
/// Returns (temp_dir, config, all_weights_by_layer).
fn make_toy_safetensors() -> (TempDir, ModelConfig, Vec<Vec<(MatrixType, Vec<i8>)>>) {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    let cfg = ModelConfig::toy();
    let dir = TempDir::new().unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(12345);

    let mut all_weights = Vec::new();
    let mut tensors: Vec<(String, Vec<usize>, Vec<i8>)> = Vec::new();

    for layer in 0..cfg.n_layers {
        let mut layer_weights = Vec::new();
        for mt in &MatrixType::ALL {
            let rows = mt.output_dim(&cfg);
            let cols = mt.input_dim(&cfg);
            let w: Vec<i8> = (0..rows * cols).map(|_| rng.gen::<i8>()).collect();
            let name = mt.weight_name().replace("{}", &layer.to_string());
            tensors.push((name, vec![rows, cols], w.clone()));
            layer_weights.push((*mt, w));
        }
        all_weights.push(layer_weights);
    }

    // Convert to the format write_safetensors expects
    let tensor_refs: Vec<(&str, Vec<usize>, &[i8])> = tensors
        .iter()
        .map(|(name, shape, data)| (name.as_str(), shape.clone(), data.as_slice()))
        .collect();

    let path = dir.path().join("model.safetensors");
    vi_keygen::write_safetensors(&path, &tensor_refs).unwrap();

    (dir, cfg, all_weights)
}

#[test]
fn test_detect_config() {
    let (dir, cfg, _) = make_toy_safetensors();
    let detected = vi_keygen::detect_config(dir.path()).unwrap();

    // These are derived directly from tensor shapes — always correct
    assert_eq!(detected.hidden_dim, cfg.hidden_dim);
    assert_eq!(detected.kv_dim, cfg.kv_dim);
    assert_eq!(detected.ffn_dim, cfg.ffn_dim);
    assert_eq!(detected.n_layers, cfg.n_layers);

    // d_head / head counts are heuristic from shapes alone.
    // For real Llama (d_head=128), detection is exact.
    // For toy configs the head split may differ, but hidden_dim/kv_dim
    // are what matter for Freivalds (they determine matrix dimensions).
    assert_eq!(detected.hidden_dim, detected.n_q_heads * detected.d_head);
    assert_eq!(detected.kv_dim, detected.n_kv_heads * detected.d_head);
}

#[test]
fn test_generate_key_dimensions() {
    let (dir, cfg, _) = make_toy_safetensors();
    let seed = [42u8; 32];
    let key = vi_keygen::generate_key(dir.path(), seed).unwrap();

    assert_eq!(key.version, 1);
    assert_eq!(key.r_vectors.len(), 7);
    assert_eq!(key.v_vectors.len(), cfg.n_layers);

    // Check r vector dimensions
    for (j, mt) in MatrixType::ALL.iter().enumerate() {
        assert_eq!(key.r_vectors[j].len(), mt.output_dim(&cfg));
    }

    // Check v vector dimensions
    for layer in &key.v_vectors {
        assert_eq!(layer.len(), 7);
        for (j, mt) in MatrixType::ALL.iter().enumerate() {
            assert_eq!(layer[j].len(), mt.input_dim(&cfg));
        }
    }
}

#[test]
fn test_precomputed_v_matches_manual() {
    let (dir, cfg, all_weights) = make_toy_safetensors();
    let seed = [42u8; 32];
    let key = vi_keygen::generate_key(dir.path(), seed).unwrap();

    // For each layer and matrix type, verify v_j = r_j^T W_j
    for (layer_idx, layer_weights) in all_weights.iter().enumerate() {
        for (j, (mt, w)) in layer_weights.iter().enumerate() {
            let r = &key.r_vectors[j];
            let rows = mt.output_dim(&cfg);
            let cols = mt.input_dim(&cfg);

            // Manually compute v = r^T W
            let v_manual = freivalds::precompute_v(r, w, rows, cols);
            let v_key = &key.v_vectors[layer_idx][j];

            assert_eq!(
                v_manual, *v_key,
                "v mismatch at layer {} {:?}",
                layer_idx, mt
            );
        }
    }
}

#[test]
fn test_freivalds_check_with_safetensors_key() {
    let (dir, cfg, all_weights) = make_toy_safetensors();
    let seed = [42u8; 32];
    let key = vi_keygen::generate_key(dir.path(), seed).unwrap();

    // Pick a random input and verify the full Freivalds check works
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(9999);

    for (layer_idx, layer_weights) in all_weights.iter().enumerate() {
        for (j, (mt, w)) in layer_weights.iter().enumerate() {
            let cols = mt.input_dim(&cfg);
            let rows = mt.output_dim(&cfg);

            // Random INT8 input
            let x: Vec<i8> = (0..cols).map(|_| rng.gen::<i8>()).collect();

            // Compute true output z = W * x (i32 accumulators)
            let z: Vec<i32> = (0..rows)
                .map(|r_idx| {
                    (0..cols)
                        .map(|c| w[r_idx * cols + c] as i32 * x[c] as i32)
                        .sum()
                })
                .collect();

            let v = &key.v_vectors[layer_idx][j];
            let r = &key.r_vectors[j];
            assert!(
                freivalds::check(v, &x, r, &z),
                "Freivalds check failed at layer {} {:?}",
                layer_idx, mt
            );
        }
    }
}

#[test]
fn test_corrupted_weight_fails_freivalds() {
    let (dir, cfg, all_weights) = make_toy_safetensors();
    let seed = [42u8; 32];
    let key = vi_keygen::generate_key(dir.path(), seed).unwrap();

    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(7777);

    // Use the correct key but compute z with corrupted weights
    let layer_idx = 0;
    let j = 0; // Wq
    let mt = MatrixType::Wq;
    let (_, w) = &all_weights[layer_idx][j];

    let cols = mt.input_dim(&cfg);
    let rows = mt.output_dim(&cfg);

    // Random input
    let x: Vec<i8> = (0..cols).map(|_| rng.gen::<i8>()).collect();

    // Corrupt one weight
    let mut w_bad = w.clone();
    w_bad[0] = w_bad[0].wrapping_add(1);

    // Compute z with corrupted weights
    let z_bad: Vec<i32> = (0..rows)
        .map(|r_idx| {
            (0..cols)
                .map(|c| w_bad[r_idx * cols + c] as i32 * x[c] as i32)
                .sum()
        })
        .collect();

    let v = &key.v_vectors[layer_idx][j];
    let r = &key.r_vectors[j];

    // This should almost certainly fail (prob 1 - 1/p ≈ 1)
    assert!(
        !freivalds::check(v, &x, r, &z_bad),
        "Freivalds should reject corrupted weights"
    );
}

#[test]
fn test_key_serialization_roundtrip() {
    let (dir, _, _) = make_toy_safetensors();
    let seed = [42u8; 32];
    let key = vi_keygen::generate_key(dir.path(), seed).unwrap();

    let data = serialize::serialize_key(&key);
    let key2 = serialize::deserialize_key(&data).unwrap();

    assert_eq!(key.version, key2.version);
    assert_eq!(key.seed, key2.seed);
    assert_eq!(key.r_vectors, key2.r_vectors);
    assert_eq!(key.v_vectors, key2.v_vectors);
    assert_eq!(key.config.hidden_dim, key2.config.hidden_dim);
    assert_eq!(key.config.n_layers, key2.config.n_layers);
}

#[test]
fn test_deterministic_seed() {
    let (dir, _, _) = make_toy_safetensors();
    let seed = [42u8; 32];

    let key1 = vi_keygen::generate_key(dir.path(), seed).unwrap();
    let key2 = vi_keygen::generate_key(dir.path(), seed).unwrap();

    assert_eq!(key1.r_vectors, key2.r_vectors);
    assert_eq!(key1.v_vectors, key2.v_vectors);
}

#[test]
fn test_different_seed_different_key() {
    let (dir, _, _) = make_toy_safetensors();

    let key1 = vi_keygen::generate_key(dir.path(), [1u8; 32]).unwrap();
    let key2 = vi_keygen::generate_key(dir.path(), [2u8; 32]).unwrap();

    assert_ne!(key1.r_vectors, key2.r_vectors);
}

// -----------------------------------------------------------------------
// BF16 keygen tests
// -----------------------------------------------------------------------

/// Build a BF16 safetensors model by converting the toy INT8 weights to BF16.
fn make_bf16_safetensors() -> (tempfile::TempDir, ModelConfig) {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    let cfg = ModelConfig::toy();
    let dir = tempfile::TempDir::new().unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(12345);

    // Build tensors as f32, then convert to BF16 bytes
    let mut tensor_names = Vec::new();
    let mut tensor_shapes = Vec::new();
    let mut tensor_data = Vec::new(); // BF16 bytes

    for layer in 0..cfg.n_layers {
        for mt in &MatrixType::ALL {
            let rows = mt.output_dim(&cfg);
            let cols = mt.input_dim(&cfg);
            let name = mt.weight_name().replace("{}", &layer.to_string());

            // Random f32 weights in a realistic range
            let f32_vals: Vec<f32> = (0..rows * cols)
                .map(|_| rng.gen::<f32>() * 0.02 - 0.01) // [-0.01, 0.01]
                .collect();

            // Convert to BF16 bytes
            let bf16_bytes: Vec<u8> = f32_vals
                .iter()
                .flat_map(|&v| {
                    let bits = v.to_bits();
                    let bf16 = (bits >> 16) as u16;
                    bf16.to_le_bytes().to_vec()
                })
                .collect();

            tensor_names.push(name);
            tensor_shapes.push(vec![rows, cols]);
            tensor_data.push(bf16_bytes);
        }
    }

    // Write using safetensors serialize API
    let views: Vec<_> = tensor_names
        .iter()
        .zip(tensor_shapes.iter())
        .zip(tensor_data.iter())
        .map(|((name, shape), data)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BF16,
                    shape.clone(),
                    data,
                )
                .unwrap(),
            )
        })
        .collect();

    let bytes = safetensors::tensor::serialize(views, &None).unwrap();
    std::fs::write(dir.path().join("model.safetensors"), bytes).unwrap();

    (dir, cfg)
}

#[test]
fn test_bf16_keygen_produces_valid_key() {
    let (dir, cfg) = make_bf16_safetensors();
    let seed = [42u8; 32];
    let key = vi_keygen::generate_key(dir.path(), seed).unwrap();

    assert_eq!(key.source_dtype, "BF16");
    assert_eq!(key.quantization_scales.len(), cfg.n_layers);
    assert_eq!(key.quantization_scales[0].len(), 7);

    // All scales should be positive (non-zero weights)
    for layer_scales in &key.quantization_scales {
        for &scale in layer_scales {
            assert!(scale > 0.0, "quantization scale should be positive");
        }
    }

    // Dimensions should be correct
    assert_eq!(key.r_vectors.len(), 7);
    assert_eq!(key.v_vectors.len(), cfg.n_layers);
}

#[test]
fn test_bf16_keygen_deterministic() {
    let (dir, _) = make_bf16_safetensors();

    let key1 = vi_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();
    let key2 = vi_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();

    assert_eq!(key1.r_vectors, key2.r_vectors);
    assert_eq!(key1.v_vectors, key2.v_vectors);
    assert_eq!(key1.quantization_scales, key2.quantization_scales);
}

#[test]
fn test_bf16_key_stores_quantization_metadata() {
    let (dir, _) = make_bf16_safetensors();
    let key = vi_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();

    // Serialize and deserialize — metadata should survive
    let data = serialize::serialize_key(&key);
    let key2 = serialize::deserialize_key(&data).unwrap();

    assert_eq!(key2.source_dtype, "BF16");
    assert_eq!(key2.quantization_scales, key.quantization_scales);
}

#[test]
fn test_int8_key_has_zero_quantization_scales() {
    let (dir, _cfg, _) = make_toy_safetensors();
    let key = vi_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();

    assert_eq!(key.source_dtype, "I8");
    // INT8 weights have scale=0.0 (no quantization applied)
    for layer_scales in &key.quantization_scales {
        for &scale in layer_scales {
            assert_eq!(scale, 0.0, "INT8 weights should have zero scale");
        }
    }
}
