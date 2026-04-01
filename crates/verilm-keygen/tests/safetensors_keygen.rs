//! Tests for keygen from safetensors files.

use tempfile::TempDir;
use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::freivalds;
use verilm_core::serialize;

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
        for mt in &MatrixType::PER_LAYER {
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
    verilm_keygen::write_safetensors(&path, &tensor_refs).unwrap();

    (dir, cfg, all_weights)
}

#[test]
fn test_detect_config() {
    let (dir, cfg, _) = make_toy_safetensors();
    let detected = verilm_keygen::detect_config(dir.path()).unwrap();

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
    let key = verilm_keygen::generate_key(dir.path(), seed).unwrap();

    assert_eq!(key.version, 1);
    assert_eq!(key.r_vectors.len(), 8);
    assert_eq!(key.v_vectors.len(), cfg.n_layers);

    // Check r vector dimensions
    for (j, mt) in MatrixType::PER_LAYER.iter().enumerate() {
        assert_eq!(key.r_vectors[j].len(), mt.output_dim(&cfg));
    }

    // Check v vector dimensions
    for layer in &key.v_vectors {
        assert_eq!(layer.len(), 7);
        for (j, mt) in MatrixType::PER_LAYER.iter().enumerate() {
            assert_eq!(layer[j].len(), mt.input_dim(&cfg));
        }
    }
}

#[test]
fn test_precomputed_v_matches_manual() {
    let (dir, cfg, all_weights) = make_toy_safetensors();
    let seed = [42u8; 32];
    let key = verilm_keygen::generate_key(dir.path(), seed).unwrap();

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
    let key = verilm_keygen::generate_key(dir.path(), seed).unwrap();

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
    let key = verilm_keygen::generate_key(dir.path(), seed).unwrap();

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
    let key = verilm_keygen::generate_key(dir.path(), seed).unwrap();

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

    let key1 = verilm_keygen::generate_key(dir.path(), seed).unwrap();
    let key2 = verilm_keygen::generate_key(dir.path(), seed).unwrap();

    assert_eq!(key1.r_vectors, key2.r_vectors);
    assert_eq!(key1.v_vectors, key2.v_vectors);
}

#[test]
fn test_different_seed_different_key() {
    let (dir, _, _) = make_toy_safetensors();

    let key1 = verilm_keygen::generate_key(dir.path(), [1u8; 32]).unwrap();
    let key2 = verilm_keygen::generate_key(dir.path(), [2u8; 32]).unwrap();

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
        for mt in &MatrixType::PER_LAYER {
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
    let key = verilm_keygen::generate_key(dir.path(), seed).unwrap();

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
    assert_eq!(key.r_vectors.len(), 8);
    assert_eq!(key.v_vectors.len(), cfg.n_layers);
}

#[test]
fn test_bf16_keygen_deterministic() {
    let (dir, _) = make_bf16_safetensors();

    let key1 = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();
    let key2 = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();

    assert_eq!(key1.r_vectors, key2.r_vectors);
    assert_eq!(key1.v_vectors, key2.v_vectors);
    assert_eq!(key1.quantization_scales, key2.quantization_scales);
}

#[test]
fn test_bf16_key_stores_quantization_metadata() {
    let (dir, _) = make_bf16_safetensors();
    let key = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();

    // Serialize and deserialize — metadata should survive
    let data = serialize::serialize_key(&key);
    let key2 = serialize::deserialize_key(&data).unwrap();

    assert_eq!(key2.source_dtype, "BF16");
    assert_eq!(key2.quantization_scales, key.quantization_scales);
}

#[test]
fn test_int8_key_has_zero_quantization_scales() {
    let (dir, _cfg, _) = make_toy_safetensors();
    let key = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();

    assert_eq!(key.source_dtype, "I8");
    // INT8 weights have scale=0.0 (no quantization applied)
    for layer_scales in &key.quantization_scales {
        for &scale in layer_scales {
            assert_eq!(scale, 0.0, "INT8 weights should have zero scale");
        }
    }
}

// -----------------------------------------------------------------------
// SafetensorsWeightProvider tests
// -----------------------------------------------------------------------

#[test]
fn test_weight_provider_loads_and_matches() {
    use verilm_core::types::ShellWeights;

    let (dir, cfg, all_weights) = make_toy_safetensors();
    let provider = verilm_keygen::SafetensorsWeightProvider::load(dir.path()).unwrap();

    assert_eq!(provider.config().hidden_dim, cfg.hidden_dim);
    assert_eq!(provider.config().n_layers, cfg.n_layers);

    // Every weight matrix from the provider must match the original
    for (layer_idx, layer_weights) in all_weights.iter().enumerate() {
        for (mt, expected) in layer_weights {
            let got = provider.weight(layer_idx, *mt);
            assert_eq!(got, expected.as_slice(),
                "weight mismatch at layer {} {:?}", layer_idx, mt);
        }
    }
}

#[test]
fn test_weight_provider_freivalds_compatible() {
    use verilm_core::types::ShellWeights;

    let (dir, cfg, _) = make_toy_safetensors();
    let key = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();
    let provider = verilm_keygen::SafetensorsWeightProvider::load(dir.path()).unwrap();

    // Shell opening from the provider must pass Freivalds with the keygen key
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(5555);

    for layer_idx in 0..cfg.n_layers {
        for mt in MatrixType::PER_LAYER.iter() {
            let cols = mt.input_dim(&cfg);
            let rows = mt.output_dim(&cfg);
            let x: Vec<i8> = (0..cols).map(|_| rng.gen::<i8>()).collect();
            let w = provider.weight(layer_idx, *mt);

            let z: Vec<i32> = (0..rows)
                .map(|r| (0..cols).map(|c| w[r * cols + c] as i32 * x[c] as i32).sum())
                .collect();

            assert!(
                freivalds::check(
                    key.v_for(layer_idx, *mt),
                    &x,
                    key.r_for(*mt),
                    &z,
                ),
                "Freivalds failed: layer {} {:?}", layer_idx, mt
            );
        }
    }
}

// -----------------------------------------------------------------------
// W8A8 (INT8 weights + per-channel F32 weight_scale) keygen tests
// -----------------------------------------------------------------------

/// Build a synthetic W8A8 safetensors model: INT8 weight matrices plus
/// per-channel F32 weight_scale tensors (shape [output_dim]).
fn make_w8a8_safetensors() -> (TempDir, ModelConfig) {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use verilm_keygen::TypedTensor;

    let cfg = ModelConfig::toy();
    let dir = TempDir::new().unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(54321);

    struct OwnedTensor {
        name: String,
        shape: Vec<usize>,
        i8_data: Option<Vec<i8>>,
        f32_data: Option<Vec<f32>>,
    }

    let mut owned: Vec<OwnedTensor> = Vec::new();

    for layer in 0..cfg.n_layers {
        for mt in &MatrixType::PER_LAYER {
            let rows = mt.output_dim(&cfg);
            let cols = mt.input_dim(&cfg);

            // INT8 weight matrix
            let w: Vec<i8> = (0..rows * cols).map(|_| rng.gen::<i8>()).collect();
            let weight_name = mt.weight_name().replace("{}", &layer.to_string());
            owned.push(OwnedTensor {
                name: weight_name,
                shape: vec![rows, cols],
                i8_data: Some(w),
                f32_data: None,
            });

            // Per-channel weight_scale (shape [output_dim])
            let scales: Vec<f32> = (0..rows)
                .map(|_| rng.gen::<f32>() * 0.01 + 0.001) // small positive scales
                .collect();
            let scale_name = mt.weight_scale_name().replace("{}", &layer.to_string());
            owned.push(OwnedTensor {
                name: scale_name,
                shape: vec![rows],
                i8_data: None,
                f32_data: Some(scales),
            });
        }
    }

    let typed_tensors: Vec<(&str, Vec<usize>, TypedTensor<'_>)> = owned
        .iter()
        .map(|t| {
            let data = if let Some(ref d) = t.i8_data {
                TypedTensor::I8(d.as_slice())
            } else {
                TypedTensor::F32(t.f32_data.as_ref().unwrap().as_slice())
            };
            (t.name.as_str(), t.shape.clone(), data)
        })
        .collect();

    let path = dir.path().join("model.safetensors");
    verilm_keygen::write_safetensors_mixed(&path, &typed_tensors).unwrap();

    (dir, cfg)
}

fn write_model_config_json(dir: &TempDir, model_type: &str, rope_theta: f64) {
    let data = format!(
        "{{\"model_type\":\"{}\",\"rms_norm_eps\":1e-5,\"rope_theta\":{}}}",
        model_type, rope_theta
    );
    std::fs::write(dir.path().join("config.json"), data).unwrap();
}

fn write_llama3_config_json(dir: &TempDir) {
    let data = r#"{
        "model_type": "llama",
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        }
    }"#;
    std::fs::write(dir.path().join("config.json"), data).unwrap();
}

#[test]
fn test_w8a8_keygen_detects_per_channel_scales() {
    let (dir, cfg) = make_w8a8_safetensors();
    let key = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();

    assert_eq!(key.source_dtype, "I8");
    assert_eq!(key.quant_family.as_deref(), Some("W8A8"));
    assert_eq!(key.scale_derivation.as_deref(), Some("per_channel_absmax"));
    assert!(key.has_per_channel_scales());

    // per_channel_weight_scales shape: [n_layers][7][output_dim]
    assert_eq!(key.per_channel_weight_scales.len(), cfg.n_layers);
    for (layer_idx, layer_scales) in key.per_channel_weight_scales.iter().enumerate() {
        assert_eq!(layer_scales.len(), 7, "layer {} should have 7 matrices", layer_idx);
        for (j, mt) in MatrixType::PER_LAYER.iter().enumerate() {
            let expected_dim = mt.output_dim(&cfg);
            assert_eq!(
                layer_scales[j].len(),
                expected_dim,
                "layer {} {:?} scale length mismatch",
                layer_idx, mt
            );
            for &s in &layer_scales[j] {
                assert!(s > 0.0, "layer {} {:?} has non-positive scale", layer_idx, mt);
            }
        }
    }
}

#[test]
fn test_w8a8_keygen_legacy_scales_zero() {
    let (dir, _cfg) = make_w8a8_safetensors();
    let key = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();

    for layer_scales in &key.quantization_scales {
        for &scale in layer_scales {
            assert_eq!(scale, 0.0, "legacy per-tensor scale should be 0.0 for W8A8");
        }
    }
}

#[test]
fn test_w8a8_key_roundtrip() {
    let (dir, _cfg) = make_w8a8_safetensors();
    let key = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();

    let data = serialize::serialize_key(&key);
    let key2 = serialize::deserialize_key(&data).unwrap();

    assert_eq!(key2.quant_family.as_deref(), Some("W8A8"));
    assert_eq!(key2.scale_derivation.as_deref(), Some("per_channel_absmax"));
    assert_eq!(key2.per_channel_weight_scales, key.per_channel_weight_scales);
    assert!(key2.has_per_channel_scales());
}

#[test]
fn test_w8a8_qwen_profile_detected_and_roundtrips() {
    let (dir, _cfg) = make_w8a8_safetensors();
    write_model_config_json(&dir, "qwen2", 1_000_000.0);

    let key = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();
    let profile = key.verification_profile.as_ref().expect("expected qwen profile");
    assert_eq!(profile.name, "qwen-w8a8");
    assert_eq!(profile.model_family, "qwen2");
    assert_eq!(profile.bridge_tolerance, 1);
    assert_eq!(profile.attention_tolerance, 10);
    assert_eq!(profile.max_validated_context, 1164);
    assert!(!profile.requires_score_anchoring);

    let data = serialize::serialize_key(&key);
    let key2 = serialize::deserialize_key(&data).unwrap();
    assert_eq!(key2.verification_profile, key.verification_profile);
}

#[test]
fn test_w8a8_llama_profile_detected() {
    let (dir, _cfg) = make_w8a8_safetensors();
    write_model_config_json(&dir, "llama", 500_000.0);

    let key = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();
    let profile = key.verification_profile.as_ref().expect("expected llama profile");
    assert_eq!(profile.name, "llama-w8a8");
    assert_eq!(profile.model_family, "llama");
    assert_eq!(profile.bridge_tolerance, 1);
    assert_eq!(profile.attention_tolerance, 10);
    assert_eq!(profile.max_validated_context, 1165);
    assert!(!profile.requires_score_anchoring);
}

#[test]
fn test_w8a8_freivalds_still_works() {
    let (dir, cfg) = make_w8a8_safetensors();
    let key = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();

    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(3333);

    use verilm_core::types::ShellWeights;
    let provider = verilm_keygen::SafetensorsWeightProvider::load(dir.path()).unwrap();

    for layer_idx in 0..cfg.n_layers {
        for mt in MatrixType::PER_LAYER.iter() {
            let cols = mt.input_dim(&cfg);
            let rows = mt.output_dim(&cfg);
            let x: Vec<i8> = (0..cols).map(|_| rng.gen::<i8>()).collect();
            let w = provider.weight(layer_idx, *mt);

            let z: Vec<i32> = (0..rows)
                .map(|r| (0..cols).map(|c| w[r * cols + c] as i32 * x[c] as i32).sum())
                .collect();

            assert!(
                freivalds::check(key.v_for(layer_idx, *mt), &x, key.r_for(*mt), &z),
                "Freivalds failed: layer {} {:?}", layer_idx, mt
            );
        }
    }
}

#[test]
fn test_llama3_rope_scaling_stored_in_key() {
    let (dir, _cfg) = make_w8a8_safetensors();
    write_llama3_config_json(&dir);

    let key = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();
    let scaling = key.config.rope_scaling.as_ref().expect("expected rope_scaling");
    assert_eq!(scaling.rope_type, "llama3");
    assert_eq!(scaling.factor, 8.0);
    assert_eq!(scaling.low_freq_factor, 1.0);
    assert_eq!(scaling.high_freq_factor, 4.0);
    assert_eq!(scaling.original_max_position_embeddings, 8192);
    assert_eq!(key.config.rope_theta, 500000.0);

    // Verify it roundtrips through serialization
    let data = serialize::serialize_key(&key);
    let key2 = serialize::deserialize_key(&data).unwrap();
    assert_eq!(key2.config.rope_scaling, key.config.rope_scaling);
}

#[test]
fn test_no_rope_scaling_when_absent() {
    let (dir, _cfg, _) = make_toy_safetensors();
    // No config.json → no rope_scaling
    let key = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();
    assert!(key.config.rope_scaling.is_none());
}

#[test]
fn test_plain_int8_has_no_per_channel_scales() {
    let (dir, _cfg, _) = make_toy_safetensors();
    let key = verilm_keygen::generate_key(dir.path(), [42u8; 32]).unwrap();

    assert!(!key.has_per_channel_scales());
    assert_eq!(key.quant_family.as_deref(), Some("INT8"));
    assert!(key.scale_derivation.is_none());
}
