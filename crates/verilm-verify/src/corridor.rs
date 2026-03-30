//! Attention corridor measurement.
//!
//! Measures the L-inf (and full histogram) difference between the GPU's
//! actual attention output `a_i8` and the verifier's replayed `a_i8`.
//! This is a measurement tool — no pass/fail threshold. The output
//! informs what tolerance tau should be.
//!
//! Mirrors the replay logic from `canonical::replay_deep_prefix_roped`
//! but calls [`verilm_core::attention::measure_attention_diff`] instead
//! of [`verilm_core::attention::compare_attention_output`].

use verilm_core::attention::{measure_attention_diff, AttentionDiffStats};
use verilm_core::constants::MatrixType;
use verilm_core::types::{V4AuditResponse, VerifierKey};

/// Per-channel weight scales for faithful corridor measurement.
///
/// W8A8 models have per-channel (per output feature) weight scales, but
/// the VerifierKey stores `0.0` for native INT8 weights — zeroing all
/// dequantized values and producing meaningless L-inf=127 results.
///
/// This struct provides the model's actual per-channel weight scales,
/// extracted at measurement time from the served model. It is used
/// only in the corridor/debug path, not in the verification protocol.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct CorridorScaleOverrides {
    /// Per-channel weight scales for Q projection, per layer.
    /// `wq[layer]` has length `hidden_dim` (n_q_heads * d_head).
    pub wq: Vec<Vec<f32>>,
    /// Per-channel weight scales for K projection, per layer.
    /// `wk[layer]` has length `kv_dim` (n_kv_heads * d_head).
    pub wk: Vec<Vec<f32>>,
    /// Per-channel weight scales for V projection, per layer.
    /// `wv[layer]` has length `kv_dim` (n_kv_heads * d_head).
    pub wv: Vec<Vec<f32>>,
}

/// Aggregated corridor measurement across all layers and token positions.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CorridorReport {
    pub measurements: Vec<AttentionDiffStats>,
    pub global_linf: i16,
    pub per_layer_max_linf: Vec<i16>,
    pub per_position_max_linf: Vec<i16>,
}

/// Measure the attention corridor for a deep-prefix audit response.
///
/// Replays attention from the QKV accumulators in the prefix shell openings
/// and measures the diff against the retained `a` vectors. Supports both
/// toy (raw requantize) and production (dequant + RoPE + f64) paths based
/// on `key.rope_aware_replay`.
///
/// When `scale_overrides` is provided, uses per-channel weight scales for
/// dequantization instead of the VerifierKey's per-tensor scales. This is
/// required for faithful measurement on W8A8 models.
///
/// Returns `Err` if prefix data is missing or malformed.
pub fn measure_corridor(
    key: &VerifierKey,
    response: &V4AuditResponse,
    scale_overrides: Option<&CorridorScaleOverrides>,
) -> Result<CorridorReport, String> {
    // Fail closed: require all three data sources.
    let prefix_ret = response
        .prefix_retained
        .as_ref()
        .ok_or("no prefix_retained in audit response")?;
    let prefix_shells = response
        .prefix_shell_openings
        .as_ref()
        .ok_or("no prefix_shell_openings in audit response")?;
    if response.shell_opening.is_none() {
        return Err("no shell_opening for opened token in audit response".into());
    }

    if prefix_shells.is_empty() {
        return Err("empty prefix_shell_openings".into());
    }

    let cfg = &key.config;
    let n_layers = cfg.n_layers.min(
        prefix_shells
            .iter()
            .map(|s| s.layers.len())
            .min()
            .unwrap_or(0),
    );

    let measurements = if key.rope_aware_replay {
        measure_roped(key, prefix_ret, prefix_shells, response, n_layers, scale_overrides)?
    } else {
        measure_toy(key, prefix_ret, prefix_shells, response, n_layers)?
    };

    build_report(measurements, n_layers)
}

fn measure_toy(
    key: &VerifierKey,
    prefix_ret: &[verilm_core::types::RetainedTokenState],
    prefix_shells: &[verilm_core::types::ShellTokenOpening],
    response: &V4AuditResponse,
    n_layers: usize,
) -> Result<Vec<AttentionDiffStats>, String> {
    let cfg = &key.config;
    let mut all_stats = Vec::new();

    for layer_idx in 0..n_layers {
        let mut kv_k: Vec<Vec<i8>> = Vec::new();
        let mut kv_v: Vec<Vec<i8>> = Vec::new();

        for (j, (shell_j, ret_j)) in prefix_shells.iter().zip(prefix_ret.iter()).enumerate() {
            if layer_idx >= ret_j.layers.len() {
                break;
            }
            let sl = &shell_j.layers[layer_idx];
            let rs = &ret_j.layers[layer_idx];
            let (q_acc, k_acc, v_acc) = match (&sl.q, &sl.k, &sl.v) {
                (Some(q), Some(k), Some(v)) => (q, k, v),
                _ => break,
            };

            kv_k.push(verilm_core::requantize(k_acc));
            kv_v.push(verilm_core::requantize(v_acc));

            if j == 0 {
                continue;
            }

            let q_i8 = verilm_core::requantize(q_acc);
            let replayed = verilm_core::attention::replay_attention_reference(
                &q_i8, &kv_k, &kv_v, cfg,
            );
            all_stats.push(measure_attention_diff(&rs.a, &replayed, layer_idx, j)?);
        }

        // Opened token
        if let Some((q_acc, k_acc, v_acc, rs)) =
            opened_token_qkv(response, layer_idx)
        {
            kv_k.push(verilm_core::requantize(k_acc));
            kv_v.push(verilm_core::requantize(v_acc));
            let q_i8 = verilm_core::requantize(q_acc);
            let replayed = verilm_core::attention::replay_attention_reference(
                &q_i8, &kv_k, &kv_v, cfg,
            );
            let pos = response.token_index as usize;
            all_stats.push(measure_attention_diff(&rs.a, &replayed, layer_idx, pos)?);
        }
    }

    Ok(all_stats)
}

fn measure_roped(
    key: &VerifierKey,
    prefix_ret: &[verilm_core::types::RetainedTokenState],
    prefix_shells: &[verilm_core::types::ShellTokenOpening],
    response: &V4AuditResponse,
    n_layers: usize,
    scale_overrides: Option<&CorridorScaleOverrides>,
) -> Result<Vec<AttentionDiffStats>, String> {
    let cfg = &key.config;
    let mut all_stats = Vec::new();

    for layer_idx in 0..n_layers {
        let mut kv_k: Vec<Vec<f64>> = Vec::new();
        let mut kv_v: Vec<Vec<f64>> = Vec::new();

        for (j, (shell_j, ret_j)) in prefix_shells.iter().zip(prefix_ret.iter()).enumerate() {
            if layer_idx >= ret_j.layers.len() {
                break;
            }
            let sl = &shell_j.layers[layer_idx];
            let rs = &ret_j.layers[layer_idx];
            let (q_acc, k_acc, v_acc) = match (&sl.q, &sl.k, &sl.v) {
                (Some(q), Some(k), Some(v)) => (q, k, v),
                _ => break,
            };

            let (_, k_roped, v_deq) =
                dequant_rope_qkv(key, layer_idx, q_acc, k_acc, v_acc, sl.scale_x_attn, j, scale_overrides);
            kv_k.push(k_roped);
            kv_v.push(v_deq);

            if j == 0 {
                continue;
            }

            let (q_roped, _, _) =
                dequant_rope_qkv(key, layer_idx, q_acc, k_acc, v_acc, sl.scale_x_attn, j, scale_overrides);

            // Targeted diagnostic: layer 0 token 1 — V-bound check
            if layer_idx == 0 && j == 1 {
                let (replayed, a_f64) = verilm_core::attention::replay_attention_roped_raw(
                    &q_roped, &kv_k, &kv_v, rs.scale_a as f64, cfg,
                );
                let n = 16
                    .min(a_f64.len())
                    .min(replayed.len())
                    .min(rs.a.len())
                    .min(cfg.d_head);
                // V min/max across all KV cache entries for head 0 (first d_head dims)
                let d_head = cfg.d_head;
                let mut v_min = vec![f64::INFINITY; d_head];
                let mut v_max = vec![f64::NEG_INFINITY; d_head];
                for v_t in &kv_v {
                    for i in 0..d_head.min(v_t.len()) {
                        if v_t[i] < v_min[i] { v_min[i] = v_t[i]; }
                        if v_t[i] > v_max[i] { v_max[i] = v_t[i]; }
                    }
                }
                eprintln!("\n[DIAG-VBOUND] layer=0 pos=1 scale_a={:.8} kv_entries={}", rs.scale_a, kv_v.len());
                eprintln!("  a_f64[..{}]:  {:?}", n, &a_f64[..n].iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>());
                eprintln!("  a_i8_rep[..{}]: {:?}", n, &replayed[..n]);
                eprintln!("  a_i8_gpu[..{}]: {:?}", n, &rs.a[..n]);
                eprintln!("  V_min[..{}]:  {:?}", n, &v_min[..n].iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
                eprintln!("  V_max[..{}]:  {:?}", n, &v_max[..n].iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
                // Classification: does a_f64[i] lie within [V_min[i], V_max[i]]?
                let mut oob = 0;
                for i in 0..n {
                    if a_f64[i] < v_min[i] || a_f64[i] > v_max[i] {
                        oob += 1;
                    }
                }
                eprintln!("  out-of-V-bound: {}/{} dims", oob, n);
                // Also check: what scale_a would make replayed match claimed?
                // For each element where claimed != 0: implied_scale = a_f64 / claimed
                let mut implied_scales: Vec<f64> = Vec::new();
                for i in 0..n {
                    if rs.a[i] != 0 {
                        implied_scales.push(a_f64[i] / rs.a[i] as f64);
                    }
                }
                if !implied_scales.is_empty() {
                    let median = {
                        let mut s = implied_scales.clone();
                        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        s[s.len() / 2]
                    };
                    eprintln!("  implied_scale (median of a_f64/a_gpu): {:.8} vs actual scale_a: {:.8} (ratio: {:.4})",
                        median, rs.scale_a as f64, median / rs.scale_a as f64);
                }
                all_stats.push(measure_attention_diff(&rs.a, &replayed, layer_idx, j)?);
            } else {
                let replayed = verilm_core::attention::replay_attention_roped(
                    &q_roped, &kv_k, &kv_v, rs.scale_a as f64, cfg,
                );
                all_stats.push(measure_attention_diff(&rs.a, &replayed, layer_idx, j)?);
            }
        }

        // Opened token
        if let Some((q_acc, k_acc, v_acc, rs)) =
            opened_token_qkv(response, layer_idx)
        {
            let shell = response.shell_opening.as_ref().unwrap();
            let sl = &shell.layers[layer_idx];
            let pos = response.token_index as usize;

            let (q_roped, k_roped, v_deq) =
                dequant_rope_qkv(key, layer_idx, q_acc, k_acc, v_acc, sl.scale_x_attn, pos, scale_overrides);
            kv_k.push(k_roped);
            kv_v.push(v_deq);

            // VBOUND diagnostic for opened token at layer 0
            if layer_idx == 0 {
                let (replayed, a_f64) = verilm_core::attention::replay_attention_roped_raw(
                    &q_roped, &kv_k, &kv_v, rs.scale_a as f64, cfg,
                );
                let n = 16
                    .min(a_f64.len())
                    .min(replayed.len())
                    .min(rs.a.len())
                    .min(cfg.d_head);
                let d_head = cfg.d_head;
                let mut v_min = vec![f64::INFINITY; d_head];
                let mut v_max = vec![f64::NEG_INFINITY; d_head];
                for v_t in &kv_v {
                    for i in 0..d_head.min(v_t.len()) {
                        if v_t[i] < v_min[i] { v_min[i] = v_t[i]; }
                        if v_t[i] > v_max[i] { v_max[i] = v_t[i]; }
                    }
                }
                let mut oob = 0;
                for i in 0..n {
                    if a_f64[i] < v_min[i] || a_f64[i] > v_max[i] { oob += 1; }
                }
                let mut implied_scales: Vec<f64> = Vec::new();
                for i in 0..n {
                    if rs.a[i] != 0 {
                        implied_scales.push(a_f64[i] / rs.a[i] as f64);
                    }
                }
                let median = if !implied_scales.is_empty() {
                    let mut s = implied_scales.clone();
                    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    s[s.len() / 2]
                } else { 0.0 };
                eprintln!("\n[DIAG-VBOUND-OPENED] layer=0 pos={} scale_a={:.8} kv_entries={}",
                    pos, rs.scale_a, kv_v.len());
                eprintln!("  a_f64[..{}]:  {:?}", n, &a_f64[..n].iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>());
                eprintln!("  a_i8_rep[..{}]: {:?}", n, &replayed[..n]);
                eprintln!("  a_i8_gpu[..{}]: {:?}", n, &rs.a[..n]);
                eprintln!("  out-of-V-bound: {}/{} dims", oob, n);
                if !implied_scales.is_empty() {
                    eprintln!("  implied_scale: {:.8} vs scale_a: {:.8} (ratio: {:.4})",
                        median, rs.scale_a as f64, median / rs.scale_a as f64);
                }
                all_stats.push(measure_attention_diff(&rs.a, &replayed, layer_idx, pos)?);
            } else {
                let replayed = verilm_core::attention::replay_attention_roped(
                    &q_roped, &kv_k, &kv_v, rs.scale_a as f64, cfg,
                );
                all_stats.push(measure_attention_diff(&rs.a, &replayed, layer_idx, pos)?);
            }
        }
    }

    Ok(all_stats)
}

/// Dequantize + RoPE for Q, K, V accumulators.
///
/// When `overrides` is provided, uses per-channel weight scales for
/// dequantization. Otherwise falls back to VerifierKey's per-tensor scales.
fn dequant_rope_qkv(
    key: &VerifierKey,
    layer_idx: usize,
    q_acc: &[i32],
    k_acc: &[i32],
    v_acc: &[i32],
    scale_x_attn: f32,
    position: usize,
    overrides: Option<&CorridorScaleOverrides>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (q_f64, k_f64, v_f64) = if let Some(ovr) = overrides {
        let q = verilm_core::rope::dequantize_acc_per_channel(
            q_acc, &ovr.wq[layer_idx], scale_x_attn,
        );
        let k = verilm_core::rope::dequantize_acc_per_channel(
            k_acc, &ovr.wk[layer_idx], scale_x_attn,
        );
        let v = verilm_core::rope::dequantize_acc_per_channel(
            v_acc, &ovr.wv[layer_idx], scale_x_attn,
        );
        (q, k, v)
    } else {
        let scale_wq = key.weight_scale_for(layer_idx, MatrixType::Wq);
        let scale_wk = key.weight_scale_for(layer_idx, MatrixType::Wk);
        let scale_wv = key.weight_scale_for(layer_idx, MatrixType::Wv);
        let sx = Some(scale_x_attn);
        let q = verilm_core::rope::dequantize_acc(q_acc, Some(scale_wq), sx);
        let k = verilm_core::rope::dequantize_acc(k_acc, Some(scale_wk), sx);
        let v = verilm_core::rope::dequantize_acc(v_acc, Some(scale_wv), sx);
        (q, k, v)
    };

    let q_roped = verilm_core::rope::apply_rope_q(&q_f64, position, &key.config);
    let k_roped = verilm_core::rope::apply_rope_k(&k_f64, position, &key.config);

    (q_roped, k_roped, v_f64)
}

/// Extract opened token's QKV accumulators from the response.
fn opened_token_qkv(
    r: &V4AuditResponse,
    layer_idx: usize,
) -> Option<(&[i32], &[i32], &[i32], &verilm_core::types::RetainedLayerState)> {
    let shell = r.shell_opening.as_ref()?;
    if layer_idx >= shell.layers.len() || layer_idx >= r.retained.layers.len() {
        return None;
    }
    let sl = &shell.layers[layer_idx];
    let rs = &r.retained.layers[layer_idx];
    match (&sl.q, &sl.k, &sl.v) {
        (Some(q), Some(k), Some(v)) => Some((q, k, v, rs)),
        _ => None,
    }
}

fn build_report(
    measurements: Vec<AttentionDiffStats>,
    n_layers: usize,
) -> Result<CorridorReport, String> {
    let global_linf = measurements.iter().map(|m| m.linf).max().unwrap_or(0);

    let mut per_layer_max_linf = vec![0i16; n_layers];
    let mut pos_max: std::collections::HashMap<usize, i16> = std::collections::HashMap::new();

    for m in &measurements {
        if m.layer < per_layer_max_linf.len() && m.linf > per_layer_max_linf[m.layer] {
            per_layer_max_linf[m.layer] = m.linf;
        }
        let entry = pos_max.entry(m.token_position).or_insert(0);
        if m.linf > *entry {
            *entry = m.linf;
        }
    }

    let mut positions: Vec<usize> = pos_max.keys().copied().collect();
    positions.sort();
    let per_position_max_linf: Vec<i16> = positions.iter().map(|p| pos_max[p]).collect();

    Ok(CorridorReport {
        measurements,
        global_linf,
        per_layer_max_linf,
        per_position_max_linf,
    })
}
