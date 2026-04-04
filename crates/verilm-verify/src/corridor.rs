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

use verilm_core::attention::{measure_attention_diff, AttentionDiffStats, ReplayPrecision};
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
        measure_roped(
            key,
            prefix_ret,
            prefix_shells,
            response,
            n_layers,
            scale_overrides,
        )?
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
            let replayed =
                verilm_core::attention::replay_attention_reference(&q_i8, &kv_k, &kv_v, cfg);
            all_stats.push(measure_attention_diff(&rs.a, &replayed, layer_idx, j)?);
        }

        // Opened token
        if let Some((q_acc, k_acc, v_acc, rs)) = opened_token_qkv(response, layer_idx) {
            kv_k.push(verilm_core::requantize(k_acc));
            kv_v.push(verilm_core::requantize(v_acc));
            let q_i8 = verilm_core::requantize(q_acc);
            let replayed =
                verilm_core::attention::replay_attention_reference(&q_i8, &kv_k, &kv_v, cfg);
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

            let (_, k_roped, v_deq) = dequant_rope_qkv(
                key,
                layer_idx,
                q_acc,
                k_acc,
                v_acc,
                sl.scale_x_attn,
                j,
                scale_overrides,
            );
            kv_k.push(k_roped);
            kv_v.push(v_deq);

            if j == 0 {
                continue;
            }

            let (q_roped, _, _) = dequant_rope_qkv(
                key,
                layer_idx,
                q_acc,
                k_acc,
                v_acc,
                sl.scale_x_attn,
                j,
                scale_overrides,
            );

            // Targeted diagnostic: layer 0 token 1 — V-bound check
            if layer_idx == 0 && j == 1 {
                let (replayed, a_f64) = verilm_core::attention::replay_attention_roped_raw(
                    &q_roped,
                    &kv_k,
                    &kv_v,
                    rs.scale_a as f64,
                    cfg,
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
                        if v_t[i] < v_min[i] {
                            v_min[i] = v_t[i];
                        }
                        if v_t[i] > v_max[i] {
                            v_max[i] = v_t[i];
                        }
                    }
                }
                eprintln!(
                    "\n[DIAG-VBOUND] layer=0 pos=1 scale_a={:.8} kv_entries={}",
                    rs.scale_a,
                    kv_v.len()
                );
                eprintln!(
                    "  a_f64[..{}]:  {:?}",
                    n,
                    &a_f64[..n]
                        .iter()
                        .map(|v| format!("{:.6}", v))
                        .collect::<Vec<_>>()
                );
                eprintln!("  a_i8_rep[..{}]: {:?}", n, &replayed[..n]);
                eprintln!("  a_i8_gpu[..{}]: {:?}", n, &rs.a[..n]);
                eprintln!(
                    "  V_min[..{}]:  {:?}",
                    n,
                    &v_min[..n]
                        .iter()
                        .map(|v| format!("{:.4}", v))
                        .collect::<Vec<_>>()
                );
                eprintln!(
                    "  V_max[..{}]:  {:?}",
                    n,
                    &v_max[..n]
                        .iter()
                        .map(|v| format!("{:.4}", v))
                        .collect::<Vec<_>>()
                );
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
                    &q_roped,
                    &kv_k,
                    &kv_v,
                    rs.scale_a as f64,
                    cfg,
                );
                all_stats.push(measure_attention_diff(&rs.a, &replayed, layer_idx, j)?);
            }
        }

        // Opened token
        if let Some((q_acc, k_acc, v_acc, rs)) = opened_token_qkv(response, layer_idx) {
            let shell = response.shell_opening.as_ref().unwrap();
            let sl = &shell.layers[layer_idx];
            let pos = response.token_index as usize;

            let (q_roped, k_roped, v_deq) = dequant_rope_qkv(
                key,
                layer_idx,
                q_acc,
                k_acc,
                v_acc,
                sl.scale_x_attn,
                pos,
                scale_overrides,
            );
            kv_k.push(k_roped);
            kv_v.push(v_deq);

            // VBOUND diagnostic for opened token at layer 0
            if layer_idx == 0 {
                let (replayed, a_f64) = verilm_core::attention::replay_attention_roped_raw(
                    &q_roped,
                    &kv_k,
                    &kv_v,
                    rs.scale_a as f64,
                    cfg,
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
                        if v_t[i] < v_min[i] {
                            v_min[i] = v_t[i];
                        }
                        if v_t[i] > v_max[i] {
                            v_max[i] = v_t[i];
                        }
                    }
                }
                let mut oob = 0;
                for i in 0..n {
                    if a_f64[i] < v_min[i] || a_f64[i] > v_max[i] {
                        oob += 1;
                    }
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
                } else {
                    0.0
                };
                eprintln!(
                    "\n[DIAG-VBOUND-OPENED] layer=0 pos={} scale_a={:.8} kv_entries={}",
                    pos,
                    rs.scale_a,
                    kv_v.len()
                );
                eprintln!(
                    "  a_f64[..{}]:  {:?}",
                    n,
                    &a_f64[..n]
                        .iter()
                        .map(|v| format!("{:.6}", v))
                        .collect::<Vec<_>>()
                );
                eprintln!("  a_i8_rep[..{}]: {:?}", n, &replayed[..n]);
                eprintln!("  a_i8_gpu[..{}]: {:?}", n, &rs.a[..n]);
                eprintln!("  out-of-V-bound: {}/{} dims", oob, n);
                if !implied_scales.is_empty() {
                    eprintln!(
                        "  implied_scale: {:.8} vs scale_a: {:.8} (ratio: {:.4})",
                        median,
                        rs.scale_a as f64,
                        median / rs.scale_a as f64
                    );
                }
                all_stats.push(measure_attention_diff(&rs.a, &replayed, layer_idx, pos)?);
            } else {
                let replayed = verilm_core::attention::replay_attention_roped(
                    &q_roped,
                    &kv_k,
                    &kv_v,
                    rs.scale_a as f64,
                    cfg,
                );
                all_stats.push(measure_attention_diff(&rs.a, &replayed, layer_idx, pos)?);
            }
        }
    }

    Ok(all_stats)
}

/// Dequantize + RoPE for Q, K, V accumulators.
///
/// Priority for weight scales:
/// 1. Explicit `overrides` (measurement-time extraction, legacy path)
/// 2. Key's per-channel scales (`per_channel_weight_scales`, W8A8 keygen)
/// 3. Key's per-tensor scales (`weight_scales`, BF16/FP16 keygen)
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
    let q_f64 = dequant_one(
        key,
        layer_idx,
        MatrixType::Wq,
        q_acc,
        scale_x_attn,
        overrides.map(|o| o.wq[layer_idx].as_slice()),
    );
    let k_f64 = dequant_one(
        key,
        layer_idx,
        MatrixType::Wk,
        k_acc,
        scale_x_attn,
        overrides.map(|o| o.wk[layer_idx].as_slice()),
    );
    let v_f64 = dequant_one(
        key,
        layer_idx,
        MatrixType::Wv,
        v_acc,
        scale_x_attn,
        overrides.map(|o| o.wv[layer_idx].as_slice()),
    );

    // Add projection biases (model-dependent, e.g. Qwen2)
    let q_f64 = add_qkv_bias(key, layer_idx, MatrixType::Wq, q_f64);
    let k_f64 = add_qkv_bias(key, layer_idx, MatrixType::Wk, k_f64);
    let v_f64 = add_qkv_bias(key, layer_idx, MatrixType::Wv, v_f64);

    let q_roped = verilm_core::rope::apply_rope_q(&q_f64, position, &key.config);
    let k_roped = verilm_core::rope::apply_rope_k(&k_f64, position, &key.config);

    (q_roped, k_roped, v_f64)
}

/// Dequantize one accumulator with the best available scales.
fn dequant_one(
    key: &VerifierKey,
    layer_idx: usize,
    mt: MatrixType,
    acc: &[i32],
    scale_x: f32,
    override_scales: Option<&[f32]>,
) -> Vec<f64> {
    if let Some(pc) = override_scales {
        verilm_core::rope::dequantize_acc_per_channel(acc, pc, scale_x)
    } else if let Some(pc) = key.per_channel_scales_for(layer_idx, mt) {
        verilm_core::rope::dequantize_acc_per_channel(acc, pc, scale_x)
    } else {
        let scale_w = key.weight_scale_for(layer_idx, mt);
        verilm_core::rope::dequantize_acc(acc, Some(scale_w), Some(scale_x))
    }
}

/// Add QKV projection bias if the model has one for this matrix type.
fn add_qkv_bias(key: &VerifierKey, layer_idx: usize, mt: MatrixType, mut v: Vec<f64>) -> Vec<f64> {
    if let Some(bias) = key.qkv_bias_for(layer_idx, mt) {
        for (x, &b) in v.iter_mut().zip(bias) {
            *x += b as f64;
        }
    }
    v
}

/// Extract opened token's QKV accumulators from the response.
fn opened_token_qkv(
    r: &V4AuditResponse,
    layer_idx: usize,
) -> Option<(
    &[i32],
    &[i32],
    &[i32],
    &verilm_core::types::RetainedLayerState,
)> {
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

/// Measure the attention corridor using committed KV entries.
///
/// Uses KV entries from `kv_entries` (committed under `kv_roots`) directly
/// as the KV cache instead of reconstructing from shell QKV accumulators.
/// This is the production measurement path — committed KV entries are
/// already dequantized and RoPE'd f64 values.
///
/// For the opened token: dequantizes Q from the shell opening, replays
/// attention against committed KV, compares against the retained `a`.
pub fn measure_corridor_committed_kv(
    key: &VerifierKey,
    response: &V4AuditResponse,
    scale_overrides: Option<&CorridorScaleOverrides>,
) -> Result<CorridorReport, String> {
    let kv_entries = response
        .kv_entries
        .as_ref()
        .ok_or("no kv_entries in audit response (committed KV not present)")?;

    let shell = response
        .shell_opening
        .as_ref()
        .ok_or("no shell_opening for opened token")?;

    let cfg = &key.config;
    let n_layers = cfg.n_layers.min(kv_entries.len()).min(shell.layers.len());
    let token_pos = response.token_index as usize;

    let mut all_stats = Vec::new();

    for layer_idx in 0..n_layers {
        let layer_kv = &kv_entries[layer_idx];
        // Build KV cache from committed entries up to and including token_pos.
        let n_positions = (token_pos + 1).min(layer_kv.len());
        let kv_k: Vec<Vec<f64>> = layer_kv[..n_positions]
            .iter()
            .map(|e| e.k_roped.clone())
            .collect();
        let kv_v: Vec<Vec<f64>> = layer_kv[..n_positions]
            .iter()
            .map(|e| e.v_deq.clone())
            .collect();

        if kv_k.is_empty() {
            continue;
        }

        // Get Q from the shell opening, dequant + RoPE.
        let sl = &shell.layers[layer_idx];
        let rs = &response.retained.layers[layer_idx];
        let q_acc = match &sl.q {
            Some(q) => q,
            None => continue,
        };

        let q_f64 = dequant_one(
            key,
            layer_idx,
            MatrixType::Wq,
            q_acc,
            sl.scale_x_attn,
            scale_overrides.map(|o| o.wq[layer_idx].as_slice()),
        );
        let q_f64 = add_qkv_bias(key, layer_idx, MatrixType::Wq, q_f64);
        let q_roped = verilm_core::rope::apply_rope_q(&q_f64, token_pos, cfg);

        // Replay attention from committed KV.
        if layer_idx == 0 && token_pos == 0 {
            // Diagnostic: at pos=0 there is 1 KV entry, softmax=[1.0],
            // so a_f64 must equal v_deq (for head 0 at least).
            let (replayed, a_f64) = verilm_core::attention::replay_attention_roped_raw(
                &q_roped,
                &kv_k,
                &kv_v,
                rs.scale_a as f64,
                cfg,
            );
            let n = 16.min(a_f64.len()).min(rs.a.len());
            let v0 = &kv_v[0];
            let inv_sa = if (rs.scale_a as f64).abs() > 1e-30 {
                1.0 / rs.scale_a as f64
            } else {
                1.0
            };
            eprintln!(
                "\n[DIAG-CKV] token_pos=0 layer=0 n_kv={} scale_a={:.8e} inv_scale_a={:.8e}",
                kv_v.len(),
                rs.scale_a,
                inv_sa
            );
            eprintln!(
                "  v_deq[0][..{n}]:     {:?}",
                &v0[..n]
                    .iter()
                    .map(|v| format!("{:.6}", v))
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "  a_f64[..{n}]:        {:?}",
                &a_f64[..n]
                    .iter()
                    .map(|v| format!("{:.6}", v))
                    .collect::<Vec<_>>()
            );
            // v_deq / scale_a — what a_i8 should be if scale_a is right
            let v_requant: Vec<i8> = v0[..n]
                .iter()
                .map(|&v| (v * inv_sa).round().clamp(-128.0, 127.0) as i8)
                .collect();
            eprintln!("  v_deq/scale_a[..{n}]: {:?}", v_requant);
            eprintln!("  replayed_i8[..{n}]:  {:?}", &replayed[..n]);
            eprintln!("  claimed_i8[..{n}]:   {:?}", &rs.a[..n]);
            // Check: does a_f64 == v_deq? (head 0, d_head dims)
            let d = cfg.d_head.min(n);
            let max_diff_f64: f64 = (0..d).map(|i| (a_f64[i] - v0[i]).abs()).fold(0.0, f64::max);
            eprintln!("  |a_f64 - v_deq| head0 max: {:.8e}", max_diff_f64);
            // Check head layout: kv_dim vs hidden_dim
            eprintln!(
                "  v_deq[0] len={} kv_dim={} hidden_dim={} n_kv_heads={} n_q_heads={} d_head={}",
                v0.len(),
                cfg.n_kv_heads * cfg.d_head,
                cfg.hidden_dim,
                cfg.n_kv_heads,
                cfg.n_q_heads,
                cfg.d_head
            );

            // ── Per-head best-match matrix ──
            // With n_kv=1, attention output for Q-head qh = V[kv_head].
            // Build 28×4 L-inf matrix to detect head mapping bugs.
            let d_head = cfg.d_head;
            let n_q = cfg.n_q_heads;
            let n_kv = cfg.n_kv_heads;
            let heads_per_kv = n_q / n_kv;
            eprintln!(
                "\n[DIAG-HEAD-MAP] n_q_heads={} n_kv_heads={} heads_per_kv={} d_head={}",
                n_q, n_kv, heads_per_kv, d_head
            );

            // For each KV head, quantize v_deq slice to i8
            let mut v_quant_per_kvh: Vec<Vec<i8>> = Vec::new();
            for kvh in 0..n_kv {
                let start = kvh * d_head;
                let end = start + d_head;
                let slice = &v0[start..end.min(v0.len())];
                let quant: Vec<i8> = slice
                    .iter()
                    .map(|&v| (v * inv_sa).round().clamp(-128.0, 127.0) as i8)
                    .collect();
                v_quant_per_kvh.push(quant);
            }

            // L-inf matrix: match[qh][kvh] = max |claimed_qh[i] - v_quant_kvh[i]|
            eprintln!("  L-inf match matrix (rows=Q-heads, cols=KV-heads):");
            eprintln!(
                "  {:>4} | {:>6} {:>6} {:>6} {:>6} | best  expected",
                "qh", "kvh0", "kvh1", "kvh2", "kvh3"
            );
            eprintln!("  {}", "-".repeat(56));
            for qh in 0..n_q {
                let claimed_start = qh * d_head;
                let claimed_end = claimed_start + d_head;
                if claimed_end > rs.a.len() {
                    break;
                }
                let claimed_slice = &rs.a[claimed_start..claimed_end];

                let mut linfs = Vec::new();
                for kvh in 0..n_kv {
                    let vq = &v_quant_per_kvh[kvh];
                    let linf: i16 = claimed_slice
                        .iter()
                        .zip(vq.iter())
                        .map(|(&c, &v)| (c as i16 - v as i16).abs())
                        .max()
                        .unwrap_or(0);
                    linfs.push(linf);
                }

                let best_kvh = linfs
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &l)| l)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let expected_kvh = qh / heads_per_kv;
                let marker = if best_kvh != expected_kvh {
                    " *** MISMATCH"
                } else {
                    ""
                };
                eprintln!(
                    "  {:>4} | {:>6} {:>6} {:>6} {:>6} | kvh{}  kvh{}{}",
                    qh, linfs[0], linfs[1], linfs[2], linfs[3], best_kvh, expected_kvh, marker
                );
            }

            // Check: are Q heads within the same GQA group identical in claimed?
            eprintln!(
                "\n[DIAG-GQA-GROUP] Intra-group consistency (should be identical for n_kv=1):"
            );
            for kvg in 0..n_kv {
                let first_qh = kvg * heads_per_kv;
                let ref_start = first_qh * d_head;
                let ref_end = ref_start + d_head;
                if ref_end > rs.a.len() {
                    break;
                }
                let ref_slice = &rs.a[ref_start..ref_end];

                let mut group_info = Vec::new();
                for offset in 0..heads_per_kv {
                    let qh = first_qh + offset;
                    let s = qh * d_head;
                    let e = s + d_head;
                    if e > rs.a.len() {
                        break;
                    }
                    let slice = &rs.a[s..e];
                    let linf: i16 = ref_slice
                        .iter()
                        .zip(slice.iter())
                        .map(|(&a, &b)| (a as i16 - b as i16).abs())
                        .max()
                        .unwrap_or(0);
                    group_info.push((qh, linf));
                }
                let maxdiff = group_info.iter().map(|(_, l)| *l).max().unwrap_or(0);
                eprintln!(
                    "  KV group {} (Q heads {}-{}): max intra-group L-inf = {}",
                    kvg,
                    first_qh,
                    first_qh + heads_per_kv - 1,
                    maxdiff
                );
            }

            // Show first 8 elements of claimed_i8 for Q heads 0 and 7 (different KV groups)
            eprintln!("\n[DIAG-HEAD-SLICES] First 8 elements per Q-head:");
            for qh in [0, 1, 7, 14, 21].iter().copied() {
                if (qh + 1) * d_head > rs.a.len() {
                    break;
                }
                let s = qh * d_head;
                let slice = &rs.a[s..s + 8.min(d_head)];
                let expected_kvh = qh / heads_per_kv;
                eprintln!("  qh={:>2} (kvh={}): {:?}", qh, expected_kvh, slice);
            }
            for kvh in 0..n_kv {
                let slice = &v_quant_per_kvh[kvh][..8.min(d_head)];
                eprintln!("  v_quant kvh={}: {:?}", kvh, slice);
            }

            all_stats.push(measure_attention_diff(
                &rs.a, &replayed, layer_idx, token_pos,
            )?);
        } else {
            let replayed = verilm_core::attention::replay_attention_roped(
                &q_roped,
                &kv_k,
                &kv_v,
                rs.scale_a as f64,
                cfg,
            );
            all_stats.push(measure_attention_diff(
                &rs.a, &replayed, layer_idx, token_pos,
            )?);
        }
    }

    build_report(all_stats, n_layers)
}

/// Like [`measure_corridor_committed_kv`] but with selectable replay precision.
///
/// Uses [`verilm_core::attention::replay_attention_roped_precision`] to test
/// how different arithmetic precisions affect the corridor gap:
///
/// - `F64`: current verifier default (over-precise vs GPU)
/// - `F32`: tests whether f32 accumulation alone closes the gap
/// - `Fp16InputsF32Accum`: fp16 input truncation + f32 accumulation (closest to GPU SDPA)
/// - `Bf16InputsF32Accum`: bf16 input truncation + f32 accumulation (for bf16 models)
pub fn measure_corridor_committed_kv_precision(
    key: &VerifierKey,
    response: &V4AuditResponse,
    scale_overrides: Option<&CorridorScaleOverrides>,
    precision: ReplayPrecision,
) -> Result<CorridorReport, String> {
    let kv_entries = response
        .kv_entries
        .as_ref()
        .ok_or("no kv_entries in audit response (committed KV not present)")?;

    let shell = response
        .shell_opening
        .as_ref()
        .ok_or("no shell_opening for opened token")?;

    let cfg = &key.config;
    let n_layers = cfg.n_layers.min(kv_entries.len()).min(shell.layers.len());
    let token_pos = response.token_index as usize;

    let mut all_stats = Vec::new();

    for layer_idx in 0..n_layers {
        let layer_kv = &kv_entries[layer_idx];
        let n_positions = (token_pos + 1).min(layer_kv.len());
        let kv_k: Vec<Vec<f64>> = layer_kv[..n_positions]
            .iter()
            .map(|e| e.k_roped.clone())
            .collect();
        let kv_v: Vec<Vec<f64>> = layer_kv[..n_positions]
            .iter()
            .map(|e| e.v_deq.clone())
            .collect();

        if kv_k.is_empty() {
            continue;
        }

        let sl = &shell.layers[layer_idx];
        let rs = &response.retained.layers[layer_idx];
        let q_acc = match &sl.q {
            Some(q) => q,
            None => continue,
        };

        let q_f64 = dequant_one(
            key,
            layer_idx,
            MatrixType::Wq,
            q_acc,
            sl.scale_x_attn,
            scale_overrides.map(|o| o.wq[layer_idx].as_slice()),
        );
        let q_f64 = add_qkv_bias(key, layer_idx, MatrixType::Wq, q_f64);
        let q_roped = verilm_core::rope::apply_rope_q(&q_f64, token_pos, cfg);

        let replayed = verilm_core::attention::replay_attention_roped_precision(
            &q_roped,
            &kv_k,
            &kv_v,
            rs.scale_a as f64,
            cfg,
            precision,
        );
        all_stats.push(measure_attention_diff(
            &rs.a, &replayed, layer_idx, token_pos,
        )?);
    }

    build_report(all_stats, n_layers)
}

/// Backwards-compatible wrapper: f32 corridor measurement.
pub fn measure_corridor_committed_kv_f32(
    key: &VerifierKey,
    response: &V4AuditResponse,
    scale_overrides: Option<&CorridorScaleOverrides>,
) -> Result<CorridorReport, String> {
    measure_corridor_committed_kv_precision(key, response, scale_overrides, ReplayPrecision::F32)
}

/// Per-head boundary simulation results for a single layer+position.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BoundarySimEntry {
    pub layer: usize,
    pub token_position: usize,
    /// Per-tensor scale (current protocol).
    pub scale_a_tensor: f64,
    /// Per-head scales (simulated).
    pub scale_a_per_head: Vec<f64>,
    /// Current corridor: L-inf under per-tensor requantization.
    pub linf_tensor: i16,
    /// Simulated corridor: max over heads of per-head L-inf.
    pub linf_per_head: i16,
    /// Per-head L-inf breakdown (one per Q-head).
    pub linf_per_head_detail: Vec<i16>,
    /// Float-space adversarial room under current τ=10.
    /// Per-tensor: τ * scale_a_tensor.
    pub float_room_tensor: f64,
    /// Max per-head float room: max_h(τ * scale_a_per_head[h]).
    pub float_room_per_head_max: f64,
    /// Mean per-head float room.
    pub float_room_per_head_mean: f64,
    /// Ratio: float_room_per_head_max / float_room_tensor (< 1 means improvement).
    pub float_room_ratio: f64,
}

/// Full boundary simulation report across all layers/positions.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BoundarySimReport {
    pub entries: Vec<BoundarySimEntry>,
    /// Summary: global max L-inf under per-tensor.
    pub global_linf_tensor: i16,
    /// Summary: global max L-inf under per-head.
    pub global_linf_per_head: i16,
    /// Summary: worst-case float room ratio across all entries.
    pub worst_float_room_ratio: f64,
    /// Summary: mean float room ratio across all entries.
    pub mean_float_room_ratio: f64,
}

/// INT16 retained-boundary simulation for a single layer+position.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Int16BoundaryEntry {
    pub layer: usize,
    pub token_position: usize,
    /// Current protocol scale (INT8 retained `a`).
    pub scale_a_int8: f64,
    /// Simulated INT16 scale for the same replayed tensor.
    pub scale_a_int16: f64,
    /// Current corridor in retained INT8 units.
    pub linf_int8: i16,
    /// Simulated corridor in retained INT16 units.
    pub linf_int16: i32,
    /// Honest float-space gap under current INT8 boundary.
    pub float_linf_int8: f64,
    /// Honest float-space gap under simulated INT16 boundary.
    pub float_linf_int16: f64,
    /// Ratio: float_linf_int16 / float_linf_int8 (< 1 means improvement).
    pub float_linf_ratio: f64,
    /// Ratio: scale_a_int16 / scale_a_int8 (~1/258 if INT16 helps ideally).
    pub scale_ratio: f64,
}

/// Full INT16 retained-boundary simulation report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Int16BoundaryReport {
    pub entries: Vec<Int16BoundaryEntry>,
    pub global_linf_int8: i16,
    pub global_linf_int16: i32,
    pub global_float_linf_int8: f64,
    pub global_float_linf_int16: f64,
    pub worst_float_linf_ratio: f64,
    pub mean_float_linf_ratio: f64,
    pub worst_scale_ratio: f64,
    pub mean_scale_ratio: f64,
}

/// Simulate per-head vs per-tensor boundary strategies on committed KV audit data.
///
/// For each layer, replays attention to get `a_f64`, then:
///   1. Requantizes with the original per-tensor `scale_a` → current corridor
///   2. Computes per-head scales from `a_f64`, requantizes → simulated corridor
///   3. Compares float-space adversarial room under both strategies
///
/// This is a measurement-only simulation — no protocol changes required.
pub fn simulate_boundary_strategies(
    key: &VerifierKey,
    response: &V4AuditResponse,
    scale_overrides: Option<&CorridorScaleOverrides>,
) -> Result<BoundarySimReport, String> {
    let kv_entries = response
        .kv_entries
        .as_ref()
        .ok_or("no kv_entries in audit response")?;
    let shell = response
        .shell_opening
        .as_ref()
        .ok_or("no shell_opening for opened token")?;

    let cfg = &key.config;
    let n_layers = cfg.n_layers.min(kv_entries.len()).min(shell.layers.len());
    let token_pos = response.token_index as usize;
    let d_head = cfg.d_head;
    let n_q_heads = cfg.n_q_heads;
    let tau = 10.0_f64; // Current protocol tolerance

    let mut entries = Vec::new();

    for layer_idx in 0..n_layers {
        let layer_kv = &kv_entries[layer_idx];
        let n_positions = (token_pos + 1).min(layer_kv.len());
        let kv_k: Vec<Vec<f64>> = layer_kv[..n_positions]
            .iter()
            .map(|e| e.k_roped.clone())
            .collect();
        let kv_v: Vec<Vec<f64>> = layer_kv[..n_positions]
            .iter()
            .map(|e| e.v_deq.clone())
            .collect();
        if kv_k.is_empty() {
            continue;
        }

        let sl = &shell.layers[layer_idx];
        let rs = &response.retained.layers[layer_idx];
        let q_acc = match &sl.q {
            Some(q) => q,
            None => continue,
        };

        let q_f64 = dequant_one(
            key,
            layer_idx,
            MatrixType::Wq,
            q_acc,
            sl.scale_x_attn,
            scale_overrides.map(|o| o.wq[layer_idx].as_slice()),
        );
        let q_f64 = add_qkv_bias(key, layer_idx, MatrixType::Wq, q_f64);
        let q_roped = verilm_core::rope::apply_rope_q(&q_f64, token_pos, cfg);

        // Replay to get raw f64 attention output + per-tensor i8
        let (_replayed_i8, a_f64) = verilm_core::attention::replay_attention_roped_raw(
            &q_roped,
            &kv_k,
            &kv_v,
            rs.scale_a as f64,
            cfg,
        );

        let scale_a_tensor = rs.scale_a as f64;
        let inv_tensor = if scale_a_tensor.abs() > 1e-30 {
            1.0 / scale_a_tensor
        } else {
            1.0
        };

        // --- Strategy 1: per-tensor (current) ---
        // Requantize verifier's a_f64 with per-tensor scale, compare to GPU's a_i8
        let mut linf_tensor: i16 = 0;
        for i in 0..rs.a.len().min(a_f64.len()) {
            let rep = (a_f64[i] * inv_tensor).round().clamp(-128.0, 127.0) as i8;
            let diff = (rs.a[i] as i16 - rep as i16).abs();
            if diff > linf_tensor {
                linf_tensor = diff;
            }
        }

        // --- Strategy 2: per-head ---
        // Compute per-head scale from GPU's a_i8 dequantized (best we can do)
        // GPU's float output ≈ a_i8 * scale_a (lossy but close)
        let mut scale_per_head = vec![0.0f64; n_q_heads];
        for qh in 0..n_q_heads {
            let start = qh * d_head;
            let end = (start + d_head).min(a_f64.len());
            let mut max_abs = 0.0f64;
            for i in start..end {
                // Use verifier's a_f64 as proxy for GPU's float output
                let abs_val = a_f64[i].abs();
                if abs_val > max_abs {
                    max_abs = abs_val;
                }
            }
            scale_per_head[qh] = max_abs / 127.0;
        }

        // Requantize GPU's claimed output with per-head scales
        // GPU's float ≈ a_i8_gpu * scale_a_tensor
        let mut linf_per_head_detail = vec![0i16; n_q_heads];
        for qh in 0..n_q_heads {
            let start = qh * d_head;
            let end = (start + d_head).min(rs.a.len()).min(a_f64.len());
            let sh = scale_per_head[qh];
            let inv_sh = if sh.abs() > 1e-30 { 1.0 / sh } else { 1.0 };

            for i in start..end {
                // GPU's float output (approximated from claimed i8)
                let gpu_float = rs.a[i] as f64 * scale_a_tensor;
                let gpu_perhead = (gpu_float * inv_sh).round().clamp(-128.0, 127.0) as i8;

                // Verifier's replayed output requantized with per-head scale
                let ver_perhead = (a_f64[i] * inv_sh).round().clamp(-128.0, 127.0) as i8;

                let diff = (gpu_perhead as i16 - ver_perhead as i16).abs();
                if diff > linf_per_head_detail[qh] {
                    linf_per_head_detail[qh] = diff;
                }
            }
        }
        let linf_per_head = linf_per_head_detail.iter().copied().max().unwrap_or(0);

        // Float-space adversarial room
        let float_room_tensor = tau * scale_a_tensor;
        let float_room_per_head_max = scale_per_head
            .iter()
            .map(|&s| tau * s)
            .fold(0.0f64, f64::max);
        let float_room_per_head_mean =
            scale_per_head.iter().map(|&s| tau * s).sum::<f64>() / n_q_heads as f64;
        let float_room_ratio = if float_room_tensor.abs() > 1e-30 {
            float_room_per_head_max / float_room_tensor
        } else {
            1.0
        };

        entries.push(BoundarySimEntry {
            layer: layer_idx,
            token_position: token_pos,
            scale_a_tensor,
            scale_a_per_head: scale_per_head,
            linf_tensor,
            linf_per_head,
            linf_per_head_detail,
            float_room_tensor,
            float_room_per_head_max,
            float_room_per_head_mean,
            float_room_ratio,
        });
    }

    let global_linf_tensor = entries.iter().map(|e| e.linf_tensor).max().unwrap_or(0);
    let global_linf_per_head = entries.iter().map(|e| e.linf_per_head).max().unwrap_or(0);
    let worst_float_room_ratio = entries
        .iter()
        .map(|e| e.float_room_ratio)
        .fold(0.0f64, f64::max);
    let mean_float_room_ratio = if entries.is_empty() {
        1.0
    } else {
        entries.iter().map(|e| e.float_room_ratio).sum::<f64>() / entries.len() as f64
    };

    Ok(BoundarySimReport {
        entries,
        global_linf_tensor,
        global_linf_per_head,
        worst_float_room_ratio,
        mean_float_room_ratio,
    })
}

/// Simulate INT16 retained `a` with the same per-tensor boundary structure.
///
/// This does not change protocol code paths. It approximates the provider's
/// float output using `claimed_i8 * scale_a_int8`, then asks:
/// if the same tensor had been retained as INT16 with a fresh tensor-wide
/// scale, how much would the honest float-space corridor shrink?
pub fn simulate_int16_boundary(
    key: &VerifierKey,
    response: &V4AuditResponse,
    scale_overrides: Option<&CorridorScaleOverrides>,
) -> Result<Int16BoundaryReport, String> {
    let kv_entries = response
        .kv_entries
        .as_ref()
        .ok_or("no kv_entries in audit response")?;
    let shell = response
        .shell_opening
        .as_ref()
        .ok_or("no shell_opening for opened token")?;

    let cfg = &key.config;
    let n_layers = cfg.n_layers.min(kv_entries.len()).min(shell.layers.len());
    let token_pos = response.token_index as usize;

    let mut entries = Vec::new();

    for layer_idx in 0..n_layers {
        let layer_kv = &kv_entries[layer_idx];
        let n_positions = (token_pos + 1).min(layer_kv.len());
        let kv_k: Vec<Vec<f64>> = layer_kv[..n_positions]
            .iter()
            .map(|e| e.k_roped.clone())
            .collect();
        let kv_v: Vec<Vec<f64>> = layer_kv[..n_positions]
            .iter()
            .map(|e| e.v_deq.clone())
            .collect();
        if kv_k.is_empty() {
            continue;
        }

        let sl = &shell.layers[layer_idx];
        let rs = &response.retained.layers[layer_idx];
        let q_acc = match &sl.q {
            Some(q) => q,
            None => continue,
        };

        let q_f64 = dequant_one(
            key,
            layer_idx,
            MatrixType::Wq,
            q_acc,
            sl.scale_x_attn,
            scale_overrides.map(|o| o.wq[layer_idx].as_slice()),
        );
        let q_f64 = add_qkv_bias(key, layer_idx, MatrixType::Wq, q_f64);
        let q_roped = verilm_core::rope::apply_rope_q(&q_f64, token_pos, cfg);

        let (_replayed_i8, a_f64) = verilm_core::attention::replay_attention_roped_raw(
            &q_roped,
            &kv_k,
            &kv_v,
            rs.scale_a as f64,
            cfg,
        );

        let scale_a_int8 = rs.scale_a as f64;
        let inv_int8 = if scale_a_int8.abs() > 1e-30 {
            1.0 / scale_a_int8
        } else {
            1.0
        };

        let mut linf_int8: i16 = 0;
        for i in 0..rs.a.len().min(a_f64.len()) {
            let replay_q = (a_f64[i] * inv_int8).round().clamp(-128.0, 127.0) as i8;
            let diff = (rs.a[i] as i16 - replay_q as i16).abs();
            if diff > linf_int8 {
                linf_int8 = diff;
            }
        }
        let float_linf_int8 = linf_int8 as f64 * scale_a_int8;

        // Use the larger of replay and dequantized claimed magnitudes so the
        // simulated INT16 scale can faithfully represent either side.
        let mut max_abs = 0.0f64;
        for i in 0..rs.a.len().min(a_f64.len()) {
            let gpu_float_proxy = rs.a[i] as f64 * scale_a_int8;
            max_abs = max_abs.max(a_f64[i].abs()).max(gpu_float_proxy.abs());
        }
        let scale_a_int16 = if max_abs > 0.0 { max_abs / 32767.0 } else { 1.0 };
        let inv_int16 = if scale_a_int16.abs() > 1e-30 {
            1.0 / scale_a_int16
        } else {
            1.0
        };

        let mut linf_int16: i32 = 0;
        for i in 0..rs.a.len().min(a_f64.len()) {
            let gpu_float_proxy = rs.a[i] as f64 * scale_a_int8;
            let gpu_q = (gpu_float_proxy * inv_int16)
                .round()
                .clamp(-32768.0, 32767.0) as i32;
            let replay_q = (a_f64[i] * inv_int16)
                .round()
                .clamp(-32768.0, 32767.0) as i32;
            let diff = (gpu_q - replay_q).abs();
            if diff > linf_int16 {
                linf_int16 = diff;
            }
        }
        let float_linf_int16 = linf_int16 as f64 * scale_a_int16;
        let float_linf_ratio = if float_linf_int8.abs() > 1e-30 {
            float_linf_int16 / float_linf_int8
        } else {
            1.0
        };
        let scale_ratio = if scale_a_int8.abs() > 1e-30 {
            scale_a_int16 / scale_a_int8
        } else {
            1.0
        };

        entries.push(Int16BoundaryEntry {
            layer: layer_idx,
            token_position: token_pos,
            scale_a_int8,
            scale_a_int16,
            linf_int8,
            linf_int16,
            float_linf_int8,
            float_linf_int16,
            float_linf_ratio,
            scale_ratio,
        });
    }

    let global_linf_int8 = entries.iter().map(|e| e.linf_int8).max().unwrap_or(0);
    let global_linf_int16 = entries.iter().map(|e| e.linf_int16).max().unwrap_or(0);
    let global_float_linf_int8 = entries
        .iter()
        .map(|e| e.float_linf_int8)
        .fold(0.0f64, f64::max);
    let global_float_linf_int16 = entries
        .iter()
        .map(|e| e.float_linf_int16)
        .fold(0.0f64, f64::max);
    let worst_float_linf_ratio = entries
        .iter()
        .map(|e| e.float_linf_ratio)
        .fold(0.0f64, f64::max);
    let mean_float_linf_ratio = if entries.is_empty() {
        1.0
    } else {
        entries.iter().map(|e| e.float_linf_ratio).sum::<f64>() / entries.len() as f64
    };
    let worst_scale_ratio = entries.iter().map(|e| e.scale_ratio).fold(0.0f64, f64::max);
    let mean_scale_ratio = if entries.is_empty() {
        1.0
    } else {
        entries.iter().map(|e| e.scale_ratio).sum::<f64>() / entries.len() as f64
    };

    Ok(Int16BoundaryReport {
        entries,
        global_linf_int8,
        global_linf_int16,
        global_float_linf_int8,
        global_float_linf_int16,
        worst_float_linf_ratio,
        mean_float_linf_ratio,
        worst_scale_ratio,
        mean_scale_ratio,
    })
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
