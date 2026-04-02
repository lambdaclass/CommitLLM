//! Quantization parity tests.
//!
//! Hardcoded fixtures that pin the exact behavior of requantize, SiLU LUT
//! construction, and SiLU-gated computation so that Python and Rust
//! implementations cannot silently drift.

use verilm_core::silu;

// ---------------------------------------------------------------------------
// Inline requantize: i32 accumulator -> i8 via clamp
// ---------------------------------------------------------------------------

fn requantize(x: i32) -> i8 {
    x.clamp(-128, 127) as i8
}

// ---------------------------------------------------------------------------
// 1. requantize basic
// ---------------------------------------------------------------------------

#[test]
fn test_requantize_basic() {
    // Values already in [-128, 127] pass through unchanged.
    assert_eq!(requantize(0), 0i8);
    assert_eq!(requantize(1), 1i8);
    assert_eq!(requantize(-1), -1i8);
    assert_eq!(requantize(42), 42i8);
    assert_eq!(requantize(-100), -100i8);
    assert_eq!(requantize(127), 127i8);
    assert_eq!(requantize(-128), -128i8);

    // Values outside the range clamp.
    assert_eq!(requantize(200), 127i8);
    assert_eq!(requantize(-200), -128i8);
    assert_eq!(requantize(1000), 127i8);
    assert_eq!(requantize(-1000), -128i8);
}

// ---------------------------------------------------------------------------
// 2. requantize boundary values
// ---------------------------------------------------------------------------

#[test]
fn test_requantize_boundary_values() {
    assert_eq!(requantize(-129), -128i8); // just below range -> clamp low
    assert_eq!(requantize(-128), -128i8); // exact low boundary
    assert_eq!(requantize(0), 0i8); // zero
    assert_eq!(requantize(127), 127i8); // exact high boundary
    assert_eq!(requantize(128), 127i8); // just above range -> clamp high
    assert_eq!(requantize(i32::MAX), 127i8);
    assert_eq!(requantize(i32::MIN), -128i8);
}

// ---------------------------------------------------------------------------
// 3. SiLU LUT with unit scale – known values
// ---------------------------------------------------------------------------

#[test]
fn test_silu_lut_unit_scale_known_values() {
    let lut = silu::build_silu_lut(1.0);

    // g=0 -> SiLU(0) = 0.0
    assert!(
        (lut[0u8 as usize] - 0.0).abs() < 1e-6,
        "g=0: expected 0.0, got {}",
        lut[0]
    );

    // g=1 -> SiLU(1) = 1/(1+e^-1) ≈ 0.7311
    let idx_1 = 1i8 as u8 as usize; // 1
    assert!(
        (lut[idx_1] - 0.7311).abs() < 0.001,
        "g=1: expected ~0.7311, got {}",
        lut[idx_1]
    );

    // g=-1 -> SiLU(-1) = -1/(1+e^1) ≈ -0.2689
    let idx_neg1 = (-1i8) as u8 as usize; // 255
    assert!(
        (lut[idx_neg1] - (-0.2689)).abs() < 0.001,
        "g=-1: expected ~-0.2689, got {}",
        lut[idx_neg1]
    );

    // g=127 -> SiLU(127) ≈ 127.0 (sigmoid(127) ≈ 1.0)
    let idx_127 = 127i8 as u8 as usize; // 127
    assert!(
        (lut[idx_127] - 127.0).abs() < 0.01,
        "g=127: expected ~127.0, got {}",
        lut[idx_127]
    );

    // g=-128 -> SiLU(-128) ≈ 0.0 (sigmoid(-128) ≈ 0.0)
    let idx_neg128 = (-128i8) as u8 as usize; // 128
    assert!(
        lut[idx_neg128].abs() < 1e-6,
        "g=-128: expected ~0.0, got {}",
        lut[idx_neg128]
    );
}

// ---------------------------------------------------------------------------
// 4. compute_h_unit_scale known values
// ---------------------------------------------------------------------------

#[test]
fn test_silu_h_computation_known_values() {
    // Vectors that exercise several regimes:
    //   index 0: zero inputs          -> h = 0
    //   index 1: positive * positive  -> positive h
    //   index 2: positive * negative  -> negative h
    //   index 3: large values -> saturation to 127
    //   index 4: large negative gate  -> SiLU ≈ 0 -> h ≈ 0
    let g_acc: Vec<i32> = vec![0, 5, 10, 500, -500];
    let u_acc: Vec<i32> = vec![0, 10, -8, 500, 100];

    let h = silu::compute_h_unit_scale(&g_acc, &u_acc);

    // index 0: SiLU(0)*0 = 0
    assert_eq!(h[0], 0i8, "zero inputs should give 0");

    // index 1: g=5, u=10; SiLU(5)≈4.9665; 4.9665*10≈49.665 -> round 50
    assert!(
        (h[1] as i32 - 50).abs() <= 1,
        "pos*pos: expected ~50, got {}",
        h[1]
    );

    // index 2: g=10, u=-8; SiLU(10)≈10.0; 10.0*(-8)=-80 -> round -80
    assert!(
        (h[2] as i32 - (-80)).abs() <= 1,
        "pos*neg: expected ~-80, got {}",
        h[2]
    );

    // index 3: g=500->clamp 127, u=500->clamp 127; SiLU(127)≈127; 127*127=16129 -> clamp 127
    assert_eq!(h[3], 127i8, "saturation high: expected 127, got {}", h[3]);

    // index 4: g=-500->clamp -128, u=100->clamp 100; SiLU(-128)≈0; 0*100≈0
    assert!(
        (h[4] as i32).abs() <= 1,
        "near-zero SiLU: expected ~0, got {}",
        h[4]
    );
}

// ---------------------------------------------------------------------------
// 5. compute_h_unit_scale matches check_silu
// ---------------------------------------------------------------------------

#[test]
fn test_silu_h_matches_check_silu() {
    let g_acc: Vec<i32> = vec![0, 3, -7, 50, -50, 127, -128, 100, -1, 42];
    let u_acc: Vec<i32> = vec![10, -5, 20, -30, 60, 1, -1, 0, 127, -128];

    let h = silu::compute_h_unit_scale(&g_acc, &u_acc);

    // Recover the i8 gate/up vectors (same clamp as compute_h_unit_scale uses).
    let g_i8: Vec<i8> = g_acc.iter().map(|&v| requantize(v)).collect();
    let u_i8: Vec<i8> = u_acc.iter().map(|&v| requantize(v)).collect();

    let lut = silu::build_silu_lut(1.0);

    // With unit scales and zero offset, check_silu should accept the output.
    assert!(
        silu::check_silu(&g_i8, &u_i8, &h, &lut, 1.0, 1.0, 0),
        "check_silu rejected output from compute_h_unit_scale"
    );
}

// ---------------------------------------------------------------------------
// 6. requantize is deterministic
// ---------------------------------------------------------------------------

#[test]
fn test_requantize_is_deterministic() {
    let inputs: Vec<i32> = vec![
        i32::MIN,
        -129,
        -128,
        -1,
        0,
        1,
        127,
        128,
        i32::MAX,
        999,
        -999,
    ];

    for &x in &inputs {
        let expected = requantize(x);
        for _ in 0..100 {
            assert_eq!(
                requantize(x),
                expected,
                "requantize({}) was not deterministic",
                x
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 7. SiLU LUT is deterministic
// ---------------------------------------------------------------------------

#[test]
fn test_silu_lut_is_deterministic() {
    let scale = 1.0f32;
    let expected = silu::build_silu_lut(scale);

    for _ in 0..100 {
        let lut = silu::build_silu_lut(scale);
        for i in 0..256 {
            assert_eq!(
                lut[i].to_bits(),
                expected[i].to_bits(),
                "LUT entry {} differed on repeated build (got {} vs {})",
                i,
                lut[i],
                expected[i]
            );
        }
    }
}
