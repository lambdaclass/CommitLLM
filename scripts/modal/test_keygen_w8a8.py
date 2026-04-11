"""
Smoke test: verify keygen loads per-channel weight scales from W8A8 models.

Validates roadmap #1 (quant semantics as first-class protocol data):
  - weight_scale tensors discovered in safetensors
  - per_channel_weight_scales populated in VerifierKey
  - quant_family = "W8A8", scale_derivation = "per_channel_absmax"
  - per-channel scales have correct shapes

Does NOT require GPU inference — only needs the model weights on disk.

Usage:
    modal run --detach scripts/modal/test_keygen_w8a8.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
try:
    from _pins import KEYGEN
except ImportError:
    KEYGEN = []

import modal

app = modal.App("verilm-test-keygen-w8a8")

MODEL_ID = "neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev", "ca-certificates")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    })
    .pip_install(*KEYGEN)
    .add_local_dir(".", remote_path="/build", copy=True, ignore=[
        ".git", "target", "scripts/__pycache__", "*.pdf",
    ])
    .run_commands(
        "cd /build/crates/verilm-py && maturin build --release",
        "bash -c 'pip install /build/target/wheels/verilm_rs-*.whl'",
        "python -c 'import verilm_rs; print(\"verilm_rs OK\")'",
        "rm -rf /build",
    )
)


def _run_test():
    import hashlib
    import json
    import os

    from huggingface_hub import snapshot_download

    import verilm_rs

    failures = []

    def assert_true(cond, msg):
        if not cond:
            failures.append(msg)
            print(f"  FAIL: {msg}")
        else:
            print(f"  OK: {msg}")

    # ── Step 1: Download model weights ──
    print(f"\n1. Downloading {MODEL_ID} weights...")
    model_dir = snapshot_download(
        MODEL_ID,
        allow_patterns=["*.safetensors", "*.json"],
    )
    print(f"  model_dir: {model_dir}")

    # Verify weight_scale tensors exist in safetensors
    from safetensors import safe_open
    st_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    print(f"  safetensors shards: {len(st_files)}")

    scale_tensors = []
    for st_file in st_files:
        with safe_open(os.path.join(model_dir, st_file), framework="pt") as f:
            for name in f.keys():
                if "weight_scale" in name:
                    tensor = f.get_tensor(name)
                    scale_tensors.append((name, tuple(tensor.shape), str(tensor.dtype)))

    print(f"  weight_scale tensors found: {len(scale_tensors)}")
    for name, shape, dtype in scale_tensors[:5]:
        print(f"    {name}: shape={shape} dtype={dtype}")
    if len(scale_tensors) > 5:
        print(f"    ... and {len(scale_tensors) - 5} more")

    assert_true(len(scale_tensors) > 0, "weight_scale tensors exist in safetensors")

    # ── Step 2: Run keygen ──
    print("\n2. Running keygen...")
    seed = hashlib.sha256(b"test-keygen-w8a8").digest()
    key_json = verilm_rs.generate_key(model_dir, list(seed))
    key = json.loads(key_json)
    print(f"  key generated, {len(key_json)} bytes JSON")

    # ── Step 3: Verify quant metadata ──
    print("\n3. Checking quant metadata...")
    assert_true(
        key.get("quant_family") == "W8A8",
        f"quant_family == 'W8A8' (got {key.get('quant_family')!r})",
    )
    assert_true(
        key.get("scale_derivation") == "per_channel_absmax",
        f"scale_derivation == 'per_channel_absmax' (got {key.get('scale_derivation')!r})",
    )
    assert_true(
        key.get("source_dtype") == "I8",
        f"source_dtype == 'I8' (got {key.get('source_dtype')!r})",
    )
    assert_true(
        key.get("rope_aware_replay") is True,
        f"rope_aware_replay == true (got {key.get('rope_aware_replay')!r})",
    )

    # ── Step 4: Verify per-channel weight scales ──
    print("\n4. Checking per_channel_weight_scales...")
    pc_scales = key.get("per_channel_weight_scales", [])
    n_layers = key["config"]["n_layers"]
    hidden_dim = key["config"]["hidden_dim"]
    kv_dim = key["config"]["kv_dim"]
    ffn_dim = key["config"]["ffn_dim"]

    print(f"  n_layers={n_layers}, hidden_dim={hidden_dim}, kv_dim={kv_dim}, ffn_dim={ffn_dim}")

    assert_true(
        len(pc_scales) == n_layers,
        f"per_channel_weight_scales has {len(pc_scales)} layers (expected {n_layers})",
    )

    if len(pc_scales) > 0:
        # Check layer 0 shapes: [Wq, Wk, Wv, Wo, Wg, Wu, Wd] = 7 matrices
        layer0 = pc_scales[0]
        assert_true(len(layer0) == 7, f"layer 0 has {len(layer0)} matrix entries (expected 7)")

        # Expected output_dim for each matrix type in PER_LAYER order:
        # Wq: hidden_dim, Wk: kv_dim, Wv: kv_dim, Wo: hidden_dim,
        # Wg: ffn_dim, Wu: ffn_dim, Wd: hidden_dim
        expected_dims = [hidden_dim, kv_dim, kv_dim, hidden_dim, ffn_dim, ffn_dim, hidden_dim]
        names = ["Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd"]

        for i, (name, expected) in enumerate(zip(names, expected_dims)):
            actual = len(layer0[i])
            assert_true(
                actual == expected,
                f"layer 0 {name} per-channel scales: {actual} channels (expected {expected})",
            )

            # Verify scales are nonzero (meaningful)
            if actual > 0:
                vals = layer0[i]
                nonzero = sum(1 for v in vals if abs(v) > 1e-10)
                assert_true(
                    nonzero == actual,
                    f"layer 0 {name}: all {actual} scales are nonzero",
                )
                min_s = min(abs(v) for v in vals)
                max_s = max(abs(v) for v in vals)
                print(f"    {name}: {actual} channels, range [{min_s:.6f}, {max_s:.6f}]")

    # ── Step 5: Verify old per-tensor weight_scales is empty/zero for native INT8 ──
    print("\n5. Checking legacy weight_scales...")
    ws = key.get("weight_scales", [])
    if len(ws) > 0 and len(ws[0]) > 0:
        # For native INT8, quantize_absmax returns 0.0
        all_zero = all(s == 0.0 for layer in ws for s in layer)
        assert_true(
            all_zero,
            f"legacy weight_scales are all 0.0 for native INT8 (got non-zero)" if not all_zero
            else "legacy weight_scales correctly 0.0 for native INT8",
        )
    else:
        print("  weight_scales is empty (also acceptable)")

    # ── Summary ──
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} assertion(s) failed:")
        for f in failures:
            print(f"  - {f}")
        raise AssertionError(f"{len(failures)} assertion(s) failed")
    else:
        print("ALL CHECKS PASSED")
        print(f"  Model: {MODEL_ID}")
        print(f"  Layers: {n_layers}")
        print(f"  Per-channel scales: {len(pc_scales)} layers × 7 matrices")
        print(f"  quant_family: {key['quant_family']}")
        print(f"  scale_derivation: {key['scale_derivation']}")


@app.function(
    image=image,
    timeout=600,
    # No GPU needed — keygen is CPU-only
)
def run_test():
    _run_test()
