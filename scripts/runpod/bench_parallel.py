#!/usr/bin/env python3
"""Run A/B benchmark configs in parallel on multiple RunPod pods.

Reuses existing pod infrastructure: creates N pods (one per config),
syncs code to all in parallel, runs configs simultaneously, collects results.

Usage:
    export RUNPOD_API_KEY=rpa_...
    python scripts/runpod/bench_parallel.py

    # Terminate all benchmark pods:
    python scripts/runpod/bench_parallel.py --terminate
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POOL_STATE_FILE = os.path.join(_PROJECT_ROOT, ".runpod_pool_state.json")
GPU_TYPE = "NVIDIA A100 80GB PCIe"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
WORKSPACE = "/workspace/verilm"

BENCH_SCRIPT = "scripts/modal/bench_ab_overhead.py"

# Configs to run in parallel — must match CONFIGS in bench_ab_overhead.py
CONFIG_KEYS = ["long_64", "long_128", "long_256", "short_eos_256"]

SETUP_COMMANDS = [
    "apt-get update -qq && apt-get install -y -qq rsync > /dev/null",
    "command -v cargo || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y)",
    'export PATH="$HOME/.cargo/bin:$PATH"',
    "pip install -q 'vllm==0.8.3' 'torch==2.6.0' 'transformers==4.57.6' 'compressed-tensors==0.9.2' 'numpy==2.1.3' 'safetensors==0.7.0' fastapi maturin modal",
]

BUILD_COMMANDS = [
    'export PATH="$HOME/.cargo/bin:$PATH"',
    f"cd {WORKSPACE}/crates/verilm-py && maturin build --release 2>&1 | tail -3",
    f"pip install --force-reinstall {WORKSPACE}/target/wheels/verilm_rs-*.whl 2>&1 | tail -3",
    f"pip install -e {WORKSPACE}/sidecar 2>&1 | tail -3",
    (
        "python3 -c 'import site, os; "
        'open(os.path.join(site.getsitepackages()[0], "verilm_capture.pth"), "w")'
        '.write("import verilm._startup\\n")\''
    ),
]


def get_api_key():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("ERROR: Set RUNPOD_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)
    return key


def load_pool_state():
    if os.path.exists(POOL_STATE_FILE):
        with open(POOL_STATE_FILE) as f:
            return json.load(f)
    return None


def save_pool_state(state):
    with open(POOL_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def clear_pool_state():
    if os.path.exists(POOL_STATE_FILE):
        os.remove(POOL_STATE_FILE)


def get_ssh_pub_key():
    for name in ("id_ed25519.pub", "id_rsa.pub"):
        path = os.path.expanduser(f"~/.ssh/{name}")
        if os.path.exists(path):
            with open(path) as f:
                return f.read().strip()
    print("ERROR: No SSH public key found")
    sys.exit(1)


def create_pod(api_key, name, pub_key):
    """Create one RunPod pod. Returns pod state dict or None."""
    import runpod
    runpod.api_key = api_key

    pod = runpod.create_pod(
        name=name,
        image_name=IMAGE,
        gpu_type_id=GPU_TYPE,
        gpu_count=1,
        volume_in_gb=50,
        container_disk_in_gb=30,
        ports="22/tcp",
        support_public_ip=True,
        env={
            "PUBLIC_KEY": pub_key,
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VERILM_CAPTURE": "1",
        },
        docker_args="",
    )
    return pod["id"]


def wait_for_pod_ssh(api_key, pod_id, timeout=300):
    """Wait for pod to be RUNNING and return SSH info."""
    import runpod
    runpod.api_key = api_key

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pod_info = runpod.get_pod(pod_id)
        status = pod_info.get("desiredStatus", "")
        runtime = pod_info.get("runtime")
        if status == "RUNNING" and runtime:
            ports = runtime.get("ports") or []
            for p in ports:
                if p.get("privatePort") == 22:
                    return p.get("ip"), p.get("publicPort")
        time.sleep(5)
    return None, None


def ssh_cmd(ip, port):
    return [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", str(port),
        f"root@{ip}",
    ]


def wait_ssh_ready(ip, port, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ssh_cmd(ip, port) + ["echo ok"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def run_ssh(ip, port, commands, stream=False, timeout=600):
    script = " && ".join(commands)
    cmd = ssh_cmd(ip, port) + [script]
    if stream:
        return subprocess.run(cmd, timeout=timeout).returncode
    else:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def sync_code(ip, port):
    rsync_cmd = [
        "rsync", "-az", "--delete",
        "-e", f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {port}",
        "--exclude", ".git", "--exclude", "target",
        "--exclude", "__pycache__", "--exclude", "*.pdf",
        "--exclude", ".runpod_pod_state.json", "--exclude", ".runpod_pool_state.json",
        ".", f"root@{ip}:{WORKSPACE}/",
    ]
    result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0


def run_single_config_on_pod(ip, port, config_key, output):
    """Run one config on one pod. Appends output lines to `output` list."""
    # Run the benchmark for just this config key.
    test_cmd = [
        'export PATH="$HOME/.cargo/bin:$PATH"',
        "export VLLM_ENABLE_V1_MULTIPROCESSING=0",
        "export VERILM_CAPTURE=1",
        "export VERILM_COMMIT_TIMERS=1",
        f"cd {WORKSPACE} && python -c \""
        f"import sys; sys.path.insert(0, '.'); "
        f"exec(open('{WORKSPACE}/{BENCH_SCRIPT}').read()); "
        f"llm, server, buf, fr_capture = _load_model(); "
        f"cfg = [c for c in CONFIGS if c[0] == '{config_key}'][0]; "
        f"result = _run_config(cfg[0], cfg[1], cfg[2], cfg[3], llm, server, buf, fr_capture); "
        f"import json; print('RESULT_JSON:' + json.dumps(result))"
        f"\"",
    ]

    script = " && ".join(test_cmd)
    cmd = ssh_cmd(ip, port) + [script]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    result_json = None
    for line in proc.stdout:
        line = line.rstrip()
        output.append(line)
        print(f"[{config_key}] {line}")
        if line.startswith("RESULT_JSON:"):
            result_json = json.loads(line[len("RESULT_JSON:"):])
    proc.wait()
    return result_json


def terminate_all(api_key):
    import runpod
    runpod.api_key = api_key

    pool = load_pool_state()
    if not pool:
        print("No active pool found.")
        return

    for name, info in pool.get("pods", {}).items():
        pod_id = info.get("pod_id")
        if pod_id:
            print(f"Terminating {name} ({pod_id})...")
            try:
                runpod.terminate_pod(pod_id)
            except Exception as e:
                print(f"  Warning: {e}")
    clear_pool_state()
    print("All pods terminated.")


def main():
    parser = argparse.ArgumentParser(description="Parallel A/B benchmark on RunPod")
    parser.add_argument("--terminate", action="store_true", help="Terminate all benchmark pods")
    args = parser.parse_args()

    api_key = get_api_key()

    if args.terminate:
        terminate_all(api_key)
        return

    pub_key = get_ssh_pub_key()
    pool = load_pool_state()
    n_configs = len(CONFIG_KEYS)

    # ── Create or reuse pods ──
    pods = {}  # config_key -> {"pod_id", "ip", "port", "setup_done"}

    if pool and pool.get("pods"):
        import runpod
        runpod.api_key = api_key
        # Check which pods are still alive.
        for config_key, info in pool["pods"].items():
            try:
                pod_info = runpod.get_pod(info["pod_id"])
                if pod_info.get("desiredStatus") == "RUNNING":
                    pods[config_key] = info
                    print(f"  Reusing pod for {config_key}: {info['pod_id']}")
            except Exception:
                pass

    # Create missing pods in parallel.
    missing = [k for k in CONFIG_KEYS if k not in pods]
    if missing:
        print(f"Creating {len(missing)} new pods...")
        pod_ids = {}
        for config_key in missing:
            pod_id = create_pod(api_key, f"verilm-bench-{config_key}", pub_key)
            pod_ids[config_key] = pod_id
            print(f"  Created {config_key}: {pod_id}")

        # Wait for SSH on all new pods in parallel.
        def _wait(config_key):
            ip, port = wait_for_pod_ssh(api_key, pod_ids[config_key])
            if ip and port:
                pods[config_key] = {
                    "pod_id": pod_ids[config_key],
                    "ip": ip, "port": port,
                    "setup_done": False,
                }
                print(f"  {config_key} ready: {ip}:{port}")
            else:
                print(f"  ERROR: {config_key} failed to start")

        threads = [threading.Thread(target=_wait, args=(k,)) for k in missing]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    if len(pods) != n_configs:
        print(f"ERROR: Only {len(pods)}/{n_configs} pods ready")
        sys.exit(1)

    # Save pool state.
    save_pool_state({"pods": pods})

    # ── Setup + sync + build in parallel ──
    print(f"\nSetting up {n_configs} pods in parallel...")

    def _setup_and_build(config_key):
        info = pods[config_key]
        ip, port = info["ip"], info["port"]

        # Wait for SSH.
        if not wait_ssh_ready(ip, port):
            print(f"  [{config_key}] SSH not ready!")
            return False

        # Setup (first time only).
        if not info.get("setup_done"):
            print(f"  [{config_key}] Installing deps...")
            rc = run_ssh(ip, port, SETUP_COMMANDS, stream=False, timeout=300)
            if isinstance(rc, int) and rc != 0:
                print(f"  [{config_key}] Setup failed!")
                return False
            elif hasattr(rc, 'returncode') and rc.returncode != 0:
                print(f"  [{config_key}] Setup failed!")
                return False
            info["setup_done"] = True

        # Sync code.
        print(f"  [{config_key}] Syncing code...")
        if not sync_code(ip, port):
            print(f"  [{config_key}] Sync failed!")
            return False

        # Build.
        print(f"  [{config_key}] Building...")
        rc = run_ssh(ip, port, BUILD_COMMANDS, stream=False, timeout=300)
        if isinstance(rc, int):
            ok = rc == 0
        else:
            ok = rc.returncode == 0
        if not ok:
            print(f"  [{config_key}] Build failed!")
            return False

        print(f"  [{config_key}] Ready.")
        return True

    setup_threads = []
    setup_results = {}
    def _setup_wrapper(k):
        setup_results[k] = _setup_and_build(k)
    for k in CONFIG_KEYS:
        t = threading.Thread(target=_setup_wrapper, args=(k,))
        setup_threads.append(t)
        t.start()
    for t in setup_threads:
        t.join()

    save_pool_state({"pods": pods})

    failed = [k for k, ok in setup_results.items() if not ok]
    if failed:
        print(f"ERROR: Setup failed for: {failed}")
        sys.exit(1)

    # ── Run configs in parallel ──
    print(f"\nRunning {n_configs} configs in parallel...")
    t_start = time.monotonic()

    run_results = {}
    run_outputs = {}

    def _run_wrapper(config_key):
        info = pods[config_key]
        output = []
        run_outputs[config_key] = output
        result = run_single_config_on_pod(info["ip"], info["port"], config_key, output)
        run_results[config_key] = result

    run_threads = []
    for k in CONFIG_KEYS:
        t = threading.Thread(target=_run_wrapper, args=(k,))
        run_threads.append(t)
        t.start()
    for t in run_threads:
        t.join()

    elapsed = time.monotonic() - t_start
    print(f"\nAll configs complete in {elapsed:.1f}s")

    # ── Summary ──
    valid = {k: v for k, v in run_results.items() if v is not None}
    if valid:
        print(f"\n{'='*78}")
        print("SUMMARY — Marginal overhead (median ms)")
        print(f"{'='*78}")
        print(f"{'Config':<20} {'Tok':>5} {'Capture':>9} {'Hook+Sync':>10} {'Commit_U':>9} {'Commit_P':>9} {'P-U':>7} {'Total_P':>9}")
        print(f"{'-'*78}")
        for key in sorted(valid):
            r = valid[key]
            tok = r["gen_tokens"].get("full_packed", r["gen_tokens"].get("baseline", 0))
            print(
                f"{key:<20} {tok:>5} "
                f"{r['marginal_capture_ms']:>+8.1f} "
                f"{r['marginal_hooks_sync_ms']:>+9.1f} "
                f"{r['marginal_commit_unpacked_ms']:>+8.1f} "
                f"{r['marginal_commit_packed_ms']:>+8.1f} "
                f"{r['packed_vs_unpacked_ms']:>+6.1f} "
                f"{r['total_overhead_packed_ms']:>+8.1f}"
            )

    failed_configs = [k for k in CONFIG_KEYS if k not in valid]
    if failed_configs:
        print(f"\nFailed configs: {failed_configs}")
        sys.exit(1)

    print(f"\nPods kept alive for re-runs. Terminate with: python scripts/runpod/bench_parallel.py --terminate")


if __name__ == "__main__":
    main()
