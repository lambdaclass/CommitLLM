#!/usr/bin/env python3
"""Run verilm GPU tests on a persistent RunPod pod.

First run: creates pod + installs deps (~5 min). Subsequent runs: syncs
code + re-runs (~30s). The pod stays alive between runs for fast iteration.

Usage:
    # Set key in env (never committed):
    export RUNPOD_API_KEY=rpa_...

    # Run sampled decoding test:
    python scripts/runpod/test.py

    # Run a specific test script:
    python scripts/runpod/test.py --script scripts/modal/test_e2e_v4.py

    # Teardown (terminate pod):
    python scripts/runpod/test.py --terminate

    # Force fresh pod:
    python scripts/runpod/test.py --fresh
"""

import argparse
import json
import os
import subprocess
import sys
import time

# State file at project root (stable across script moves).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POD_STATE_FILE = os.path.join(_PROJECT_ROOT, ".runpod_pod_state.json")
GPU_TYPE = "NVIDIA A100 80GB PCIe"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
WORKSPACE = "/workspace/verilm"

# Deps that persist on the pod's volume across code syncs.
SETUP_COMMANDS = [
    # System deps (rsync for code sync).
    "apt-get update -qq && apt-get install -y -qq rsync > /dev/null",
    # Rust toolchain.
    "command -v cargo || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y)",
    'export PATH="$HOME/.cargo/bin:$PATH"',
    # Clean corrupt package metadata that crashes pip's resolver.
    "rm -rf /usr/local/lib/python3.11/dist-packages/~*",
    # Python deps.
    "pip install -q 'vllm==0.8.3' 'torch==2.6.0' 'transformers==4.57.6' 'compressed-tensors==0.9.2' 'numpy==2.1.3' 'safetensors==0.7.0' fastapi maturin modal ninja",
]

# Per-run commands: rebuild Rust bindings + install sidecar.
BUILD_COMMANDS = [
    'export PATH="$HOME/.cargo/bin:$PATH"',
    f"cd {WORKSPACE}/crates/verilm-py && maturin build --release 2>&1 | tail -3",
    f"pip install --force-reinstall {WORKSPACE}/target/wheels/verilm_rs-*.whl 2>&1 | tail -3",
    f"pip install -e {WORKSPACE}/sidecar 2>&1 | tail -3",
    # Install the .pth hook for capture.
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


def load_pod_state():
    if os.path.exists(POD_STATE_FILE):
        with open(POD_STATE_FILE) as f:
            return json.load(f)
    return None


def save_pod_state(state):
    with open(POD_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def clear_pod_state():
    if os.path.exists(POD_STATE_FILE):
        os.remove(POD_STATE_FILE)


def create_pod(api_key):
    """Create a RunPod GPU pod with SSH enabled."""
    import runpod
    runpod.api_key = api_key

    # Read SSH public key.
    ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
    if not os.path.exists(ssh_key_path):
        ssh_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")
    if not os.path.exists(ssh_key_path):
        print("ERROR: No SSH public key found at ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub")
        sys.exit(1)

    with open(ssh_key_path) as f:
        pub_key = f.read().strip()

    print(f"Creating RunPod pod ({GPU_TYPE})...")
    pod = runpod.create_pod(
        name="verilm-test",
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

    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")

    # Wait for pod to be ready.
    print("Waiting for pod to start...", end="", flush=True)
    for _ in range(120):
        pod_info = runpod.get_pod(pod_id)
        status = pod_info.get("desiredStatus", "")
        runtime = pod_info.get("runtime")
        if status == "RUNNING" and runtime:
            ports = runtime.get("ports") or []
            ssh_port = None
            ssh_ip = None
            for p in ports:
                if p.get("privatePort") == 22:
                    ssh_ip = p.get("ip")
                    ssh_port = p.get("publicPort")
                    break
            if ssh_ip and ssh_port:
                print(f"\nPod ready: ssh root@{ssh_ip} -p {ssh_port}")
                state = {
                    "pod_id": pod_id,
                    "ssh_ip": ssh_ip,
                    "ssh_port": ssh_port,
                    "setup_done": False,
                }
                save_pod_state(state)
                return state
        print(".", end="", flush=True)
        time.sleep(5)

    print("\nERROR: Pod did not start within 10 minutes")
    sys.exit(1)


def terminate_pod(api_key):
    """Terminate the RunPod pod."""
    import runpod
    runpod.api_key = api_key

    state = load_pod_state()
    if not state:
        print("No active pod found.")
        return

    pod_id = state["pod_id"]
    print(f"Terminating pod {pod_id}...")
    runpod.terminate_pod(pod_id)
    clear_pod_state()
    print("Pod terminated.")


def ssh_cmd(state):
    """Build base SSH command with connection args."""
    return [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", str(state["ssh_port"]),
        f"root@{state['ssh_ip']}",
    ]


def run_ssh(state, commands, stream=False):
    """Run commands on the pod via SSH."""
    script = " && ".join(commands)
    cmd = ssh_cmd(state) + [script]
    if stream:
        result = subprocess.run(cmd)
        return result.returncode
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"SSH command failed (rc={result.returncode}):")
            print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        return result


def sync_code(state):
    """Rsync local code to the pod."""
    print("Syncing code...")
    rsync_cmd = [
        "rsync", "-az", "--delete",
        "-e", f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {state['ssh_port']}",
        "--exclude", ".git",
        "--exclude", "target",
        "--exclude", "__pycache__",
        "--exclude", "*.pdf",
        "--exclude", ".runpod_pod_state.json",
        ".",
        f"root@{state['ssh_ip']}:{WORKSPACE}/",
    ]
    result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"rsync failed: {result.stderr}")
        sys.exit(1)
    print("Code synced.")


def setup_pod(state):
    """One-time dependency installation."""
    print("Installing dependencies (first time)...")
    rc = run_ssh(state, SETUP_COMMANDS, stream=True)
    if rc != 0:
        print("Setup failed!")
        sys.exit(1)
    state["setup_done"] = True
    save_pod_state(state)
    print("Dependencies installed.")


def build_and_test(state, test_script):
    """Build Rust bindings, install sidecar, run test."""
    print("Building Rust bindings...")
    rc = run_ssh(state, BUILD_COMMANDS, stream=True)
    if rc != 0:
        print("Build failed!")
        sys.exit(1)

    # Extract just the test function from the Modal script — run _run_test() directly.
    remote_script = f"{WORKSPACE}/{test_script}"
    test_cmd = [
        'export PATH="$HOME/.cargo/bin:$PATH"',
        f"export VLLM_ENABLE_V1_MULTIPROCESSING=0",
        f"export VERILM_CAPTURE=1",
        f"cd {WORKSPACE} && python -c \""
        f"import sys; sys.path.insert(0, '.'); "
        f"exec(open('{remote_script}').read()); "
        f"result = _run_test() if '_run_test' in dir() else _run_e2e(); "
        f"import sys; sys.exit(0 if result.get('passed', result.get('n_failures', 1) == 0) else 1)"
        f"\"",
    ]

    print(f"\nRunning test: {test_script}")
    print("=" * 60)
    rc = run_ssh(state, test_cmd, stream=True)
    print("=" * 60)

    if rc == 0:
        print("\nTEST PASSED")
    else:
        print(f"\nTEST FAILED (exit code {rc})")
    return rc


def main():
    parser = argparse.ArgumentParser(description="Run verilm tests on RunPod")
    parser.add_argument("--terminate", action="store_true", help="Terminate the pod")
    parser.add_argument("--fresh", action="store_true", help="Force a new pod")
    parser.add_argument("--script", default="scripts/modal/test_sampled_decoding.py",
                        help="Test script to run (default: sampled decoding)")
    args = parser.parse_args()

    api_key = get_api_key()

    if args.terminate:
        terminate_pod(api_key)
        return

    state = load_pod_state()

    # Check if existing pod is still alive.
    if state and not args.fresh:
        import runpod
        runpod.api_key = api_key
        try:
            pod_info = runpod.get_pod(state["pod_id"])
            if pod_info.get("desiredStatus") != "RUNNING":
                print(f"Pod {state['pod_id']} is not running. Creating new one...")
                state = None
        except Exception:
            print("Previous pod not found. Creating new one...")
            state = None

    if state is None or args.fresh:
        if args.fresh and state:
            terminate_pod(api_key)
        state = create_pod(api_key)

    # Wait a moment for SSH to be ready.
    print("Waiting for SSH...", end="", flush=True)
    for _ in range(30):
        result = subprocess.run(
            ssh_cmd(state) + ["echo ok"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(" ready.")
            break
        print(".", end="", flush=True)
        time.sleep(3)
    else:
        print("\nERROR: SSH not reachable after 90s")
        sys.exit(1)

    if not state.get("setup_done"):
        setup_pod(state)

    sync_code(state)
    rc = build_and_test(state, args.script)
    sys.exit(rc)


if __name__ == "__main__":
    main()
