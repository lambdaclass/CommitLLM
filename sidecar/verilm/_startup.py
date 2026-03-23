"""
Auto-patching module loaded at Python interpreter startup via .pth file.

Installs an import hook that wraps vllm._custom_ops.cutlass_scaled_mm
for input capture. This ensures the patch is active in ALL processes,
including vLLM's TP worker subprocesses (which don't load plugins).

The .pth file in site-packages contains:
    import verilm._startup

This runs before any user code, guaranteeing the hook is installed
before vLLM imports its custom ops.
"""

import importlib
import importlib.abc
import importlib.util
import os
import sys


def _should_activate():
    """Check if capture should be active in this process."""
    if os.environ.get("VI_CAPTURE", "0") == "0":
        if "vi_capture" not in os.environ.get("VLLM_PLUGINS", ""):
            return False
    return True


def _apply_patch(ops_module):
    """Wrap ops_module.cutlass_scaled_mm with capture logic."""
    # Import lazily to avoid circular imports at startup
    from verilm import capture as cap_mod

    original = getattr(ops_module, "cutlass_scaled_mm", None)
    if original is None or original is cap_mod._wrapped_cutlass_scaled_mm:
        return  # Already patched or not available

    cap_mod._real_kernel[0] = original
    ops_module.cutlass_scaled_mm = cap_mod._wrapped_cutlass_scaled_mm
    cap_mod._patched = True

    import logging

    logging.getLogger("verilm").info(
        "verilm: patched cutlass_scaled_mm via import hook (pid=%d)",
        os.getpid(),
    )


class _PatchingLoader:
    """Wraps a real loader to apply the capture patch after module init."""

    def __init__(self, real_loader):
        self._real = real_loader

    def create_module(self, spec):
        if hasattr(self._real, "create_module"):
            return self._real.create_module(spec)
        return None

    def exec_module(self, module):
        self._real.exec_module(module)
        _apply_patch(module)


class _CutlassScaledMmPatchFinder(importlib.abc.MetaPathFinder):
    """Intercepts import of vllm._custom_ops to wrap cutlass_scaled_mm.

    On first import of vllm._custom_ops:
    1. Removes itself from sys.meta_path to avoid recursion
    2. Finds the real module spec via the standard import machinery
    3. Wraps the loader so _apply_patch runs after exec_module
    4. Returns the modified spec (one-shot, does not re-add itself)
    """

    def find_spec(self, fullname, path, target=None):
        if fullname != "vllm._custom_ops":
            return None

        # Remove self to let the real finder handle this import
        try:
            sys.meta_path.remove(self)
        except ValueError:
            return None

        try:
            real_spec = importlib.util.find_spec(fullname)
        except (ModuleNotFoundError, ValueError):
            # vLLM not installed — nothing to patch
            return None

        if real_spec is None or real_spec.loader is None:
            return None

        # Wrap the loader so we patch after the real module initializes
        real_spec.loader = _PatchingLoader(real_spec.loader)
        return real_spec


# Install the finder if capture is requested
if _should_activate():
    sys.meta_path.insert(0, _CutlassScaledMmPatchFinder())
