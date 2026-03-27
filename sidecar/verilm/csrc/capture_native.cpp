// Native capture accumulator for verified inference — v3 hybrid.
//
// Two modes of operation:
//
//   1. Accumulator mode (hybrid): Python wrapper calls the real kernel
//      directly (fastest path), then calls record() to push scale refs
//      into a C++ vector. Drain uses C++ at::cat for bulk transfer.
//
//   2. Full-replace mode: forward() replaces ops.cutlass_scaled_mm
//      entirely, calling the original kernel via typed C++ dispatcher.
//      (Kept for A/B comparison; accumulator mode is faster.)
//
// Why hybrid wins: the c10 typed dispatcher adds ~5-8us/call overhead
// vs calling vLLM's Python-bound op directly. With 28K calls/request,
// that's 140-224ms extra. Accumulator mode avoids this by only using
// C++ for storage + drain (where at::cat on std::vector is 3-4x faster
// than Python torch.cat on a Python list of 28K tensors).

#include <torch/extension.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <vector>

namespace {

// ---- Original op handle (for full-replace mode only) ----
c10::optional<c10::OperatorHandle> g_orig_op;

// ---- Geometry ----
int64_t g_calls_per_fwd = 0;
int64_t g_projs_per_layer = 4;
int64_t g_o_proj_idx = 1;

// ---- State ----
int64_t g_call_counter = 0;
int64_t g_total_captured = 0;
bool g_enabled = false;

// ---- Scale capture: vector of tensor refs ----
// Each scale_a is pushed by reference (refcount bump only, ~0.1us).
// At drain: at::cat -> .cpu() in one shot.
std::vector<at::Tensor> g_scales;

// ---- o_proj activation capture: pinned CPU slab ----
at::Tensor g_pinned_slab;
int64_t g_slab_idx = 0;
std::vector<std::pair<int64_t, int64_t>> g_slab_offsets;

}  // namespace


// ---- Initialization ----

void init(int64_t calls_per_fwd, int64_t projs_per_layer, int64_t o_proj_idx) {
    // Try to find the original op (needed for full-replace mode only).
    auto op_opt = c10::Dispatcher::singleton().findSchema({"_C::cutlass_scaled_mm", ""});
    if (op_opt.has_value()) {
        g_orig_op = *op_opt;
    }

    g_calls_per_fwd = calls_per_fwd;
    g_projs_per_layer = projs_per_layer;
    g_o_proj_idx = o_proj_idx;
    g_call_counter = 0;
    g_total_captured = 0;
    g_enabled = false;
    g_scales.clear();
    g_scales.reserve(65536);
    g_slab_idx = 0;
    g_slab_offsets.clear();
}

void init_slab(const at::Tensor& slab) {
    g_pinned_slab = slab;
    g_slab_idx = 0;
    g_slab_offsets.clear();
}

void set_enabled(bool val) { g_enabled = val; }
void reset_counter() { g_call_counter = 0; }
int64_t get_call_counter() { return g_call_counter; }
int64_t get_total_captured() { return g_total_captured; }
int64_t get_scale_count() { return static_cast<int64_t>(g_scales.size()); }


// ---- Accumulator mode: record() + copy_slab() ----
//
// Called from the Python wrapper AFTER the real kernel returns.
// Python wrapper handles the kernel call (fastest path).
// We handle scale storage + slab copy (C++ vector + at::cat at drain).

bool record(const at::Tensor& scale_a) {
    // Counter arithmetic + scale push. Returns true if this is an o_proj call.
    auto idx = g_call_counter % g_calls_per_fwd;
    auto proj_idx = idx % g_projs_per_layer;
    g_call_counter++;
    g_total_captured++;

    // Flatten to 1D so at::cat works even when prefill scales are 2D (N,1)
    // and decode scales are 1D (1,). flatten() is a view — no data copy.
    g_scales.push_back(scale_a.flatten());

    return proj_idx == g_o_proj_idx;
}

void copy_slab(const at::Tensor& a) {
    // Copy o_proj activation into pinned CPU slab (GPU->CPU async).
    if (!g_pinned_slab.defined()) return;
    auto batch_sz = a.size(0);
    auto end = g_slab_idx + batch_sz;
    if (end <= g_pinned_slab.size(0)) {
        g_pinned_slab.narrow(0, g_slab_idx, batch_sz).copy_(a, /*non_blocking=*/true);
        g_slab_offsets.emplace_back(g_slab_idx, end);
        g_slab_idx = end;
    }
}


// ---- Full-replace mode: forward() ----
//
// Replaces ops.cutlass_scaled_mm entirely. Calls original kernel via
// typed C++ dispatcher. Slower than accumulator mode due to dispatcher
// overhead (~5-8us/call), but kept for A/B comparison.

at::Tensor forward(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    at::ScalarType out_dtype,
    const c10::optional<at::Tensor>& bias
) {
    TORCH_CHECK(g_orig_op.has_value(),
        "capture_native: not initialized (call init() first)");

    auto out = at::empty({a.size(0), b.size(1)},
        at::TensorOptions().dtype(out_dtype).device(a.device()));

    g_orig_op->typed<void(at::Tensor&, const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, const at::Tensor&,
                          const c10::optional<at::Tensor>&)>()
        .call(out, a, b, scale_a, scale_b, bias);

    if (!g_enabled) return out;

    auto idx = g_call_counter % g_calls_per_fwd;
    auto proj_idx = idx % g_projs_per_layer;
    g_call_counter++;
    g_total_captured++;

    g_scales.push_back(scale_a.flatten());

    if (proj_idx == g_o_proj_idx && g_pinned_slab.defined()) {
        auto batch_sz = a.size(0);
        auto end = g_slab_idx + batch_sz;
        if (end <= g_pinned_slab.size(0)) {
            g_pinned_slab.narrow(0, g_slab_idx, batch_sz).copy_(a, /*non_blocking=*/true);
            g_slab_offsets.emplace_back(g_slab_idx, end);
            g_slab_idx = end;
        }
    }

    return out;
}


// ---- Drain: retrieve captured data ----

std::tuple<at::Tensor, std::vector<int64_t>, int64_t> drain() {
    auto count = static_cast<int64_t>(g_scales.size());
    at::Tensor scales;

    if (count > 0) {
        auto cat = at::cat(g_scales);
        if (cat.numel() != count) {
            // Multi-element prefill scales: reduce each to max.
            std::vector<at::Tensor> reduced;
            reduced.reserve(g_scales.size());
            for (auto& s : g_scales) {
                if (s.numel() > 1) {
                    reduced.push_back(s.max().unsqueeze(0));
                } else {
                    reduced.push_back(s);
                }
            }
            cat = at::cat(reduced);
        }
        scales = cat.cpu();
    } else {
        scales = at::empty({0}, at::kFloat);
    }

    std::vector<int64_t> offsets_flat;
    offsets_flat.reserve(g_slab_offsets.size() * 2);
    for (const auto& [s, e] : g_slab_offsets) {
        offsets_flat.push_back(s);
        offsets_flat.push_back(e);
    }

    g_scales.clear();
    g_slab_idx = 0;
    g_slab_offsets.clear();

    return {scales, offsets_flat, count};
}

void drain_discard() {
    g_scales.clear();
    g_slab_idx = 0;
    g_slab_offsets.clear();
}


// ---- Drain helper: accept external scale list for fast C++ cat ----
//
// Used with Rust CaptureHook: Rust hook stores scales during generation
// (fast PyO3 path), then at drain time Python passes the list to C++ for
// bulk at::cat + .cpu() (3-4x faster than Python torch.cat on 28K tensors).

std::tuple<at::Tensor, int64_t> cat_scales(std::vector<at::Tensor> scale_list) {
    auto count = static_cast<int64_t>(scale_list.size());
    if (count == 0) {
        return {at::empty({0}, at::kFloat), 0};
    }

    // Flatten to 1D — prefill scales can be 2D (N,1), decode are 1D (1,).
    std::vector<at::Tensor> flat;
    flat.reserve(count);
    for (auto& s : scale_list) {
        flat.push_back(s.dim() == 1 ? s : s.flatten());
    }

    auto cat = at::cat(flat);
    if (cat.numel() != count) {
        // Multi-element prefill scales: reduce each to max.
        std::vector<at::Tensor> reduced;
        reduced.reserve(count);
        for (auto& s : flat) {
            if (s.numel() > 1) {
                reduced.push_back(s.max().unsqueeze(0));
            } else {
                reduced.push_back(s);
            }
        }
        cat = at::cat(reduced);
    }

    return {cat.cpu(), count};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init, "Initialize native capture accumulator");
    m.def("init_slab", &init_slab, "Set pinned CPU slab for o_proj copies");
    m.def("set_enabled", &set_enabled, "Enable/disable capture");
    m.def("reset_counter", &reset_counter, "Reset call counter between requests");
    m.def("get_call_counter", &get_call_counter);
    m.def("get_total_captured", &get_total_captured);
    m.def("get_scale_count", &get_scale_count);
    // Accumulator mode: called from Python wrapper.
    m.def("record", &record,
          "Record scale + advance counter. Returns true if o_proj.",
          py::arg("scale_a"));
    m.def("copy_slab", &copy_slab,
          "Copy o_proj activation to pinned slab",
          py::arg("a"));
    // Full-replace mode: replaces ops.cutlass_scaled_mm.
    m.def("forward", &forward,
          "Wrapped cutlass_scaled_mm with native capture bookkeeping",
          py::arg("a"), py::arg("b"), py::arg("scale_a"), py::arg("scale_b"),
          py::arg("out_dtype") = at::ScalarType::BFloat16,
          py::arg("bias") = py::none());
    m.def("drain", &drain, "Drain scales (CPU tensor) + slab offsets + count");
    m.def("drain_discard", &drain_discard, "Discard captured data without transfer");
    // Drain helper: C++ cat for externally-collected scale lists.
    m.def("cat_scales", &cat_scales,
          "Bulk cat + cpu transfer for a list of scale tensors",
          py::arg("scale_list"));
}
