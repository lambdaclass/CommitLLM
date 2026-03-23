# Roadmap

## 1. Research & Validation

First, we need empirical data to validate the protocol's security bounds.

- [ ] **Measure the requantization corridor** — Run FP16 vs FP64 attention on real Llama activations, quantize both to INT8, measure per-element agreement rates. This determines the actual bound on adversarial freedom in attention manipulation.
- [ ] **Calibrate detection probabilities** — Concrete numbers for P(catch) at various sampling rates (k) and tampering fractions (f/n).
- [ ] **Storage cost benchmarks** — Real measurements across audit windows (30s, 60s, 2min, 1hr) and model sizes.
- [ ] **Inference overhead measurement** — Quantify the runtime cost of tracing and commitment generation (memcpy overhead, Merkle tree construction, etc.)

## 2. Core Implementation

The foundation everything else builds on.

### 2.1 Verifier
- [ ] **Rust verifier library** — 25 MB key + receipt + traces → pass/fail
  - Freivalds checks (all 7 matrices)
  - Exact requantization verification
  - SiLU LUT, RoPE, RMSNorm canonical recomputation
  - Attention replay (FP64)
  - KV provenance sampling
  - Cross-layer consistency checks
- [ ] **Client SDKs** — Python and TypeScript libraries for verifying receipts (wrapping the Rust core)

### 2.2 Tracing Plugins
- [ ] **vLLM tracing plugin** — Sidecar that captures intermediates and emits 100-byte receipts
- [ ] **llama.cpp tracing plugin** — Same for llama.cpp deployments
- [ ] **Fine-tuned models / LoRA support** — How to handle adapters (separate commitment vs merged weights)

### 2.3 Formal Verification
- [ ] **Formalization in Lean** — Prove correctness of Freivalds implementation and protocol composition

### 2.4 Testing & Quality
- [ ] **Property-based test suite** — Cheat detection should always catch manipulation; fuzzing the verifier
- [ ] **Batch verification** — Optimizations for verifying many receipts (e.g., a day's traffic)

### 2.5 Standardization
- [ ] **Receipt format spec** — Documented schema for the 100-byte receipt and all commitment structures
- [ ] **API extensions** — OpenAI-compatible `receipt` field in response headers/metadata

## 3. Launch & Documentation

Ship the protocol and explain it clearly.

### 3.1 Paper
- [ ] Add graphs and tables with concrete numbers (storage costs, audit windows, detection probabilities)
- [ ] Add security game + proof sketches for the core soundness claims
- [ ] Add batched-serving discussion (per-request trace isolation under vLLM / paged attention)
- [ ] Add adversarial model section — Formal threat model describing provider capabilities, cheating strategies, and protocol guarantees against each
- [ ] Cut a versioned paper release in GitHub
- [ ] Upload to arXiv
- [ ] Claim / create Hugging Face Paper Page and link repo + artifacts

### 3.2 Demo & Visualization
- [ ] **Interactive TUI demo** — Terminal interface showing the protocol in action
  - User chats with "provider" (simulated or real)
  - Each response shows 100-byte receipt
  - User can manually audit any response
  - Random automatic audits happen in background
  - Provider randomly cheats (model swap, attention manipulation, KV tampering)
  - Visual feedback: green check for honest, red alert with evidence for caught cheating
  - Shows detection probability in real-time
- [ ] **Hugging Face Space demo** — Public artifact linked from the paper page
  - Receipt visualization
  - One-click audit walkthrough
  - Example cheating scenarios and caught audits

### 3.3 Communication
- [ ] Write article for blog.lambdaclass.com
- [ ] Write X thread
- [ ] Prepare publication rollout: GitHub release → arXiv → Hugging Face Paper Page → demo links
- [ ] Make repo public (squash git history to single commit first)

## 4. First Product

Something people can actually use.

- [ ] **OpenAI-compatible proxy with receipts** — Sits in front of vLLM, standard API, receipts in response headers/metadata
  - 60-second default audit window (fits in RAM, no NVMe complexity)
  - Programmatic challenge endpoint
  - Simple verifier CLI

## 5. Ecosystem

Where this gets interesting.

### 5.1 Decentralized Inference
- [ ] **Ritual plugin** — Replace redundant execution with receipts
- [ ] **Bittensor plugin** — Verified subnet incentives
- [ ] **Gensyn plugin** — Proof-of-training integration

### 5.2 Marketplace
- [ ] **Inference marketplace** — Multiple providers, same model, client-verified receipts, price/quality competition

---

## Critical Path

```
Research (requantization corridor)
    ↓
Core Implementation (verifier + vLLM plugin)
    ↓
Launch (paper + blog + public repo)
    ↓
First Product (OpenAI-compatible proxy)
    ↓
Ecosystem (integrations + marketplace)
```

## Open Questions

| Question | Why it matters | Blocker for |
|----------|--------------|-------------|
| How wide is the requantization corridor? | Determines actual security bound for attention manipulation | Protocol security claims |
| Is 60s audit window acceptable to providers? | Affects storage architecture (RAM-only vs NVMe spill) | Product defaults |
| Can attention replay be batched efficiently? | Affects verifier CPU cost at long context | Scalability |
