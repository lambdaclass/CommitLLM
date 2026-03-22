# Roadmap

## Key Open Questions

- **The Requantization Corridor:** How much do FP16 and FP64 attention outputs differ when quantized to INT8? This determines how tightly attention manipulation is constrained. Without empirical measurement, we don't know the actual bound on adversarial freedom in the attention interior.

## Practical Notes

- **60-second audit window is viable:** For Llama 70B on 4×H100, this requires ~160 GB RAM (8% of a 2 TB system). This is generous time for automated auditing (receive response → decide → send challenge) with no NVMe spill or operational complexity.

---

## Launch
- [ ] Write article for blog.lambdaclass.com
- [ ] Write X thread
- [ ] Upload paper to arXiv
- [ ] Squash git history to a single commit before making repo public
- [ ] Make repo public

## Foundation
- [ ] Verifier library (Rust): 25 MB key + receipt + traces → pass/fail
- [ ] vLLM tracing plugin: sidecar that captures intermediates and emits receipts
- [ ] llama.cpp tracing plugin
- [ ] Requantization corridor measurement: FP16 vs FP64 attention on real Llama activations, measure INT8 agreement rates
- [ ] Formalization in Lean

## First product
- [ ] OpenAI-compatible proxy with receipts: sits in front of vLLM, standard API, receipts in response headers/metadata

## Integrations
- [ ] Decentralized inference network plugin (Ritual, Bittensor, Gensyn): replace redundant execution with receipts

## Tools
- [ ] Inference marketplace: multiple providers, same model, client-verified receipts
