# Attack Matrix

This file tracks adversarial coverage separately from ordinary correctness tests.

## Implemented Campaigns

| Attack class | Concrete scenarios | Status | Current runner |
|---|---|---|---|
| Receipt / audit tampering | Merkle root, IO root, token count, seed commitment, prompt bytes | Implemented | `redteam/modal/test_adversarial.py` |
| Shell-opening forgery | Tampered `attn_out`, `ffn_out`, `g`, `u`, `q`, `k`, `v` | Implemented | `redteam/modal/test_adversarial.py` |
| Retained-state tampering | Tampered `a`, `scale_a`, `scale_x_attn`, `scale_x_ffn`, `scale_h`, final residual | Implemented | `redteam/modal/test_adversarial.py` |
| Decode / output tampering | Wrong `token_id`, wrong revealed seed, manifest-temperature mismatch | Implemented | `redteam/modal/test_adversarial.py` |
| Cross-request splice | Shell splice, retained splice, token splice, layer swaps | Implemented | `redteam/modal/test_adversarial.py` |
| Prefix tampering | Prefix leaf hash tamper, prefix token-id swap, token-index shift | Implemented | `redteam/modal/test_adversarial.py` |
| Cross-model substitution | Serve Qwen audit, verify under Llama key | Implemented | `redteam/modal/test_model_substitution.py` |

## Partial Coverage

| Attack class | Current state | Gap to close |
|---|---|---|
| Selective layer cheating | Local Rust tests already exercise fake-attention and bridge-boundary attacks | Add explicit real-GPU campaign variants that cheat only selected opened and unopened layers |
| KV provenance attacks | Prefix and proof tampering are covered | Add committed-KV injection and "self-consistent fake prefix" campaigns on real GPU |
| Downgrade / unsupported-path attacks | Boundary and cross-version tests cover parts of this locally | Add explicit red-team cases for unknown versions, missing fields, and unsupported sampler/model paths |
| Receipt replay / freshness | Real-GPU probe exists and demonstrates the current limitation | Add verifier-issued freshness binding so replay of an old honest receipt stops verifying |

## Planned Campaigns

| Attack class | Goal |
|---|---|
| Smaller-model substitution | Try to serve a materially smaller or cheaper model behind a receipt/key meant for a larger model |
| Parser fuzzing | Coverage-guided fuzzing of receipt and audit binary parsers; malformed inputs must fail closed |
| Challenge-shaping attacks | Try to exploit predictable or reused challenge structure to precompute passing cheats |
| Retention-horizon abuse | See what a dishonest provider can evade by delaying or denying audit after the retention window |

## Red-Team Standard

Every attack should end in one of two states:

1. The verifier rejects it, with a specific expected failure reason.
2. The attack exposes a real accepted gap, and the gap is documented with mitigation.

Silent assumptions and "probably fine" are not acceptable outcomes here.
