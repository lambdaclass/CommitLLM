# Red Team

This directory is for adversarial work against CommitLLM's trust boundary.

It is intentionally separate from ordinary test suites.

- Correctness testing asks: does an honest provider pass?
- Red-team testing asks: can a dishonest provider cheat and still pass?

If a red-team attack succeeds, that is not "just another failing test." It is a protocol or verifier gap that must either be fixed or documented explicitly as an accepted limitation with mitigation.

## Scope

The red-team surface should try to break, at minimum:

- Receipt and audit binding
- Transcript chaining and anti-splice guarantees
- Shell openings and retained-state integrity
- Decode/output binding
- Prefix/KV provenance
- Model identity and weight substitution
- Versioning / downgrade / unsupported-path fail-closed behavior
- Parser and binary-decoding robustness

## Layout

- `attack_matrix.md`: living inventory of attack classes, current coverage, and open gaps
- `modal/test_adversarial.py`: real-GPU cheating-provider campaign against the current verifier
- `modal/test_model_substitution.py`: real-GPU cross-model substitution campaign
- `modal/test_freshness_gap.py`: explicit probe for the current replay/freshness limitation

## Running

- `make redteam-gpu`
- `make redteam-model-substitution`
- `make redteam-freshness-gap`
- `modal run --detach redteam/modal/test_adversarial.py`

Some red-team runners are expected-rejection campaigns.
Some are limitation probes that confirm an accepted gap exists today.

- Rejection campaign outcome: the verifier must fail.
- Limitation probe outcome: the script succeeds by demonstrating the gap clearly.

## Rule

Before strong security claims or final benchmark claims land, adversarial coverage should be expanded here, not hidden inside unrelated unit tests.
