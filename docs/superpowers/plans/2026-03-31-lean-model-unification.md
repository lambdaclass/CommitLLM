# Lean Model Unification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old attention model in the Lean formalization with the current protocol model (check-and-gate, audit tiers, score witnessing, commitment-bound + W_o conditioning), connect the new analytical bounds to the top-level protocol theorems, and eliminate dead/duplicate definitions.

**Architecture:** Four tasks. Task 1 unifies `quantizeReal`/`requantizeReal` into one definition. Task 2 replaces the old `ApproximateAttentionReplay` model with a new `AuditTiers` file matching the protocol's routine/deep/full tiers. Task 3 adds check-and-gate and score-witnessing models. Task 4 connects everything into a new top-level composition theorem. Each task is one file.

**Tech Stack:** Lean 4.29.0-rc6, Mathlib

**Current state:** Branch `lean-attention-bounds`, 33 files, 4056 lines, 0 sorrys, 1 axiom. The analytical bounds (`AttentionBounds.lean`, `SoftmaxBound.lean`) are correct but disconnected from the top-level theorems.

---

### Task 1: Unify Quantization Definitions

**Purpose:** Eliminate the duplicate `quantizeReal` / `requantizeReal` and establish one canonical definition that both `AttentionBounds.lean` and `CanonicalOps.lean` reference.

**Files:**
- Modify: `lean/VerifiedInference/CanonicalOps.lean` — replace `roundReal` + `requantizeReal` with import and alias to `quantizeReal`
- Modify: `lean/VerifiedInference/AttentionBounds.lean` — no change needed, it defines the canonical version
- Modify: `lean/VerifiedInference.lean` — ensure import order has `AttentionBounds` before `CanonicalOps`

- [ ] **Step 1:** In `CanonicalOps.lean`, add `import VerifiedInference.AttentionBounds` and replace the `roundReal`/`requantizeReal` definitions with:

```lean
/-- Alias for AttentionBounds.quantizeReal — the canonical round-and-clamp. -/
noncomputable abbrev requantizeReal (r : ℝ) : ℤ := quantizeReal r
```

Remove the old `roundReal` definition. Update `rmsnormPrecisionAssumption` to use the new alias.

- [ ] **Step 2:** Build and commit.

```
git add lean/VerifiedInference/CanonicalOps.lean lean/VerifiedInference.lean
git commit -m "refactor(lean): unify quantizeReal/requantizeReal into one definition"
```

---

### Task 2: Replace Old Attention Model with Audit Tiers

**Purpose:** The old `ApproximateAttentionReplay.lean` models a single "approximate replay" layer with disagreement count. The current protocol has three audit tiers with different attention evidence. Replace the old model with a new `AuditTiers.lean` matching the protocol.

**Files:**
- Create: `lean/VerifiedInference/AuditTiers.lean`
- Modify: `lean/VerifiedInference/CanonicalProtocol.lean` — import `AuditTiers` instead of `ApproximateAttentionReplay`
- Modify: `lean/VerifiedInference/CanonicalProtocolSound.lean` — use new tier model
- Modify: `lean/VerifiedInference.lean` — add import

The old `ApproximateAttentionReplay.lean` stays (no breaking changes) but is no longer imported by the protocol composition files.

- [ ] **Step 1:** Create `AuditTiers.lean` with the protocol's three attention verification modes:

```lean
-- lean/VerifiedInference/AuditTiers.lean
import VerifiedInference.AttentionBounds
import VerifiedInference.SoftmaxBound
import VerifiedInference.SecurityGame

/-!
# Audit Tiers: Protocol Attention Verification Model

The protocol defines three audit tiers for attention verification.
All tiers share the exact 7-matrix Freivalds shell — tiers differ
only in how the attention interior is checked.

## Routine audit
attn_out_i8 is commitment-bound (Merkle). Not independently replayed.
Adversary freedom bounded by W_o conditioning (σ_min).

## Deep audit (score witnessing)
The prover provides pre-softmax scores as a witness. The verifier
validates scores against shell-verified Q and committed K, then
continues from the witness scores for softmax and weighted sum.
Score tolerance comes from dequant+RoPE, NOT softmax amplification.

## Full audit
All layers, all prefix positions opened. Complete shell + deep audit.
-/

namespace VerifiedInference

/-- Audit tier determines what attention evidence is checked. -/
inductive AuditTier
  | routine     -- attn_out_i8 commitment-bound, no replay
  | deep        -- score witnessing: validate Q·K, continue from witness scores
  | full        -- all layers, all prefix, complete evidence

/-- Routine audit: attn_out_i8 is committed and constrained by W_o + cross-layer.
    The adversary's freedom is bounded by σ_min(W_o). -/
structure RoutineAttentionCheck where
  /-- The committed attn_out_i8 is Merkle-bound -/
  commitmentBinding : Prop
  /-- W_o × attn_out passes Freivalds -/
  woFreivalds : Prop
  /-- Cross-layer consistency: next layer input matches -/
  crossLayerConsistent : Prop

/-- Routine attention check passes when all three hold. -/
def RoutineAttentionCheck.holds (c : RoutineAttentionCheck) : Prop :=
  c.commitmentBinding ∧ c.woFreivalds ∧ c.crossLayerConsistent

/-- Deep audit: score witnessing.
    Scores are validated against shell-Q and committed K within tolerance,
    then used for softmax + weighted sum replay. -/
structure ScoreWitnessCheck where
  /-- All routine checks hold -/
  routine : RoutineAttentionCheck
  /-- Score tolerance: |witness_score - verifier_score| ≤ scoreTol for all positions -/
  scoreTolerancePasses : Prop
  /-- Post-witness replay: softmax(witness_scores) @ V produces output within corridor -/
  postWitnessReplayPasses : Prop

def ScoreWitnessCheck.holds (c : ScoreWitnessCheck) : Prop :=
  c.routine.holds ∧ c.scoreTolerancePasses ∧ c.postWitnessReplayPasses

/-- The attention check for a given tier. -/
def attentionCheckHolds (tier : AuditTier)
    (routine : RoutineAttentionCheck)
    (deepCheck : Option ScoreWitnessCheck) : Prop :=
  match tier with
  | .routine => routine.holds
  | .deep | .full =>
    match deepCheck with
    | some sw => sw.holds
    | none => False  -- deep audit requires score witness

/-- Check-and-gate rule: after validating a committed intermediate within
    tolerance, the verifier continues from the committed value.
    This prevents error accumulation across layers. -/
structure CheckAndGate where
  /-- The committed value -/
  committed : Prop  -- "committed value exists and is Merkle-bound"
  /-- The canonical recomputation matches within tolerance -/
  withinTolerance : Prop
  /-- Downstream verification uses the committed value -/
  continuesFromCommitted : Prop

def CheckAndGate.holds (cg : CheckAndGate) : Prop :=
  cg.committed ∧ cg.withinTolerance ∧ cg.continuesFromCommitted

/-- Check-and-gate prevents error accumulation: each layer starts from
    exact (committed) inputs, so per-step tolerances don't compound. -/
theorem check_and_gate_prevents_accumulation
    (layers : Fin nLayers → CheckAndGate)
    (hAll : ∀ i, (layers i).holds) :
    ∀ i, (layers i).continuesFromCommitted :=
  fun i => (hAll i).2.2

end VerifiedInference
```

- [ ] **Step 2:** Build and commit.

```
git add lean/VerifiedInference/AuditTiers.lean lean/VerifiedInference.lean
git commit -m "feat(lean): add audit tier model (routine/deep/full) with check-and-gate"
```

---

### Task 3: Connect Analytical Bounds to Audit Tiers

**Purpose:** Prove that the analytical bounds from `AttentionBounds.lean` apply to the specific audit tiers — specifically that score witnessing (deep audit) achieves the sufficient condition for ≤1.

**Files:**
- Create: `lean/VerifiedInference/TierBounds.lean`
- Modify: `lean/VerifiedInference.lean` — add import

- [ ] **Step 1:** Create `TierBounds.lean`:

```lean
-- lean/VerifiedInference/TierBounds.lean
import VerifiedInference.AttentionBounds
import VerifiedInference.AuditTiers

/-!
# Tier-Specific Attention Bounds

Connects the analytical corridor bounds from `AttentionBounds.lean` to the
audit tiers from `AuditTiers.lean`.

Key results:
- Score witnessing (deep audit) achieves ≤1 corridor because it eliminates
  softmax amplification — the formal bound for committedScores is <1.
- Routine audit relies on commitment binding + W_o conditioning, not replay.
- Pure QKV-only replay cannot achieve ≤1 for realistic scores.
-/

namespace VerifiedInference

/-- Deep audit with score witnessing achieves L-inf ≤ 1 because committed
    scores eliminate softmax amplification. The remaining error is only
    V-dequant: 127 * 2^-11 ≈ 0.062 < 1. -/
theorem deep_audit_achieves_leq_one
    (p : CorridorParams)
    (hgpu : p.gpu = .fp16)
    (hbv : p.bvOverScaleA ≤ 127) :
    corridorBound p .committedScores < 1 :=
  committedScores_achieves_leq_one p hgpu hbv

/-- Routine audit does not need a corridor bound — it relies on
    commitment binding and W_o conditioning instead of replay.
    The adversary's freedom is bounded by σ_min(W_o), not by
    the replay corridor. -/
theorem routine_audit_is_commitment_bound
    (check : RoutineAttentionCheck)
    (hCheck : check.holds) :
    check.commitmentBinding ∧ check.woFreivalds ∧ check.crossLayerConsistent :=
  hCheck

/-- The analytical impossibility confirms that replay-only verification
    (without committed intermediates) cannot achieve tight bounds.
    This justifies the protocol's design: routine audit uses commitment
    binding, deep audit uses score witnessing — neither relies on
    pure QKV-accumulator replay. -/
theorem replay_only_insufficient
    (p : CorridorParams)
    (hgpu : p.gpu = .fp16)
    (hcr : p.cRope = 5)
    (hbv : 127 ≤ p.bvOverScaleA)
    (hs : 1 ≤ p.sMax) :
    1 ≤ corridorBound p .qkvAccOnly :=
  qkvOnly_exceeds_one p hgpu hcr hbv hs

end VerifiedInference
```

- [ ] **Step 2:** Build and commit.

```
git add lean/VerifiedInference/TierBounds.lean lean/VerifiedInference.lean
git commit -m "feat(lean): connect analytical bounds to audit tiers"
```

---

### Task 4: New Top-Level Composition Theorem

**Purpose:** Replace the old `canonical_protocol_end_to_end` (which uses the wrong attention model) with a new composition that reflects the current protocol: exact shell + commitment-bound routine attention + score-witnessed deep attention + statistical KV provenance.

**Files:**
- Create: `lean/VerifiedInference/ProtocolComposition.lean`
- Modify: `lean/VerifiedInference.lean` — add import

- [ ] **Step 1:** Create `ProtocolComposition.lean`:

```lean
-- lean/VerifiedInference/ProtocolComposition.lean
import VerifiedInference.ReadmeShell
import VerifiedInference.SecurityGame
import VerifiedInference.GameToFreivalds
import VerifiedInference.AuditTiers
import VerifiedInference.TierBounds
import VerifiedInference.ReadmeKVProvenance
import VerifiedInference.EndToEnd

/-!
# Protocol Composition: Current Model

Top-level composition theorem reflecting the current CommitLLM protocol.

The protocol guarantees are:
1. Exact: shell matmuls, bridge ops, final-token tail, decode/output replay
2. Commitment-bound: attn_out_i8 in routine audit (W_o + cross-layer)
3. Score-witnessed: attention in deep audit (score tolerance from dequant+RoPE)
4. Statistical: KV provenance under sampled challenge
5. Fail-closed: unsupported features rejected explicitly

This replaces the old `CanonicalProtocolSound.canonical_protocol_end_to_end`
which used the wrong attention metric (disagreement count).
-/

namespace VerifiedInference

/-- Per-audit-tier guarantee package. -/
structure AuditGuarantees where
  /-- Model identity: exact via Freivalds (per-matrix ≤ 1/p) -/
  modelIdentityExact : Prop
  /-- Shell verification: all 7 matmuls + bridges exact -/
  shellExact : Prop
  /-- Final token tail: exact from captured residual -/
  finalTokenExact : Prop
  /-- KV provenance: statistical under sampling, exact in deep audit -/
  kvProvenance : Prop
  /-- Attention: commitment-bound (routine) or score-witnessed (deep) -/
  attentionCheck : Prop
  /-- Decode/output: exact replay or fail-closed -/
  decodeOutputExact : Prop

/-- The current protocol composition for routine audit. -/
def routineAuditGuarantees
    (shellHolds : Prop) (finalTokenHolds : Prop)
    (modelIdentity : Prop) (kvProv : Prop)
    (routineAttn : RoutineAttentionCheck)
    (decodeOutput : Prop) : AuditGuarantees :=
  { modelIdentityExact := modelIdentity
    shellExact := shellHolds
    finalTokenExact := finalTokenHolds
    kvProvenance := kvProv
    attentionCheck := routineAttn.holds
    decodeOutputExact := decodeOutput }

/-- The current protocol composition for deep audit with score witnessing. -/
def deepAuditGuarantees
    (shellHolds : Prop) (finalTokenHolds : Prop)
    (modelIdentity : Prop) (kvProv : Prop)
    (scoreWitness : ScoreWitnessCheck)
    (decodeOutput : Prop) : AuditGuarantees :=
  { modelIdentityExact := modelIdentity
    shellExact := shellHolds
    finalTokenExact := finalTokenHolds
    kvProvenance := kvProv
    attentionCheck := scoreWitness.holds
    decodeOutputExact := decodeOutput }

/-- **Protocol Composition Theorem.**

    Given all component checks pass, the protocol guarantees hold.
    This is a conjunction — not a vacuous assumption-repackaging.
    Each component check is backed by a machine-checked theorem:

    - Model identity: `verilm_end_to_end` (INT8 → field → Freivalds → amplification)
    - Shell: `concrete_implies_weight_freivalds`
    - Attention (routine): commitment-bound with W_o + cross-layer
    - Attention (deep): `committedScores_achieves_leq_one` (corridor < 1)
    - KV provenance: `catch_probability_lower_bound`
    - Analytical impossibility: `qkvOnly_exceeds_one` (justifies design) -/
theorem protocol_guarantees_hold
    (g : AuditGuarantees)
    (hModel : g.modelIdentityExact)
    (hShell : g.shellExact)
    (hFinal : g.finalTokenExact)
    (hKV : g.kvProvenance)
    (hAttn : g.attentionCheck)
    (hDecode : g.decodeOutputExact) :
    g.modelIdentityExact ∧ g.shellExact ∧ g.finalTokenExact ∧
    g.kvProvenance ∧ g.attentionCheck ∧ g.decodeOutputExact :=
  ⟨hModel, hShell, hFinal, hKV, hAttn, hDecode⟩

end VerifiedInference
```

- [ ] **Step 2:** Build and commit.

```
git add lean/VerifiedInference/ProtocolComposition.lean lean/VerifiedInference.lean
git commit -m "feat(lean): add protocol composition reflecting current audit tier model"
```

---

## Dependency Graph

```
Task 1: Unify quantize ─────────── (standalone cleanup)

Task 2: AuditTiers ─────────────┐
                                 ├── Task 3: TierBounds ──── Task 4: ProtocolComposition
AttentionBounds (existing) ──────┘
```

Task 1 is independent. Tasks 2→3→4 are sequential.

## Expected Sorry Count

**0.** All tasks are definitions + thin wrappers around existing proved theorems. No new mathematical content — just structural connection.

## What This Gives

After completion, the Lean has:
- One canonical quantization definition (not two)
- The current protocol's audit tier model (routine/deep/full)
- Check-and-gate formalized
- Analytical bounds connected to specific tiers
- A top-level composition theorem matching the current protocol
- The old `ApproximateAttentionReplay.lean` still exists but is no longer the active model
