# RateMyCollezzz — Decision Log

This file contains locked decisions that future assistants/agents must not silently reopen.

---

## D-001 — Five layers

```text
Layer 1: Experience / UI
Layer 2: Django Core Platform
Layer 3: Review Intelligence / RRM
Layer 4: Trust / Anti-Manipulation
Layer 5: Data / Security / Audit / Operations
```

Status: LOCKED direction.

---

## D-002 — Core principle

> **RRM UNDERSTANDS. TRUST DECIDES.**

RRM returns evidence/signals.

Trust/policy makes final moderation decisions.

Status: LOCKED.

---

## D-003 — Canonical labels

Exact order:

```text
spam
deception
toxicity
advertising
off_topic
pii
```

Status: LOCKED.

---

## D-004 — UNKNOWN

```text
UNKNOWN_LABEL = -1
```

UNKNOWN is not negative.

Status: LOCKED.

---

## D-005 — Similarity is separate

Duplicate/similarity evidence is separate from six neural heads.

Do not include similarity in six-task macro score.

Status: LOCKED.

---

## D-006 — Deception

Requires defensible ground truth.

Do not infer deception from:

- spam
- toxicity
- advertising
- writing style
- AI-like text

Status: LOCKED.

---

## D-007 — Semantic encoder

```text
vocab=22000
hidden=384
layers=8
heads=6
ffn=1536
max_seq=256
Pre-LN
repr = last_hidden_state[:,0,:]
```

Exact params:

```text
22,743,552
```

Status: LOCKED.

---

## D-008 — Character branch

```text
PAD=0
byte b -> b+1
vocab=257
max_byte_length=512
embedding_dim=64
kernels=(3,4,5)
channels=128 each
char_dim=64
```

Exact params:

```text
139,904
```

Status: LOCKED.

---

## D-009 — Fusion

```text
384 + 64 = 448
```

Simple concatenation.

No learned fusion parameters.

Status: LOCKED.

---

## D-010 — Head

```text
Linear(448,6)
```

Params:

```text
2,694
```

Total permanent neural RRM:

```text
22,886,150
```

Status: LOCKED.

---

## D-011 — Pretraining

RRM 3.6:

```text
MLM
+
sequence-level representation distillation
```

Teacher:

```text
frozen mBERT
```

No:

- task-logit KD
- supervised six-task labels
- token alignment

Status: LOCKED.

---

## D-012 — mBERT wording

Allowed:

> mBERT has multilingual/Hindi pretraining coverage.

Not allowed before real evidence:

- “mBERT is proven best for Hinglish”
- “mBERT is validated for Roman Hindi”
- “mBERT will definitely outperform”

Status: LOCKED wording.

---

## D-013 — Tokenizer

SentencePiece Unigram.

Special IDs:

```text
pad=0
unk=1
cls=2
sep=3
mask=4
bos=-1
eos=-1
```

No token_type_ids.

Status: LOCKED infrastructure.

---

## D-014 — Synthetic pilot

150-record synthetic pilot is NOT performance evidence.

Allowed:

- schema
- smoke
- metrics plumbing
- infrastructure tests

Forbidden claims:

- accuracy
- F1 quality
- AUPRC quality
- superiority
- production readiness

Status: LOCKED.

---

## D-015 — No synthetic inflation

Do not generate more synthetic reviews just to create a fake final corpus.

Status: LOCKED.

---

## D-016 — Baselines

Locked baseline set:

- word TF-IDF LR
- char n-gram LR
- mBERT
- RoBERTa

Status: LOCKED.

---

## D-017 — Primary metric

```text
macro-AUPRC
```

Checkpoint selection:

```text
validation masked BCE
```

Status: LOCKED.

---

## D-018 — One-class rule

If:

```text
positive_support == 0
OR
negative_support == 0
```

then:

```text
precision=None
recall=None
f1=None
auprc=None
```

Status counts still reported.

Status: LOCKED.

---

## D-019 — No arbitrary support threshold

Do not impose a fake rule like:

```text
5 positives + 5 negatives
```

before real data justifies a stronger cutoff.

Status: LOCKED.

---

## D-020 — Threshold behavior

Default:

```text
0.5
```

Comparison:

```text
score >= threshold
```

Status: LOCKED.

---

## D-021 — Threshold tuning

Candidates:

```text
unique validation sigmoid scores + 0.5
```

Objective:

```text
validation F1
```

Tie:

1. closest to 0.5
2. smaller number

Fallback:

```text
0.5
default_0.5_no_evaluable_validation
```

Status: LOCKED.

---

## D-022 — Neural seeds

Initial required:

```text
3 independent seeds
```

Expand to 5 only if needed.

No arbitrary preferred 4.

Status: LOCKED.

---

## D-023 — Bootstrap

Defaults:

```text
seed=42
replicates=1000
ci_level=0.95
method=percentile
```

Paired comparison uses same sampled indices.

Status: LOCKED.

---

## D-024 — Required ablation

```text
semantic-only
vs
full semantic+character
```

Semantic-only must be separately trained:

```text
RmcEncoder -> [B,384] -> Linear(384,6)
```

Status: LOCKED.

---

## D-025 — Calibration

Deferred/optional.

Sigmoid score is not automatically a calibrated probability.

Status: LOCKED.

---

## D-026 — Test-set reuse

If test observations drive model changes, that changed model requires a new development/evaluation round.

Status: LOCKED.

---

## D-027 — Media accuracy

> **Missing image is better than a wrong image.**

Use official/verified images/logos.

Status: LOCKED.

---

## D-028 — First-party reviews

Do not import external reviews as RateMyCollezzz reviews.

Status: LOCKED.

---

## D-029 — Public reviewer anonymity

Public reviewers remain anonymous.

Status: LOCKED.

---

## D-030 — Discover/map direction

- dark map
- brighter sides
- center search
- right detail
- right list
- location pin
- visible logo
- subtle interaction
- accurate local geography

Status: LOCKED unless user explicitly changes it.

---

## D-031 — Agent workflow

Every phase:

1. audit/design
2. freeze
3. implement
4. focused tests
5. full regression
6. runtime/hygiene
7. ChatGPT review
8. micro-fix if needed
9. commit
10. push
11. lock

Status: LOCKED workflow.

---

## D-032 — Git safety

Agents should not without approval:

- reset --hard
- checkout destructive forms
- restore entire tree
- clean
- force-push
- delete unrelated files
- commit/push

Status: LOCKED.

---

## D-033 — Exact hashes only

Never invent commit hashes.

Status: LOCKED.

---

## D-034 — Project interpreter

Prefer:

```text
.venv/Scripts/python.exe
```

Status: LOCKED practice.

---

## D-035 — RRM 3.10 macro isolation

Do not modify locked RRM 3.9 `evaluation.py` just to add scientific task-name fields.

RRM 3.10 owns its scientific result structure.

Status: LOCKED.

---

## D-036 — JSON source of truth

Scientific results:

- JSON = source of truth
- Markdown = derived presentation

Synthetic smoke marker:

```text
NON_SCIENTIFIC_SYNTHETIC_SMOKE
```

Human output must clearly say:

```text
NON-SCIENTIFIC
SYNTHETIC PILOT
NOT PERFORMANCE EVIDENCE
```

Status: LOCKED.

---

## D-037 — RRM 3.10 lock and finalization

RRM 3.10 scientific evaluation infrastructure is LOCKED and COMPLETE.

Final permanent commit:

```text
efb9f4f
feat(rrm): add scientific evaluation infrastructure
```

Final validated test result:

```text
1079 passed
2 known pre-existing RoBERTa scheduler warnings
```

Schema validation occurs at the machine-readable ingestion boundary:
`deserialize_scientific_result()` and `result_from_json()` reject
malformed metadata types via `SchemaValidationError`.

Bare dataclass construction does not enforce types — that is by design.

`threshold_sources` belongs in `ProvenanceMetadata`, not `ScientificResult`.

Scientific status constants have one canonical owner: `scientific_evaluation.py`.

Scientific macro result uses `ScientificMacroResult`, not old `MacroMetrics`.

Production execution remains blocked:
production corpus, tokenizer, pretraining, checkpoint, dataset,
supervised RRM, final test evaluation, and performance claims
are all unavailable or unexecuted.

Status: LOCKED.

---

## D-038 — RRM 3.11 next phase

RRM 3.11 — Packaging and Layer Contract is the NEXT phase.

It must package stable RRM outputs and contracts.

It must not silently start production model training.

Status: PLANNING ONLY — NOT STARTED.

---

## D-039 — RRM 3.11 packaging contract locked

RRM 3.11 Packaging + Layer Contract is COMPLETE, VALIDATED, COMMITTED, and PUSHED.

Implementation commit:

```text
a24b88b
feat(rrm): add runtime packaging and layer contract
```

Stable Layer-3 public boundary established:

```text
RRM_RUNTIME_SCHEMA_VERSION
NeuralAvailability
RrmConfigurationError
RrmInferenceEngine
RrmInferenceError
RrmInferenceRequest
RrmInferenceResult
```

Request contract:

```text
required:    review_text (str)
required:    review_id (str)
optional:    similarity_candidates (tuple of (review_id, text) pairs)
```

Result contract:

```text
deterministic_prechecks   always present
task_scores               canonical PRIMARY_LABELS mapping or None
neural_availability       "available" or "unavailable"
neural_unavailable_reason required when unavailable
model_identity            required when available, None when unavailable
tokenizer_identity        required when available, None when unavailable
runtime_schema_version    "1.0"
```

Task scores:

```text
UNCALIBRATED SIGMOID scores in [0, 1]
canonical PRIMARY_LABELS keys in exact order:
  spam, deception, toxicity, advertising, off_topic, pii
```

When neural runtime is unavailable:

```text
task_scores = None
```

No placeholders. No random outputs.

Error contract:

```text
bad caller/request input:       TypeError / ValueError
invalid engine configuration:   RrmConfigurationError
unexpected neural runtime fail: RrmInferenceError
expected neural absence:        structured UNAVAILABLE result
```

Immutability:

```text
RrmInferenceRequest  frozen; candidates deep-copied into immutable tuple
RrmInferenceResult   frozen; task_scores defensively copied, read-only
to_dict()            minimal JSON-compatible serialization API
```

Neural runtime is dependency-injected. No production checkpoint-loading
format is frozen.

RRM/Trust separation preserved:

```text
RRM produces evidence/signals.
Trust owns policy decisions.
```

RRM 3.11 does NOT:

```text
- discover checkpoints
- load production models
- make Trust decisions
- implement moderation policy
- produce scientific performance claims
- depend on Trust / Platform / Experience / Ops
```

Validation:

```text
focused inference tests: 134 passed
full RRM regression:     1213 passed
                          2 known pre-existing RoBERTa scheduler warnings
runtime_check:           PASS
dependencies added:      NONE
```

RRM 3.1–3.11 is now the complete locked RRM infrastructure sequence.

Production assets remain unavailable:

```text
production corpus:                   NOT AVAILABLE
production tokenizer:                NOT TRAINED
RRM 3.6 production pretraining:      NOT EXECUTED
production semantic checkpoint:      NOT AVAILABLE
real labeled production dataset:     NOT AVAILABLE
production supervised RRM:           NOT TRAINED
real final test evaluation:          NOT EXECUTED
scientific production performance claims: NONE
```

Status: LOCKED.

---

## D-040 — Gate A dataset source decisions closed

Gate A — Dataset Source & Provenance Plan: CLOSED.

Close commit:

```text
b0cd51b
```

Final source decisions:

HOLD (5):
- Deceptive Opinion Spam Corpus
- Jigsaw Toxic Comment Classification Challenge
- Jigsaw Unintended Bias in Toxicity Classification
- YelpCHI
- Real Platform / Public College Reviews

APPROVE_WITH_CONDITIONS (3):
- Human-Written RMC Research Data
- Controlled RMC Research Data
- Synthetic / Derived RMC Research Data

REJECT: NONE

Durable interpretation:

- HOLD sources remain research-relevant but are not authorized for production ingestion/training.
- HOLD is not permanent rejection.
- Human-Written RMC is the primary planned in-domain corpus.
- Controlled RMC supplies controlled ground-truth cases.
- Synthetic / Derived RMC is limited to conditional augmentation/robustness use and must preserve provenance, inherited-rights constraints, PII controls, and parent-child grouping.
- deception is UNKNOWN where truth provenance is unavailable.
- Gate A downloaded or ingested no production datasets.

Next gate:

Gate B — Production Annotation + Metadata Contract

Status: LOCKED.

---

## D-041 — Gate B Production Annotation + Metadata Contract Frozen

Date:

2026-10-01

Record:

Gate B close commit:
7b99bf4

Contract:
rrm/PRODUCTION_ANNOTATION_CONTRACT.md

Version:
1.0

Decision:

Gate B — Production Annotation + Metadata Contract frozen.

Frozen policies:

- exact six-task domains
- only deception supports -1
- review_text sole neural input
- research metadata non-neural
- PII safe-surrogate / target-input consistency
- language taxonomy/cardinality
- college-category taxonomy/cardinality/null semantics
- 100% independent double annotation
- trained senior third-party adjudication
- Human-Written RMC consent/provenance
- Controlled RMC truth provenance
- Synthetic/Derived RMC parent/provenance requirements
- downstream Gate D/E/F/G ownership

Next:

Gate C — Production Corpus Collection + Annotation

All canonical metadata schema tables must use exactly seven columns:

```text
Field | Requirement | Type / Allowed Values | Neural Input | Training Target | Owning Gate | Purpose
```

No alternate columns (e.g., "Conditional On") in canonical tables.
Conditionality is expressed within Type/Allowed Values or Purpose text.

Affected tables:

- Annotation metadata (Section 11): annotation_status, annotation_guide_version, annotator_A_id, annotator_B_id, adjudicator_id, annotation_notes
- Provenance/source (Section 12): source_type, consent_status, contributor_pseudonym, collection_method, experiment_id, controlled_targets, deception_truth, control_protocol_id, parent_review_id, derivation_type, generation_method
- Audit (Section 13): created_at, finalized_at
- Later-gate (Section 14): dataset_version, split_membership, split_group_id

Owning gates:
- Annotation/provenance: B contract / C collection
- Audit: C
- Later-gate: D for dataset_version; E for split_membership and split_group_id

Status: LOCKED.

---

## D-042 — Gate C infrastructure validated, phase remains active

Decision:

Gate C — Production Corpus Collection + Annotation infrastructure is
implementation-ready and regression-validated, but Gate C remains ACTIVE
until approved-source corpus collection and annotation are actually
completed.

Implementation commits:

```text
0923401
feat(rrm): add production corpus schema and provenance

4365b34
feat(rrm): add production annotation workflow

1a1dbe8
feat(rrm): add corpus qc and gate d export
```

Full repository regression:

```text
1417 passed
2 warnings
0 failed
80.73s
```

Warnings were pre-existing RoBERTa scheduler-order warnings, not Gate C
failures.

Runtime:

```text
RRM runtime foundation: PASS
```

Current state:

- production corpus has NOT been collected
- production corpus has NOT been annotated
- production dataset is NOT frozen
- Gate D remains INACTIVE

Next action:

Begin actual Gate C collection and annotation using ONLY approved
Gate C source classes and the frozen Gate B v1.0 contract.

Do not claim:

- production corpus AVAILABLE
- production corpus COLLECTED
- production annotation COMPLETE
- dataset FROZEN
- Gate C CLOSED
- Gate D ACTIVE

Do not invent any scientific production performance measurements.

Status: LOCKED.
