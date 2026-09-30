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
