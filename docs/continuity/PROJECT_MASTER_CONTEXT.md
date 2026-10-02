# RateMyCollezzz — Project Master Context

> Canonical long-term continuity document for future ChatGPT, Claude, Gemini, Codex, Antigravity, or other coding-agent sessions.

This file is intentionally detailed. A future session should read this before implementation work so that the project is not reconstructed from guesses or restarted from an outdated point.

---

# 1. PROJECT IDENTITY

**Project:** RateMyCollezzz / Rate My College
**Local repository:** `C:\Projects\RateMyCollezzz`
**Remote used during development:** `https://github.com/sighkalp/RateMyCollezzz.git`  
**Primary branch:** `main`

RateMyCollezzz is intended to become a complete college discovery, review, trust, and moderation platform—not merely a review classifier and not merely a Discover/map page.

The project combines:

- College discovery/search
- Map-driven browsing
- Detailed college profiles
- Official/verified logos and gallery images
- Structured college data
- A first-party review system
- Anonymous public reviewer presentation
- Helpful voting
- Reports
- Appeals
- Notifications
- Moderation/admin workflows
- A custom Review Risk Model (RRM)
- A separate Trust/anti-manipulation layer
- Security, audit, data, deployment, and operational infrastructure

The core architecture principle is:

> **RRM UNDERSTANDS. TRUST DECIDES.**

RRM must produce evidence/signals. Trust/policy owns moderation decisions.

---

# 2. FIVE-LAYER PRODUCT ARCHITECTURE

## Layer 1 — Experience / 2.5D UI

Intended technology contract:

- HTML
- CSS
- Bootstrap 5
- Vanilla JavaScript
- HTMX

Long-term product surface:

- Landing/Home
- Discover Colleges
- Search
- Map view
- College detail/profile
- College logo/gallery
- Courses
- Fees
- Placements
- Hostel
- Infrastructure
- Library
- Labs
- Campus life
- Sports
- Admissions
- Review listing
- Review detail
- Review submission
- Helpful voting
- Authentication
- Profile/student experience
- Notifications
- Reports
- Appeals
- Moderation
- Admin
- About
- Contact
- Privacy
- Terms
- Loading states
- Error states
- Empty states
- Mobile/responsive states
- Accessibility states

UI/map direction already discussed:

- dark map core
- brighter surrounding side panels
- center search
- right-side details panel
- right-side list
- location-pin style markers rather than tiny square icons
- college logo fully visible inside/around the pin
- subtle hover/click behavior
- geographically accurate local-area presentation rather than a fake “3D India”

## Layer 2 — Django Core Platform

Technology:

- Python
- Django
- SQLite development
- PostgreSQL production

Expected responsibilities:

- Roles:
  - STUDENT
  - VISITOR
  - MODERATOR
  - ADMIN
- Profile
- Colleges
- College location
- Categories
- Statistics
- Review lifecycle
- Anonymous review presentation
- Helpful votes
- Reports
- Moderation cases
- Evidence
- Appeals
- Analytics
- Notifications
- Permissions
- Authentication
- CSRF
- Migrations
- Layer adapters
- Tests

Review lifecycle direction:

```text
DRAFT -> APPROVED
```

Layer 2 consumes Layer 3/4 contracts; it must not absorb their intelligence/policy responsibilities.

## Layer 3 — Review Intelligence / RRM

Technology/research stack:

- Python
- PyTorch
- NumPy
- pandas
- scikit-learn
- SentencePiece
- Transformer encoder
- Character CNN
- Deterministic evidence components

Layer 3 produces review intelligence signals.

It does NOT:

- ban users
- delete reviews
- hide reviews
- determine final moderation outcomes

## Layer 4 — Trust / Anti-Manipulation

Layer 4 consumes:

- neural RRM signals
- deterministic PII evidence
- similarity/duplicate evidence
- platform context
- account/context signals
- policy rules

It owns final trust/moderation logic.

## Layer 5 — Data / Security / Audit / Operations

Expected responsibilities:

- production PostgreSQL
- secrets
- security headers
- CSRF/CORS policy
- rate limits
- audit logs
- monitoring
- backups
- incident/error handling
- deployment
- health checks
- data governance
- operational reliability

---

# 3. PRODUCT DATA AND MEDIA PRINCIPLES

## First-party reviews

Do not import third-party reviews into RateMyCollezzz as live RateMyCollezzz reviews.

The product should own its own review ecosystem.

## Reviewer anonymity

Public reviewers remain anonymous.

Do not expose personal data merely for richer UI.

## College logo/gallery policy

Strict rule:

> **Missing image is better than the wrong image.**

When collecting college media:

- prefer official college sources
- prefer verified sources
- deep research is encouraged
- do not artificially fill every image category
- do not use unrelated images just to make a profile “complete”
- the same rule applies to logos

Possible media categories include:

- campus
- hostel
- library
- labs
- classrooms
- sports
- infrastructure
- placements/career facilities
- campus life

Only populate categories when evidence exists.

---

# 4. RRM MISSION

The Review Risk Model analyzes review content and produces structured evidence.

Canonical six primary labels in exact order:

1. `spam`
2. `deception`
3. `toxicity`
4. `advertising`
5. `off_topic`
6. `pii`

Unknown label:

```text
UNKNOWN_LABEL = -1
```

UNKNOWN is not negative.

Unknown supervised positions are excluded from supervised loss/evaluation.

Separate/secondary signals:

- similarity/duplicate evidence
- quality
- optional weak AI-like signal

Deception requires defensible ground truth.

Do not infer deception from spam/toxicity/advertising alone.

---

# 5. RESEARCH FOUNDATION

Core papers used to shape the RRM:

1. Vaswani et al. — Attention Is All You Need
2. Devlin et al. — BERT
3. Liu et al. — RoBERTa
4. Kudo & Richardson — SentencePiece
5. Zhang/Zhao/LeCun — character-level CNN research
6. Ott et al. — deceptive opinion spam

A broader research catalog of roughly 13 papers was discussed, with 5–6 especially critical to RRM design.

The user also uses NotebookLM to study individual papers and prefers concise, crystal-clear prompts that explain even smaller relevant concepts without producing unnecessarily huge text.

---

# 6. TARGET RRM ARCHITECTURE

## 6.1 Semantic encoder — locked

Defaults:

```text
vocab_size = 22000
hidden_size = 384
num_layers = 8
num_heads = 6
head_dim = 64
ffn_dim = 1536
max_sequence_length = 256
pad_id = 0
dropout = 0.1
attention_dropout = 0.1
layer_norm_eps = 1e-12
initializer_range = 0.02
```

Architecture:

- token embedding: 22000 x 384
- learned position embedding: 256 x 384
- embedding LayerNorm + dropout
- 8 Pre-LN blocks:
  - `x = x + Attn(LN1(x))`
  - `x = x + FFN(LN2(x))`
- final LayerNorm
- semantic representation:
  - `last_hidden_state[:,0,:]`
- no learned pooler
- no token-type embedding
- no task head inside encoder

Exact parameter count:

```text
22,743,552
```

Output representation:

```text
[B,384]
```

## 6.2 Character branch — locked

Byte encoding:

```text
PAD = 0
byte b -> b + 1
byte_vocab_size = 257
max_byte_length = 512
```

Python encoding:

```text
UTF-8
errors="replace"
```

Important edge behavior:

- right truncation
- no UTF-8 repair after truncation
- lone surrogate replacement follows Python’s replacement behavior

Character model:

```text
embedding: 257 x 64
embedding_dropout: 0.1

parallel Conv1d:
  kernel 3 -> 128 channels
  kernel 4 -> 128 channels
  kernel 5 -> 128 channels

GELU
strict window-aware masked max
concat -> 384
dropout 0.1
Linear(384,64)
LayerNorm(64)
```

Output:

```text
[B,64]
```

Exact parameters:

```text
139,904
```

## 6.3 Fusion — locked

Simple concatenation:

```text
[B,384] + [B,64] -> [B,448]
```

No learned fusion parameters.

Representation stack total:

```text
22,883,456
```

## 6.4 Multi-task classifier — locked

Single classifier:

```text
Linear(448,6)
```

Head parameters:

```text
2,694
```

Permanent neural RRM total:

```text
22,886,150
```

Model output:

```text
logits [B,6]
```

No activation embedded into the output contract.

---

# 7. RRM PRETRAINING STRATEGY — LOCKED

RRM 3.6 selected:

> **Teacher-assisted self-supervised pretraining**

Student:

- custom semantic encoder
- custom SentencePiece tokenizer

Teacher candidate:

- frozen multilingual BERT (mBERT)

Teacher/student use the same raw review text but tokenize independently.

Pretraining objectives:

1. MLM
2. sequence-level representation distillation

Teacher representation target:

```text
mBERT last_hidden_state[:,0,:]
```

Temporary projection:

```text
Linear(384,768)
```

Projection is only for distillation and is discarded afterward.

Representation objective:

- MSE

Forbidden in this pretraining strategy:

- task-logit knowledge distillation
- supervised six-task labels
- deception labels
- teacher/student token alignment

Final student inference is independent of mBERT.

Important wording:

mBERT has multilingual/Hindi pretraining coverage.

Do not claim before evaluation that it is specifically superior/validated for Hinglish or Roman Hindi.

---

# 8. TOKENIZER CONTRACT — LOCKED

Approach:

- SentencePiece
- Unigram
- target vocab around 22k
- identity normalization
- preserve case
- coverage 1.0
- byte fallback enabled

Special IDs:

```text
pad = 0
unk = 1
cls = 2
sep = 3
mask = 4
bos = -1
eos = -1
```

Output:

- input_ids
- attention_mask

No token_type_ids.

Production tokenizer is not trained yet.

Synthetic pilot is not the production tokenizer corpus.

---

# 9. PILOT DATASET / ANNOTATION HISTORY

A 150-record synthetic pilot exists.

It is only for:

- schema validation
- annotation protocol testing
- smoke testing
- metric arithmetic
- infrastructure testing

It is NOT production scientific evidence.

It must not support claims of:

- accuracy
- F1 quality
- AUPRC quality
- generalization
- production readiness
- baseline superiority

Synthetic provenance was corrected to:

```text
SYNTHETIC_RESEARCH
```

Annotation history:

- A1: 40 records
- A2: 20 records
- overlap: 20

Agreement examples recorded:

```text
Spam:         55%   kappa ~= 0.100
Toxicity:     50%   kappa ~= 0.029
Advertising:  70%   kappa ~= 0.400
Off-topic:    45%   kappa ~= -0.058
PII:          45%   kappa ~= -0.100
Language:     30%   kappa ~= 0.122
Category:     25%   kappa ~= 0.178
```

The pilot was frozen.

Do not keep generating synthetic data just to inflate training size.

---

# 10. DETERMINISTIC EVIDENCE — LOCKED FOUNDATION

RRM 3.3 established deterministic infrastructure.

Components include:

- normalization
- exact fingerprinting
- character n-gram similarity
- deterministic PII
- text evidence
- unified deterministic prechecks

Similarity/duplicate evidence remains separate from six-task neural metrics.

---

# 11. BASELINES — LOCKED

## Word TF-IDF Logistic Regression

- word TF-IDF
- 1–2 grams
- Logistic Regression
- liblinear
- UNKNOWN masking
- no class-weight tuning in initial baseline

## Character n-gram Logistic Regression

- character 3–5 grams
- same general evaluation flow
- no major tuning

## mBERT baseline

Checkpoint:

```text
google-bert/bert-base-multilingual-cased
```

Exact params:

```text
177,858,054
```

Characteristics:

- shared transformer
- six logits
- UNKNOWN-aware BCE
- small batch
- gradient accumulation
- fp16 CUDA
- gradient checkpointing
- AdamW

## RoBERTa baseline

Checkpoint:

```text
FacebookAI/roberta-base
```

Not XLM-R.

Exact params:

```text
124,650,246
```

RoBERTa is English-pretrained.

Do not pre-judge its code-mixed performance before experiments.

---

# 12. RRM 3.10 SCIENTIFIC EVALUATION PROTOCOL — LOCKED

Production scientific evaluation:

```text
BLOCKED
```

Missing real artifacts include:

- approved production corpus
- trained production SentencePiece tokenizer
- completed RRM 3.6 pretraining
- frozen pretrained semantic checkpoint
- frozen real labeled dataset
- frozen train/validation/test manifest
- completed leakage audit
- supervised production checkpoints

## Primary metric

```text
macro-AUPRC
```

Checkpoint selection remains:

```text
validation masked BCE
```

## Task evaluability

A task is scientifically evaluable only if known ground truth contains:

```text
positive_support > 0
AND
negative_support > 0
```

If either class is absent:

```text
precision = None
recall = None
f1 = None
auprc = None
```

Still report:

- known_support
- positive_support
- negative_support

No arbitrary “5 positives + 5 negatives” rule.

## Macro metrics

Report:

```text
macro_f1
macro_f1_task_count
macro_f1_task_names

macro_auprc
macro_auprc_task_count
macro_auprc_task_names
```

All compared models on a frozen test set must use the same ground-truth-evaluable task set.

## Thresholding

Fixed default:

```text
0.5
```

Comparison:

```text
score >= threshold
```

Validation threshold tuning:

Candidates:

```text
unique known-label validation sigmoid scores + 0.5
```

Objective:

```text
validation F1
```

Tie:

1. closest to 0.5
2. smaller numeric threshold

Fallback:

```text
threshold = 0.5
threshold_source = "default_0.5_no_evaluable_validation"
```

TEST cannot tune thresholds.

## Calibration

Deferred / optional.

Do not call sigmoid scores calibrated probabilities.

## Neural seeds

Initial required neural comparison:

```text
3 seeds
```

Report:

- all runs
- mean
- standard deviation

Expand to 5 only if variance or stronger evidence requires it.

## Bootstrap defaults

```text
seed = 42
replicates = 1000
ci_level = 0.95
method = percentile
```

Paired comparison uses identical bootstrap indices.

## Required architecture ablation

Required later:

```text
semantic-only vs full semantic+character
```

Semantic-only is separately trained:

```text
RmcEncoder -> [B,384] -> Linear(384,6)
```

Do not drop character features only at inference.

Optional later ablations:

- character-only
- random vs pretrained semantic init
- frozen vs fine-tuned semantic encoder
- loss weighting
- task normalization
- fixed vs tuned thresholds

---

# 13. RRM ROADMAP / STATUS

## RRM 3.1 — Research Contract

Status: LOCKED

Known commit:

```text
65fbb4b
```

## RRM 3.2 — Dataset Foundation

Status: LOCKED

Known dataset-related commits include:

```text
3050974
6831d42
18627ec
81f83e8
234e478
90cb87d
1d7342a
8055c2a
9d2ddf7
```

## RRM 3.3 — Deterministic Pre-checks

Status: LOCKED

Known commits:

```text
bc54515
7784ac3
ff748fe
b32fd2d
a8ffc00
802293c
```

## RRM 3.4 — Baselines

Status: LOCKED

Known commits:

```text
940c341
5b86baa
6ef4286
```

RoBERTa baseline was also completed/pushed; exact hash not retained here. Never invent it.

## RRM 3.5 — RMC Tokenizer Infrastructure

Status: LOCKED

Production tokenizer still not trained.

## RRM 3.6 — Encoder Training Strategy

Status: LOCKED

Strategy:

```text
MLM + sequence-level representation distillation from frozen mBERT
```

Execution blocked by production corpus/tokenizer.

## RRM 3.7 — Semantic Encoder

Status: LOCKED

Committed/pushed.

Exact hash not retained here—do not invent.

## RRM 3.8 — Character Branch + Feature Fusion

Status: LOCKED

Committed/pushed.

Exact hash not retained here—do not invent.

## RRM 3.9 — Multi-task Heads + Training Infrastructure

Status:

```text
LOCKED
COMMITTED
PUSHED
```

Exact commit:

```text
81cf7ce
```

Message:

```text
feat(rrm): add multitask heads and training infrastructure
```

Locked regression baseline before 3.10:

```text
853 passed
2 known pre-existing RoBERTa scheduler warnings
```

## RRM 3.10 — Scientific Evaluation

Protocol:

```text
LOCKED
```

Infrastructure implementation:

```text
LOCKED + COMPLETE
commit: efb9f4f
feat(rrm): add scientific evaluation infrastructure
```

Final validated test result:

```text
1079 passed
2 known pre-existing RoBERTa scheduler warnings
```

See `CURRENT_STATE.md`.

## RRM 3.11 — Packaging + Layer Contract

Status:

```text
LOCKED + COMPLETE
commit: a24b88b
feat(rrm): add runtime packaging and layer contract
```

RRM 3.11 establishes the stable Layer-3 public runtime boundary.

---

# 13a. GATE A — DATASET SOURCE & PROVENANCE DECISIONS

Gate A: CLOSED

Close commit:

```text
b0cd51b
docs(rrm): close production source decision gate
```

Final source decisions:

HOLD (5):
- Source 1 — Deceptive Opinion Spam Corpus
- Source 2 — Jigsaw Toxic Comment Classification Challenge
- Source 3 — Jigsaw Unintended Bias in Toxicity Classification
- Source 4 — YelpCHI
- Source 8 — Real Platform / Public College Reviews

APPROVE_WITH_CONDITIONS (3):
- Source 5 — Human-Written RMC Research Data (primary planned in-domain corpus)
- Source 6 — Controlled RMC Research Data (controlled ground-truth supplement)
- Source 7 — Synthetic / Derived RMC Research Data (conditional augmentation/robustness)

REJECT: NONE

Durable facts:

- HOLD is not permanent rejection.
- No production dataset was downloaded or ingested by Gate A.
- deception is UNKNOWN where defensible truth provenance is absent.
- Human-Written RMC is the primary planned in-domain corpus.
- Controlled RMC supplies controlled ground-truth cases.
- Synthetic / Derived RMC is limited to conditional augmentation/robustness use and must preserve provenance, inherited-rights constraints, PII controls, and parent-child grouping.
- RRM 3.1–3.11 implementation infrastructure is locked.

Gate A artifact:

```text
rrm/PRODUCTION_SOURCE_DECISION_PACKET.md
```

---

# 13b. GATE B — PRODUCTION ANNOTATION + METADATA CONTRACT

Gate B: CLOSED

Close commit:

```text
7b99bf4
docs(rrm): freeze production annotation contract
```

Gate B contract:

```text
rrm/PRODUCTION_ANNOTATION_CONTRACT.md
```

Version:
1.0

Status:
FROZEN

Durable Gate B policies:

- canonical six risk targets frozen
- only deception supports UNKNOWN = -1
- review_text is sole neural input
- research/governance metadata is non-neural
- language_mix is single-valued with exactly:
  ENGLISH
  HINGLISH
  ROMAN_HINDI
  OTHER
  MIXED_OTHER
- college_category is single-valued, nullable, with exactly ten
  non-null categories
- 100% independent double annotation
- trained senior third-party adjudication
- Human-Written RMC consent/provenance policy frozen
- Controlled RMC truth policy frozen
- Synthetic/Derived RMC provenance policy frozen
- PII safe-surrogate / target-input consistency policy frozen
- Gate D/E/F/G ownership frozen

---

# 14. ACTIVE GATE

Gate C — Production Corpus Collection + Annotation

Gate C infrastructure is implemented and validated.

Actual production corpus collection and annotation have NOT been completed.

Production dataset is NOT frozen.

Gate D is INACTIVE.

Gate C implementation commits:

```text
0923401
feat(rrm): add production corpus schema and provenance

4365b34
feat(rrm): add production annotation workflow

1a1dbe8
feat(rrm): add corpus qc and gate d export
```

Full regression:

```text
1417 passed
2 warnings
0 failed
80.73s
```

Warnings from pre-existing RoBERTa scheduler-order warnings.

Runtime:

```text
RRM runtime foundation: PASS
```

Next action:

Begin actual Gate C collection and annotation using ONLY approved
Gate C source classes and the frozen Gate B v1.0 annotation/metadata
contract.

Do NOT proceed to Gate D until production corpus is collected,
annotated, and frozen.

---

# 15. KNOWN GIT HISTORY

Known exact hashes:

```text
e44fe95  chore: establish locked project architecture
65fbb4b  docs(rrm): lock research contract
3050974  data(rrm): add spam and advertising pilot cases
6831d42  data(rrm): add toxicity boundary pilot cases
18627ec  data(rrm): add off-topic and PII boundary cases
81f83e8  data(rrm): add controlled deception pilot cases
234e478  data(rrm): add multi-label and ambiguous pilot cases
90cb87d  fix(rrm): correct pilot provenance and annotation state
1d7342a  data(rrm): add independent A2 pilot annotations
8055c2a  feat(rrm): add pilot agreement analysis
9d2ddf7  docs(rrm): freeze dataset pilot and activate deterministic prechecks
bc54515  docs(rrm): activate deterministic prechecks
7784ac3  docs(rrm): fix active component encoding
ff748fe  feat(rrm): add deterministic similarity prechecks
b32fd2d  feat(rrm): add deterministic PII and text evidence
a8ffc00  feat(rrm): add unified deterministic prechecks
802293c  docs(rrm): close deterministic prechecks and activate baselines
940c341  feat(rrm): add tfidf logistic regression baseline
5b86baa  char ngram logistic regression baseline
6ef4286  feat(rrm): add multilingual BERT baseline
81cf7ce  feat(rrm): add multitask heads and training infrastructure
efb9f4f  feat(rrm): add scientific evaluation infrastructure
a24b88b  feat(rrm): add runtime packaging and layer contract
0923401  feat(rrm): add production corpus schema and provenance
4365b34  feat(rrm): add production annotation workflow
1a1dbe8  feat(rrm): add corpus qc and gate d export
```

Do not fabricate unknown hashes.

---

# 15. LOCAL DEVELOPMENT ENVIRONMENT

Known local environment:

```text
OS: Windows
Repo: C:\Projects\RateMyCollezzz

Python venv:
.venv

Python:
Astral CPython 3.11.15

PyTorch:
2.14.0+cu130

GPU:
RTX 2050 4GB

CPU:
i5-11260H
6 cores / 12 threads

RAM:
~7.73GB
```

Important:

Plain `python` may point to another interpreter.

Prefer:

```text
.venv/Scripts/python.exe
```

Windows LF→CRLF notices are usually informational, not code failures.

---

# 16. UI / DISCOVER HISTORY

Earlier UI work focused heavily on Discover/map.

Important constraints remembered:

- There were “final” and “normal” working variants/folders in earlier iterations.
- Do not casually rename the user’s chosen working folder.
- Dark map core.
- Brighter surrounding sides.
- Search centered.
- Right details panel.
- Right list.
- College map logos should use location-pin treatment.
- Logo should remain fully visible.
- Subtle hover/click behavior.
- No imported external reviews.
- Public reviewers anonymous.
- College profile should expose full structured information where verified.
- Use official/verified college assets.
- Missing image > wrong image.

Long-term UI must expand far beyond Discover.

---

# 17. AFTER RRM 3.11: MODEL WORK STILL REMAINS

Very important distinction:

```text
RRM infrastructure complete
!=
production model trained
!=
RateMyCollezzz complete
```

After RRM software infrastructure:

1. Build real RMC dataset.
2. Establish legal/provenance/data-source controls.
3. Finalize annotation guide.
4. Perform real annotation.
5. Freeze dataset version.
6. Freeze train/validation/test manifest.
7. Run leakage audit.
8. Train production SentencePiece tokenizer.
9. Execute RRM 3.6:
   - MLM
   - representation distillation
10. Freeze semantic checkpoint.
11. Train supervised six-task RRM.
12. Run 3-seed neural experiments.
13. Run semantic-only ablation.
14. Train/evaluate locked baselines fairly.
15. Run final scientific evaluation.
16. Freeze production model only after valid evidence.

---

# 18. AFTER MODEL: PRODUCT WORK STILL REMAINS

## Trust

- evidence-to-policy integration
- moderation policies
- review/report handling
- appeals
- auditability

## Django platform

- complete college model
- review lifecycle
- helpful voting
- reports
- moderation
- appeals
- notifications
- permissions
- analytics
- RRM/Trust adapters
- security

## UI

- landing/home
- discover/map
- search
- college profiles
- galleries/logos
- courses
- fees
- placements
- hostel
- infrastructure
- library
- admissions
- reviews
- review creation
- profile/auth
- notifications
- moderation/admin
- report/appeal
- contact/about
- privacy/terms
- responsive/mobile
- accessibility
- empty/loading/error states

## College data

- verified logos
- verified photos
- verified structured details
- source quality tracking
- no fake completeness

## Operations

- production PostgreSQL
- deployment
- monitoring
- secrets
- backups
- audit
- rate limiting
- health checks

## QA

- unit
- integration
- end-to-end
- security
- performance
- accessibility
- responsive QA
- Trust/RRM contract QA

---

# 19. FINAL DEFINITION OF “PROJECT COMPLETE”

The project is NOT complete merely because:

- Discover works
- RRM 3.11 is done
- six logits exist
- synthetic tests pass

A stronger completion definition:

## Product

- major user journeys complete
- college discovery useful
- college profile credible
- reviews submit/read/vote/report
- moderation/admin/appeal flows work

## Data

- verified college data/media
- correct source governance
- real review dataset governance

## Model

- production tokenizer
- semantic pretraining
- supervised training
- 3-seed experiments
- baselines
- ablation
- scientific evaluation
- model freeze

## Trust

- policy consumes evidence
- model does not own final decisions
- decisions auditable

## Engineering

- production-capable backend
- security
- observability
- deployment reproducibility
- tests

## UX

- complete multi-page experience
- mobile
- accessibility
- loading/error/empty states
- visual/data accuracy

---

# 20. RULES FOR FUTURE AGENTS

Every future agent must:

1. Read this file.
2. Read `CURRENT_STATE.md`.
3. Read `DECISION_LOG.md`.
4. Read `CONTINUATION_PROTOCOL.md`.
5. Read repository architecture/plan/file map.
6. Inspect current Git state.
7. Inspect active phase code/tests.
8. Never restart a locked phase due to lost chat context.
9. Never invent test results.
10. Never invent commit hashes.
11. Never treat synthetic pilot as production evidence.
12. Never let RRM make Trust decisions.
13. Never commit/push before user/ChatGPT approves the final validation report.
