# RateMyCollezzz — File Map

This document is the navigation index for the repository.

Its purpose is simple:

**If someone asks where something lives, this file should help them find it.**

Update this file whenever meaningful project files are:

- created
- moved
- renamed
- removed

Do not fill this document with speculative future files.

Only document files that actually exist or have been explicitly approved.

---

## Root Control Files

### `ARCHITECTURE.md`

**Layer:** Project-wide control

**Purpose:** Defines the stable five-layer architecture.

Defines:

- layer ownership
- layer boundaries
- RRM / Trust separation
- 2D + 3D map requirement
- UI migration policy
- repository structure
- legacy migration policy
- architecture change policy

**Must NOT:** Be modified by normal implementation tasks.

---

### `IMPLEMENTATION_PLAN.md`

**Layer:** Project-wide control

**Purpose:** Defines what is currently being built.

Contains:

- active layer
- active component
- RRM build sequence
- allowed work
- forbidden work
- experiment requirements
- completion rules
- learning/teaching gate

This is where implementation priority changes.

---

### `CLAUDE.md`

**Layer:** Project-wide control

**Purpose:** Defines mandatory implementation-agent behaviour.

Prevents:

- architecture expansion
- scope expansion
- unapproved files
- unapproved folders
- unnecessary dependencies
- unrelated refactoring
- fabricated research results
- uncontrolled feature removal
- uncontrolled Git operations

---

### `FILE_MAP.md`

**Layer:** Project-wide control

**Purpose:** Repository navigation and file ownership map.

This file should remain concise and accurate.

---

### `README.md`

**Layer:** Project-wide overview

**Purpose:** Human-facing overview of RateMyCollezzz.

Explains:

- what the project is
- five architectural layers
- current development focus
- current RRM direction
- where to start reading

---

### `.gitignore`

**Layer:** Project-wide repository hygiene

**Purpose:** Prevent local/generated content from entering Git.

Examples:

- secrets
- virtual environments
- Python caches
- Node modules
- build output
- local databases
- model checkpoints
- temporary artifacts
- logs

---

## Layer 1 — `experience/`

**Ownership:** User-facing experience.

**Current state:** Placeholder only unless `IMPLEMENTATION_PLAN.md` activates Layer 1.

**Currently existing tracked placeholder:** `experience/.gitkeep`

Known responsibilities:

- Discover UI
- search
- filters
- map
- 2D navigation
- 3D perspective navigation
- college pins
- college details
- reviews UI
- gallery UI
- compare
- saved colleges
- profile UI
- notifications UI
- community UI

Legacy frontend exists externally and will later be migrated deliberately.

Do not pre-create speculative Layer-1 files while RRM work is active.

---

## Layer 2 — `platform/`

**Ownership:** Django Platform Core.

**Current state:** Placeholder only unless activated.

**Currently existing tracked placeholder:** `platform/.gitkeep`

Known responsibilities:

- accounts
- profiles
- colleges
- locations
- reviews
- ratings
- helpful votes
- saved colleges
- communities
- confessions
- discussions
- chat
- notifications
- reports
- moderation records
- platform APIs
- permissions
- business logic

Do not pre-create speculative Layer-2 files while RRM work is active.

---

## Layer 3 — `rrm/`

**Ownership:** Review Intelligence / Review Risk Model.

**Current state:** ACTIVE

**Current active component:** RRM 3.4 - Baselines

### `rrm/pii_detection.py`

**Layer:** Layer 3 - Review Intelligence / RRM

**Purpose:** Deterministic extraction of observable email, phone, and URL pattern evidence while preserving original text spans and offsets.

**Used by:** RRM deterministic prechecks and descriptive text-evidence extraction.

**Must NOT:** Treat pattern matches as confirmed privacy violations, redact content automatically, or make Trust-layer moderation decisions.

### `rrm/deterministic_prechecks.py`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Unified deterministic precheck pipeline aggregating evidence from fingerprint, text evidence, PII detection, and similarity modules into a single immutable result.

**Used by:** RRM deterministic pre-check pipeline; future orchestration layer.

**Must NOT:** Make moderation decisions, assign risk scores, or classify spam/toxicity/advertising.

### `rrm/tests/test_deterministic_prechecks.py`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Pytest test suite for deterministic precheck orchestration and result contract.

**Used by:** CI, local validation, development correctness checks.

**Must NOT:** Contain moderation threshold logic or Trust-layer tests.

### `rrm/pii_detection.py`

**Layer:** Layer 3 — Review Intelligence / RRM

**Used by:** RRM deterministic pre-checks; future feature extraction pipeline.

**Must NOT:** Classify spam/toxicity/advertising, assign risk scores, or make moderation decisions.

### `rrm/tests/test_text_evidence.py`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Pytest test suite for PII detection and text-evidence generation correctness.

**Used by:** CI, local validation, development correctness checks.

**Must NOT:** Contain moderation threshold logic or Trust-layer tests.

### `rrm/text_normalization.py`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Conservative text normalization for deterministic similarity and duplicate analysis.

**Used by:** rrm/similarity.py; future RRM pre-check pipeline.

**Must NOT:** Transliterate Hindi, stem, remove stopwords, or perform semantic rewriting.

### `rrm/similarity.py`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Character n-gram Jaccard similarity for near-duplicate detection. Produces evidence signals only.

**Used by:** RRM deterministic pre-checks; future candidate-matching pipelines.

**Must NOT:** Make moderation decisions, delete reviews, ban users, or impose similarity thresholds.

### `rrm/tests/test_similarity.py`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Pytest test suite for text normalization and similarity evidence generation.

**Used by:** CI, local validation, development correctness checks.

**Must NOT:** Contain moderation threshold logic or Trust-layer tests.

### `rrm/RESEARCH_CONTRACT.md`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Locks the RRM research problem, label taxonomy, evidence boundaries,
research questions, evaluation plan, and scientific claims policy.

**Used by:** All later RRM dataset, tokenizer, model, training, and evaluation work.

**Must NOT:** Contain final experimental results or Layer-4 moderation logic.

### `rrm/DATASET_SPEC.md`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Defines the canonical RMC dataset schema, field semantics, target encoding, metadata boundaries, provenance, privacy, and leakage requirements.

**Used by:** Annotation, dataset creation, preprocessing, training, evaluation, and reproducibility workflows.

**Must NOT:** Contain final dataset results, final source claims, or model architecture implementation.

### `rrm/ANNOTATION_GUIDE.md`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Defines operational annotation rules for all RMC labels, including positive, negative, ambiguous, multi-label, disagreement, and exclusion cases.

**Used by:** Human annotation, adjudication, pilot dataset creation, label-quality analysis, and dataset validation.

**Must NOT:** Invent unavailable ground truth, force uncertain deception into binary labels, or define Layer-4 moderation decisions.

### `rrm/DATA_SOURCE_STRATEGY.md`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Defines approved dataset-source classes, source-to-label mappings, provenance requirements, domain limitations, licensing checks, privacy controls, and source-leakage protections.

**Used by:** Dataset acquisition, pilot design, annotation, preprocessing, experimental splits, and research reporting.

**Must NOT:** Treat candidate datasets as automatically approved, invent unsupported labels, or assume redistribution rights without verification.

### `rrm/PILOT_DATASET_DESIGN.md`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Defines the first 150-record RMC pilot, including language coverage, scenario composition, source rules, multi-label coverage, annotation workflow, agreement checks, and pilot success criteria.

**Used by:** Pilot dataset construction, human annotation, quality analysis, and later dataset-scaling decisions.

**Must NOT:** Be treated as the final training corpus design or as evidence of final model performance.

### `rrm/rmc_pilot_v0.1.jsonl`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Canonical JSONL artifact for the RMC v0.1 pilot dataset.

**Used by:** Dataset validation, annotation testing, pilot statistics, and later controlled experiments.

**Must NOT:** Contain unlabeled third-party data without source approval or be presented as genuine platform reviews when records were created for research.

---

### `rrm/validate_pilot.py`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Validates RMC pilot schema rules, allowed values, label encodings, unique record IDs, and basic dataset statistics.

**Used by:** Pilot dataset construction and quality control.

**Must NOT:** Modify labels, generate reviews, train models, or implement moderation decisions.

### `rrm/annotate_pilot.py`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Provides a blind human-annotation workflow for the RMC pilot while hiding synthetic construction labels, provenance, and notes from annotators.

**Used by:** Pilot annotation, agreement analysis, adjudication, and creation of human-reviewed RMC labels.

**Must NOT:** Expose seed labels to annotators, infer deception from text alone, modify the source seed corpus, or perform model training.

### `rrm/pilot_annotations.jsonl`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Stores independent human annotations for RMC pilot records.

**Used by:** Agreement measurement, adjudication, annotation-quality analysis, and later gold-label construction.

**Must NOT:** Contain synthetic construction labels presented as human judgments or overwrite the original seed corpus.

### `rrm/analyze_agreement.py`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Measures pilot inter-annotator agreement on overlapping A1/A2 annotations using raw agreement, Cohen's kappa, class distributions, and disagreement review IDs.

**Used by:** Annotation-quality analysis, guide refinement, and adjudication planning.

**Must NOT:** Treat deception as text-derived ground truth, modify annotations, or make final model-performance claims.

### Current foundation files

- `rrm/RESEARCH_CONTRACT.md`
- `rrm/DATASET_SPEC.md`
- `rrm/ANNOTATION_GUIDE.md`
- `rrm/DATA_SOURCE_STRATEGY.md`
- `rrm/PILOT_DATASET_DESIGN.md`
- `rrm/rmc_pilot_v0.1.jsonl`
- `rrm/validate_pilot.py`
- `rrm/annotate_pilot.py`
- `rrm/pilot_annotations.jsonl`
- `rrm/analyze_agreement.py`
- `rrm/requirements.txt`
- `rrm/runtime_check.py`

**Layer:** Layer 3 — RRM

**Purpose:** Pins the currently approved direct Python dependencies for the RRM research environment.

**Used by:** RRM development, experiments, testing, and environment recreation.

**Must NOT:** Become a dumping ground for dependencies that have not been explicitly approved.

### `rrm/runtime_check.py`

**Layer:** Layer 3 — RRM

**Purpose:** Validates the locked Python runtime, approved core packages, and current CUDA/GPU availability.

**Used by:** Environment setup and troubleshooting.

**Must NOT:** Contain model architecture, training, dataset, or trust logic.

### Current foundation files

- `rrm/RESEARCH_CONTRACT.md`
- `rrm/requirements.txt`
- `rrm/runtime_check.py`

Expected responsibilities will eventually include:

- research configuration
- dataset handling
- annotation support
- deterministic pre-checks
- duplicate/similarity evidence
- baseline models
- tokenizer
- Transformer encoder
- Character CNN
- feature fusion
- multi-task heads
- training
- evaluation
- robustness testing
- inference

Exact files must be added incrementally.

Do **not** generate the entire RRM folder tree in advance.

Every new file must have:

- one clear responsibility
- an approved reason to exist
- an entry in this file

---

## Layer 4 — `trust/`

**Ownership:** Trust & Safety.

**Current state:** Placeholder only unless activated.

**Currently existing tracked placeholder:** `trust/.gitkeep`

Future responsibilities may include:

- individual risk aggregation
- behavioural analysis
- temporal analysis
- coordination analysis
- risk resolution
- moderation decisions
- campaign detection
- reputation signals

Layer 4 consumes RRM signals.

It does not belong inside `rrm/`.

Do not implement Trust algorithms while Layer 3 is the active scope unless explicitly authorized.

---

## Layer 5 — `ops/`

**Ownership:** Data, Security, and Operations.

**Current state:** Placeholder only unless activated.

**Currently existing tracked placeholder:** `ops/.gitkeep`

Future responsibilities include:

- PostgreSQL
- Redis
- security configuration
- authentication infrastructure
- WebSocket infrastructure
- audit
- backups
- monitoring
- model storage
- deployment
- operational performance

Do not pre-create operational infrastructure during RRM research unless explicitly required.

---

## Legacy Implementation

The legacy implementation is intentionally outside the active repository.

Known local backup created during migration:

`C:\Projects\RateMyCollezzz_LEGACY_20260922_115705`

A ZIP backup also exists locally.

This path is local recovery information, not a portable repository dependency.

Legacy code is reference and migration material.

Migration rule:

```text
Inspect
  ↓
Understand
  ↓
Classify
  ↓
Preserve useful behaviour
  ↓
Simplify where justified
  ↓
Migrate deliberately
```

Legacy code must not be blindly copied.

Legacy functionality must not be blindly discarded.

---

### `rrm/baseline_tfidf_lr.py`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Implements a reproducible word-level TF-IDF + Logistic Regression baseline for six primary risk labels.  Serves as a comparator for the custom RRM.

**Used by:** rrm/tests/test_baseline_tfidf_lr.py; evaluation pipeline.

**Must NOT:** Consume Trust decisions, similarity scores, PII counts, or moderation features.  Not a production classifier.

### `rrm/tests/test_baseline_tfidf_lr.py`

**Layer:** Layer 3 — Review Intelligence / RRM

**Purpose:** Pytest test suite for the TF-IDF + Logistic Regression baseline, covering config validation, training, leakage guards, evaluation metrics, and label masking.

**Used by:** CI, local validation, development correctness checks.

**Must NOT:** Contain synthetic-pilot performance claims, moderation threshold logic, or Trust-layer tests.

## File Registration Format

Whenever a meaningful new file is approved, document it using:

```md
### `path/to/file.py`

**Layer:** Layer X

**Purpose:** One concise description.

**Used by:** Relevant components.

**Must NOT:** Responsibility that belongs elsewhere.
```

This format keeps ownership obvious.
