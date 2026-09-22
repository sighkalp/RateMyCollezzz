# RateMyCollezzz

RateMyCollezzz is a student-first college discovery, review, community, and trust platform.

The project uses a controlled five-layer architecture designed to keep the codebase compact, understandable, research-grounded, and maintainable.

## Current Development Focus

**Active layer:** Layer 3 — Review Intelligence / RRM

**Active component:** RRM 3.1 — Research Contract

The immediate priority is to design, implement, train, evaluate, and package the Review Risk Model before returning to the paused UI/platform work.

---

## Architecture

### Layer 1 — Experience

**Folder:** `experience/`

Owns:

- Discover UI
- search
- filters
- college details
- reviews UI
- map interaction
- 2D map navigation
- 3D perspective navigation
- gallery
- compare
- saved colleges
- community experience

### Layer 2 — Platform Core

**Folder:** `platform/`

**Primary technology:** Django + Python

Owns:

- accounts
- colleges
- reviews
- ratings
- communities
- platform APIs
- permissions
- business logic

### Layer 3 — Review Intelligence / RRM

**Folder:** `rrm/`

Owns review understanding and risk/evidence generation.

Current research direction combines:

- SentencePiece tokenization
- compact Transformer encoder
- Character CNN
- feature fusion
- multi-task prediction

Target review-risk signals include:

- spam
- deception
- toxicity
- advertising
- off-topic
- PII
- duplicate/similarity indicators

The current custom-model size target is approximately **22–30M trainable parameters**.

This is a design target, not a proven optimum.

### Layer 4 — Trust & Safety

**Folder:** `trust/`

Consumes model and platform evidence.

Future responsibilities include:

- behavioural analysis
- coordination analysis
- risk aggregation
- moderation decisions
- campaign detection

Core principle:

**Layer 3 understands.**

**Layer 4 decides.**

The RRM does not autonomously delete reviews.

### Layer 5 — Data / Security / Operations

**Folder:** `ops/`

Owns operational infrastructure including:

- PostgreSQL
- Redis
- security
- audit
- backups
- monitoring
- deployment
- model storage

---

## Review Risk Model

The RRM is being designed for student-review language including:

- English
- Hinglish
- Roman Hindi
- Indian English
- college-specific terminology

Conceptual subsystem:

```text
Review
  ↓
Validation + deterministic pre-checks
  ↓
Duplicate / similarity evidence
  ↓
RMC tokenizer
  ↓
Transformer Encoder + Character CNN
  ↓
Feature Fusion
  ↓
Multi-task Risk Heads
  ↓
RRM Risk / Evidence Signals
```

The model is evaluated scientifically against simpler and pretrained baselines before custom-model claims are made.

The final custom-encoder training strategy must be explicitly decided and justified before final training.

---

## Research Principles

The project does not fabricate:

- dataset sizes
- annotation counts
- model results
- F1 scores
- accuracy
- AUPRC
- latency
- throughput
- robustness improvements
- novelty claims

Design targets remain design targets until experiments provide evidence.

The custom RRM must justify its complexity through:

- baseline comparison
- ablation
- robustness testing
- efficiency testing
- error analysis

Important model decisions are classified as:

- PAPER-DERIVED
- RRM EXPERIMENTAL DESIGN CHOICE
- STANDARD ENGINEERING PRACTICE

---

## Core RRM Paper Foundation

The current six-paper foundation is:

1. Vaswani et al. — *Attention Is All You Need*
2. Devlin et al. — BERT
3. Liu et al. — RoBERTa
4. Kudo & Richardson — SentencePiece
5. Zhang, Zhao & LeCun — Character-level CNN for Text Classification
6. Ott et al. — Deceptive Opinion Spam

Additional papers may influence future Layer-4 architecture without being implemented prematurely.

---

## UI Direction

The existing working UI has been preserved as legacy migration material.

Future Layer-1 development will preserve working behaviour while improving architecture and visual quality.

The map experience supports:

- 2D navigation
- 3D perspective navigation
- college-logo location pins
- selected-pin emphasis
- high map clarity

3D perspective is for navigation.

It does not fabricate college buildings or campus infrastructure.

---

## Repository Structure

Only five architectural root folders are used:

```text
RateMyCollezzz/
├── experience/
├── platform/
├── rrm/
├── trust/
└── ops/
```

Control files live at repository root.

---

## Start Here

Read in this order:

1. `ARCHITECTURE.md`
2. `IMPLEMENTATION_PLAN.md`
3. `FILE_MAP.md`
4. `CLAUDE.md`

`ARCHITECTURE.md` explains what the system is.

`IMPLEMENTATION_PLAN.md` explains what is currently being built.

`FILE_MAP.md` explains where everything lives.

`CLAUDE.md` controls implementation-agent behaviour.

---

## Legacy Project

The previous implementation is safely preserved outside the clean repository.

Useful functionality will be migrated deliberately.

The old architecture will not be copied blindly.

Working behaviour will not be discarded blindly.

---

## Development Philosophy

RateMyCollezzz prioritizes:

- compact folder structure
- understandable files
- explicit ownership
- minimal unnecessary abstraction
- reproducible experiments
- research-grounded model development
- preservation of working functionality
- clean layer boundaries
- measurable evidence before claims
