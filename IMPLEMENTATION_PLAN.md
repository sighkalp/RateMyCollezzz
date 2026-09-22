# RateMyCollezzz — Implementation Plan

## 1. Project State

**Project foundation:** LOCKED

**Git baseline:** ESTABLISHED

**Legacy implementation:** BACKED UP AND PRESERVED

**Current active layer:** Layer 3 — Review Intelligence / RRM

**Current active component:** RRM 3.3 - Deterministic Prechecks & Similarity

**Current priority:** Build, understand, train, evaluate, and package the Review Risk Model.

Layer 1 UI work is currently paused.

Layer 2 platform work is currently paused.

Layer 4 trust implementation is future work.

Layer 5 production integration is future work.

---

## 2. Why Layer 3 Is Active First

Layer numbering represents system responsibility.

It does not define mandatory coding order.

The RRM can be researched and developed independently using:

- datasets
- annotation rules
- deterministic review checks
- local training
- evaluation datasets
- model checkpoints
- controlled inference
- defined input/output contracts

Platform integration happens later.

---

## 3. RRM Development Sequence

### RRM 3.1 — Research Contract

Define and lock:

- exact research problem
- RRM purpose
- RRM limitations
- label taxonomy
- evidence requirements
- research questions
- core-paper contribution matrix
- experiment requirements
- claims that must not be made
- distinction between rule-derived and learned signals
- final-model training-strategy decision criteria

Required six-paper foundation:

1. Vaswani et al. — _Attention Is All You Need_
2. Devlin et al. — BERT
3. Liu et al. — RoBERTa
4. Kudo & Richardson — SentencePiece
5. Zhang, Zhao & LeCun — Character-level CNN for Text Classification
6. Ott et al. — Deceptive Opinion Spam

No final-model implementation begins before this contract is clear.

---

### RRM 3.2 — Dataset Foundation

Design and build the RMC dataset.

Target language coverage:

- English
- Hinglish
- Roman Hindi
- Indian English
- college-domain terminology

Primary labels:

- spam
- deception
- toxicity
- advertising
- off-topic
- PII

Dataset schema should support, where applicable:

- `review_id`
- `review_text`
- `language_mix`
- optional `college_category`
- binary risk labels
- `source_type`
- annotation metadata
- split membership/version metadata

`deception` must only be assigned when defensible ground truth exists.

Dataset work includes:

- annotation guide
- positive examples
- negative examples
- borderline cases
- confusion rules
- annotation disagreement rules
- source tracking
- source/licensing review where required
- consent where required
- de-identification where required
- multiple annotators on overlapping samples
- annotation agreement
- train / validation / test strategy
- duplicate leakage prevention
- frozen dataset version
- frozen split seed
- label distribution reporting

Dataset size must never be fabricated.

No result may be reported from an unfrozen or ambiguously split dataset as if it were final evidence.

---

### RRM 3.3 — Deterministic Pre-checks and Similarity Foundation

Implement only the approved non-neural evidence needed before learned classification.

Potential responsibilities:

- input validation
- malformed/empty review handling
- obvious URL/advertising patterns
- deterministic PII patterns where appropriate
- exact duplicate detection
- basic near-duplicate/similarity baseline
- evidence reason codes

Rules must remain distinguishable from neural predictions.

Duplicate detection must not rely only on ML.

This stage does not make moderation decisions.

---

### RRM 3.4 — Baselines

Implement and evaluate:

1. TF-IDF + Logistic Regression
2. Character n-gram baseline
3. BERT fine-tuning baseline
4. RoBERTa fine-tuning baseline

Purpose:

Establish evidence that the custom RRM provides justified value.

The custom RRM must not automatically be assumed superior.

All baseline preprocessing and split usage must be reproducible.

---

### RRM 3.5 — RMC Tokenizer

Implement and evaluate SentencePiece-based tokenization.

Study:

- subword tokenization
- vocabulary size
- English handling
- Hinglish handling
- Roman Hindi handling
- Indian English
- college terminology
- spelling variation
- token fertility
- truncation
- unknown-character handling
- byte fallback where justified

Current design direction:

- vocabulary approximately 20–24K
- sequence window around 256 tokens

These are design targets, not fixed truths.

Tokenizer decisions must be measured rather than assumed.

---

### RRM 3.6 — Encoder Training Strategy Gate

Before training the final custom semantic encoder, explicitly decide how it obtains language knowledge.

Candidate strategies may include:

- training the custom encoder from scratch
- domain/language pretraining followed by task fine-tuning
- distillation from a stronger teacher
- adaptation of an existing compact encoder

This decision affects:

- data requirements
- compute
- scientific novelty
- model ownership
- training time
- expected performance

No strategy is selected merely because it is convenient.

BERT and RoBERTa remain pretrained baselines regardless of the final custom-model strategy.

Any final-model strategy that materially changes the agreed custom-RRM architecture requires explicit approval.

---

### RRM 3.7 — Semantic Encoder

Implement the compact Transformer encoder after the training-strategy gate is resolved.

Required concepts to understand before locking the component:

- token embeddings
- positional information
- self-attention
- Query
- Key
- Value
- dot product
- scaled dot-product attention
- softmax
- multi-head attention
- residual connections
- LayerNorm
- feed-forward network
- encoder blocks
- pooling
- contextual representations

Current design direction:

- approximately 8 encoder layers
- hidden dimension around 384
- approximately 6 attention heads
- FFN dimension around 1536
- sequence length around 256

These are experimental design targets.

They are not final claims.

---

### RRM 3.8 — Character Branch and Feature Fusion

Implement Character CNN.

Purpose:

Capture character-level patterns including:

- spelling variation
- repeated characters
- obfuscation
- noisy typing
- Romanized variation
- character manipulation
- informal student writing

Examples:

- `goooood`
- `b3st`
- `achiiii`
- `facultyyyy`

Then combine:

```text
Transformer representation
        +
Character representation
        ↓
Feature fusion
```

The initial fusion method must remain simple.

Do not introduce unnecessarily complex fusion mechanisms before establishing a working baseline.

The Character CNN must later be validated through ablation.

---

### RRM 3.9 — Multi-task Heads and Training

Create separate prediction heads for:

- spam
- deception
- toxicity
- advertising
- off-topic
- PII

Optional experimental head:

- AI-like writing signal

Study and explicitly document:

- multi-task learning
- shared representation
- binary classification heads
- class imbalance
- loss functions
- task weighting
- regularization
- dropout
- optimizer
- learning-rate schedule
- batch size
- early stopping
- checkpoint selection
- threshold selection
- calibration where useful
- random seeds

Every important choice must be classified as:

- PAPER-DERIVED
- RRM EXPERIMENTAL DESIGN CHOICE
- STANDARD ENGINEERING PRACTICE

---

### RRM 3.10 — Scientific Evaluation

Required experiments include:

#### Baseline comparison

Evaluate:

- TF-IDF + Logistic Regression
- Character n-gram baseline
- BERT
- RoBERTa
- RRM without Character CNN
- Full RRM

Primary metrics:

- Macro-F1
- per-label F1
- precision
- recall
- AUPRC

Accuracy must not be used as the only evaluation metric.

#### Character CNN ablation

Compare:

- Full RRM
- RRM without Character CNN

Measure:

- Macro-F1 difference
- noisy-text performance
- robustness to spelling variation

#### Tokenizer experiment

Compare:

- RMC SentencePiece tokenizer
- appropriate pretrained/default tokenizer

Measure:

- token fertility
- truncation frequency
- vocabulary behaviour
- downstream performance

#### Multi-task experiment

Compare where practical:

- shared multi-task RRM
- separate classifiers

Measure:

- per-label F1
- parameter count
- inference cost
- model size
- training behaviour

#### Robustness evaluation

Test controlled perturbations including:

- spelling noise
- repeated characters
- Roman Hindi variation
- leetspeak where justified
- simple character obfuscation

Measure performance degradation.

#### Efficiency evaluation

Measure:

- trainable parameter count
- model artifact size
- inference latency
- p50 latency where practical
- p95 latency where practical
- throughput
- memory usage where practical

#### Reliability and error analysis

Use:

- fixed seeds
- reproducible configuration
- multiple-run variability where practical
- confidence intervals where justified
- false-positive analysis
- false-negative analysis
- language-mix error analysis
- per-label confusion analysis

No performance claim may be written before actual results exist.

---

### RRM 3.11 — Packaging and Layer Contract

Produce a clean inference interface.

Conceptual output:

```json
{
  "spam_score": 0.0,
  "deception_score": 0.0,
  "toxicity_score": 0.0,
  "advertising_score": 0.0,
  "off_topic_score": 0.0,
  "pii_score": 0.0,
  "similarity_signal": 0.0,
  "reason_codes": []
}
```

The exact schema must be finalized from real implementation needs.

Layer 3 stops at risk/evidence outputs.

Layer 3 must **not** output:

- DELETE
- BAN
- REMOVE USER
- FINAL MODERATION DECISION

Layer 4 consumes Layer-3 outputs later.

---

## 4. Future Research Awareness

Future Trust-layer research may involve:

- collective opinion spam
- temporal behaviour
- account coordination
- rating patterns
- campaign detection
- graph relationships
- reputation signals

Future requirements may influence:

- metadata retention
- timestamps
- stable anonymized identifiers where ethically appropriate
- similarity outputs
- probability outputs
- audit-friendly reason codes

Future algorithms must not be implemented during current RRM work.

Rule:

**ARCHITECTURAL AWARENESS NOW.**

**ALGORITHMIC IMPLEMENTATION LATER.**

---

## 5. Layer 1 Status

Layer 1 is **PAUSED**, not abandoned.

Known locked requirements:

- preserve working legacy frontend behaviour
- map-centric Discover interface
- future UI visual direction already established
- improved map clarity
- improved map colour hierarchy
- college logos inside location pins
- strong selected-pin state
- 2D map mode
- 3D perspective map mode
- no fabricated campus buildings
- search
- filters
- college details
- reviews
- gallery
- save
- compare

Do not alter Layer 1 during current RRM development unless explicitly activated.

---

## 6. Current Allowed Work

Allowed:

- RRM research
- research-paper analysis
- dataset design
- annotation design
- deterministic RRM pre-check design
- RRM code
- RRM tests
- RRM experiments
- RRM configuration
- RRM evaluation
- RRM inference interface
- RRM-related documentation updates when explicitly approved

---

## 7. Current Forbidden Work

Do **not** implement:

- Experience UI
- Django Platform Core
- Trust Engine
- behavioural detection
- coordination engine
- campaign detection
- community
- chat
- advertising
- deployment infrastructure

unless explicitly activated later.

---

## 8. Component Workflow

Every RRM component follows:

1. Define exactly what is being built.
2. Explain every important concept.
3. Identify research-paper provenance.
4. Identify architecture ownership.
5. Identify allowed files.
6. Identify forbidden files.
7. Identify allowed dependencies.
8. Implement only approved scope.
9. Run focused tests.
10. Inspect changed files.
11. Run broader validation when appropriate.
12. Explain the implementation deeply.
13. Verify experiment outputs where relevant.
14. Update `FILE_MAP.md`.
15. Lock the component.
16. Move forward.

---

## 9. Learning / Teaching Gate

A component is not considered learned merely because it runs.

After each substantial RRM component, the explanation must cover:

- what it is
- why it exists
- how it works internally
- inputs
- outputs
- tensor/data shapes where relevant
- exact code location
- important classes/functions
- hyperparameters
- relevant mathematics
- paper contribution
- what was borrowed from the paper
- what is our own experimental design choice
- alternatives considered
- tests performed
- experiment evidence
- failure modes
- how the next component uses it

Unknown prerequisite concepts must be explained rather than silently assumed.

---

## 10. Completion Rule

A component is complete only when:

- architecture is respected
- code ownership is clear
- unexpected files = zero
- unexpected folders = zero
- unnecessary dependencies = zero
- duplicate logic = zero
- tests pass
- experiments are reproducible
- `FILE_MAP.md` is current
- implementation is understood
- claims are supported by evidence
- no unexplained scope deviation exists

Only then may the component be marked **LOCKED**.
