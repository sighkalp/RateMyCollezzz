# RateMyCollezzz — Implementation Plan

## 1. Project State

Project foundation:

**LOCKED**

Git baseline:

**ESTABLISHED**

Legacy implementation:

**BACKED UP AND PRESERVED**

Current active layer:

**Layer 3 — Review Intelligence / RRM**

Current priority:

Build, understand, train, evaluate, and package the Review Risk Model.

Layer 1 UI work is currently paused.

Layer 2 platform work is currently paused.

Layer 4 trust implementation is future work.

Layer 5 production integration is future work.

---

# 2. Why Layer 3 Is Active First

Layer numbering represents system responsibility.

It does not define mandatory coding order.

The RRM can be researched and developed independently using:

- datasets
- annotation rules
- local training
- evaluation datasets
- model checkpoints
- controlled inference
- defined input/output contracts

Platform integration happens later.

---

# 3. RRM Development Sequence

## RRM 3.1 — Research Contract

Define:

- exact research problem
- RRM purpose
- RRM limitations
- label taxonomy
- evidence requirements
- research questions
- six core-paper contributions
- experiment requirements
- claims that must NOT be made

No final-model implementation begins before this is clear.

---

## RRM 3.2 — Dataset Foundation

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

Dataset work includes:

- annotation guide
- positive examples
- negative examples
- borderline cases
- confusion rules
- annotation disagreement rules
- source tracking
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

---

## RRM 3.3 — Baselines

Implement and evaluate:

1. TF-IDF + Logistic Regression
2. Character n-gram baseline
3. BERT fine-tuning baseline
4. RoBERTa fine-tuning baseline

Purpose:

Establish evidence that the custom RRM provides justified value.

The custom RRM must not automatically be assumed superior.

---

## RRM 3.4 — RMC Tokenizer

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

Tokenizer decisions must be measured rather than assumed.

---

## RRM 3.5 — Semantic Encoder

Implement the compact Transformer encoder.

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

## RRM 3.6 — Character Branch

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

`goooood`

`b3st`

`achiiii`

`facultyyyy`

The Character CNN must later be validated through ablation.

---

## RRM 3.7 — Feature Fusion

Combine:

Transformer Representation
+
Character Representation

The fusion method must remain simple initially.

Do not introduce unnecessarily complex fusion mechanisms before establishing
a working baseline.

Fusion design must later be tested experimentally.

---

## RRM 3.8 — Multi-task Heads

Create separate prediction heads for:

- spam
- deception
- toxicity
- advertising
- off-topic
- PII

Optional experimental head:

- AI-like writing signal

Study:

- multi-task learning
- shared representation
- binary classification heads
- class imbalance
- loss functions
- task weighting
- regularization
- dropout
- threshold selection
- calibration where useful

---

## RRM 3.9 — Scientific Evaluation

Required experiments include:

### Baseline Comparison

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

---

### Character CNN Ablation

Compare:

Full RRM

vs

RRM without Character CNN

Measure:

- Macro-F1 difference
- noisy-text performance
- robustness to spelling variation

---

### Tokenizer Experiment

Compare:

RMC SentencePiece tokenizer

vs

appropriate pretrained/default tokenizer

Measure:

- token fertility
- truncation frequency
- vocabulary behaviour
- downstream performance

---

### Multi-task Experiment

Compare where practical:

Shared multi-task RRM

vs

separate classifiers

Measure:

- per-label F1
- parameter count
- inference cost
- model size
- training behaviour

---

### Robustness Evaluation

Test controlled perturbations including:

- spelling noise
- repeated characters
- Roman Hindi variation
- leetspeak where justified
- simple character obfuscation

Measure performance degradation.

---

### Efficiency Evaluation

Measure:

- trainable parameter count
- model artifact size
- inference latency
- p50 latency where practical
- p95 latency where practical
- throughput
- memory usage where practical

---

### Reliability

Use:

- fixed seeds
- reproducible configuration
- multiple-run variability where practical
- confidence intervals where justified
- error analysis

No performance claim may be written before actual results exist.

---

## RRM 3.10 — Packaging and Layer Contract

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
  "similarity_signal": 0.0
}