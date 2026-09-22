# RateMyCollezzz RRM — Dataset Source Strategy

## 1. Purpose

This document defines where Review Model Corpus (RMC) data may come from,
what each source may legitimately contribute, and how source provenance must
be handled.

The goal is not to collect the largest possible dataset.

The goal is to build a dataset whose:

- labels are defensible
- provenance is known
- licensing can be verified
- English/Hinglish college-domain coverage is meaningful
- train/validation/test separation is scientifically valid
- source identity does not become a shortcut for the model

This document must be used together with:

- `RESEARCH_CONTRACT.md`
- `DATASET_SPEC.md`
- `ANNOTATION_GUIDE.md`

---

# 2. Core Source Principle

A dataset may only provide labels that its original task and ground truth
actually support.

Example:

```text
Dataset contains toxicity annotations
        ↓
May support toxicity research
        ↓
Must NOT automatically become deception,
spam, advertising, off-topic, or PII ground truth
```

The project must never invent missing labels simply to make external datasets
fit the RRM schema.

---

# 3. Source Classes

RMC may contain data from four major source classes.

## 3.1 External Research Datasets

Previously published datasets used for:

- baseline development
- auxiliary training
- representation learning
- robustness experiments
- comparison with prior work

External datasets may not match the college-review domain.

Their domain difference must remain documented.

---

## 3.2 Human-Written RMC Research Data

College-review examples written specifically for RMC research.

These should provide domain coverage for:

- English
- Hinglish
- Roman Hindi
- Indian English
- college terminology
- realistic student phrasing

They must not be falsely represented as genuine public student reviews if
they were created for research.

---

## 3.3 Controlled Research Data

Samples produced under conditions where the true target is known.

Controlled data is particularly useful where ordinary text does not provide
sufficient ground truth.

Examples may include:

- deliberately deceptive writing tasks
- deliberate advertising tasks
- controlled off-topic examples
- controlled toxicity examples
- spelling and character perturbation experiments

Controlled samples must remain identifiable through provenance metadata.

---

## 3.4 Real Platform or Public College Reviews

Real college-review text may later provide important domain realism.

However, it requires additional care regarding:

- permission and terms of use
- copyright
- privacy
- PII
- provenance
- annotation quality
- deception uncertainty
- redistribution restrictions

No public review source should be imported into RMC until its use has been
reviewed and approved.

---

# 4. Candidate Source Registry

The current candidate source families are:

| Source                                            | Primary Potential Use                 | Domain Match | Current Status                            |
| ------------------------------------------------- | ------------------------------------- | ------------ | ----------------------------------------- |
| Deceptive Opinion Spam Corpus                     | Deception research                    | Low/Medium   | Candidate                                 |
| Jigsaw Toxic Comment Classification               | Toxicity                              | Low          | Candidate                                 |
| Jigsaw Unintended Bias in Toxicity Classification | Toxicity / robustness / bias analysis | Low          | Candidate                                 |
| YelpCHI / related Yelp review-spam research data  | Spam-related research                 | Medium       | Candidate requiring ground-truth audit    |
| Custom RMC English/Hinglish Corpus                | Main college-domain corpus            | High         | Planned                                   |
| Controlled RMC research samples                   | Known-ground-truth experiments        | High         | Planned                                   |
| Real college-review sources                       | Domain realism                        | High         | Future, source-specific approval required |

`Candidate` does not mean automatically approved for ingestion.

Every source requires a source audit before use.

---

# 5. Source Audit Requirements

Before any external dataset enters RMC experiments, record:

| Field                    | Meaning                                        |
| ------------------------ | ---------------------------------------------- |
| `source_name`            | Official dataset name                          |
| `source_version`         | Version/release if available                   |
| `source_url`             | Official or authoritative location             |
| `paper_reference`        | Related publication                            |
| `original_task`          | What the dataset was originally built to study |
| `original_labels`        | Exact original label semantics                 |
| `language`               | Languages represented                          |
| `domain`                 | Reviews/comments/etc.                          |
| `license`                | Verified license or terms                      |
| `redistribution_allowed` | Whether raw redistribution is permitted        |
| `pii_risk`               | Known or possible privacy risk                 |
| `duplicate_risk`         | Duplicate/template concerns                    |
| `approved_rrm_labels`    | Which RRM labels it may support                |
| `forbidden_rrm_labels`   | Labels it must not be used to infer            |
| `split_policy`           | How it may enter train/validation/test         |
| `notes`                  | Important limitations                          |

License or terms must be verified from the authoritative source before
distribution or publication.

Do not guess licensing conditions.

---

# 6. Deceptive Opinion Spam Corpus

## Intended RRM Role

Primary potential use:

```text
deception
```

This dataset is valuable because it contains controlled or otherwise
research-defined deceptive/genuine review conditions.

It provides stronger deception ground truth than ordinary unlabeled online
reviews.

---

## Domain Limitation

Its original domain is not Indian college reviews.

Therefore:

```text
useful deception evidence
        ≠
college-domain representativeness
```

Performance on this dataset must not be presented as proof that deception
detection works equally well on Indian college reviews.

---

## Approved Potential Use

May support:

```text
deception
```

where original provenance genuinely supports the distinction.

May also be useful for:

- deception baseline experiments
- representation experiments
- domain-transfer analysis

---

## Must NOT Automatically Support

Do not automatically derive:

```text
spam
toxicity
advertising
off_topic
pii
```

unless those targets are independently annotated.

---

## Split Rule

If used for training, its records must not leak into RMC evaluation through:

- copies
- paraphrases
- derived variants
- translated versions

Any transformed version of a source sample must remain grouped with the
original during splitting.

---

# 7. Jigsaw Toxic Comment Dataset

## Intended RRM Role

Primary potential use:

```text
toxicity
```

It provides large-scale examples of harmful or toxic language.

---

## Domain Limitation

The source domain is general online discussion rather than college reviews.

Therefore it may teach toxicity-related language but does not provide direct
college-domain coverage.

---

## Approved Potential Use

May support:

```text
toxicity
```

It may also support robustness experiments for toxic-language detection.

---

## Must NOT Automatically Support

Do not infer:

```text
spam
deception
advertising
off_topic
pii
```

from its toxicity annotations.

---

## RMC Use

Jigsaw toxicity data should be treated as external auxiliary data rather than
the entire basis of the RMC toxicity task.

The final RMC evaluation must contain college-domain toxicity examples where
possible.

---

# 8. Jigsaw Unintended Bias Dataset

## Intended RRM Role

Potential uses include:

```text
toxicity
toxicity robustness
bias/error analysis
```

Its value is particularly relevant when studying whether toxicity systems
behave poorly around identity-related language or particular linguistic
patterns.

---

## Important Restriction

Identity-related metadata must not be casually converted into RRM risk labels.

The presence of an identity term does not imply toxicity.

---

## Approved Potential Use

May contribute to:

- toxicity training
- toxicity validation experiments
- false-positive analysis
- robustness analysis

---

## Must NOT Automatically Support

Do not derive:

```text
deception
spam
advertising
off_topic
pii
```

without independent labels.

---

# 9. YelpCHI / Yelp Review-Spam Research Data

## Intended RRM Role

Potentially useful for:

```text
spam-related review research
review anomaly research
```

However, this source requires a careful ground-truth audit before use.

---

## Critical Label Rule

The project's terminology must follow the original dataset's actual label
meaning.

If the original source uses:

- filtered reviews
- suspicious reviews
- recommended/not-recommended status
- platform-derived indicators
- review-spam proxies

those must not automatically be renamed:

```text
deception = 1
```

or treated as certain deliberate fraud.

---

## Potential RRM Mapping

Depending on the verified original label semantics, the source may support:

```text
spam
```

or an auxiliary spam/anomaly experiment.

The exact mapping must be documented after source verification.

---

## Must NOT Automatically Support

Do not assume support for:

```text
deception
toxicity
advertising
off_topic
pii
```

unless independently justified.

---

# 10. Custom RMC English/Hinglish Corpus

This is the most important source for domain-specific RRM development.

It should represent actual language patterns relevant to Indian students and
college-review writing.

---

## Target Language Coverage

The corpus should intentionally include:

```text
ENGLISH
HINGLISH
ROMAN_HINDI
```

with natural variation.

Examples of relevant variation include:

```text
"The faculty is supportive but placements are average."

"Faculty supportive hai but placement scene average hai."

"Teachers kaafi helpful hain lekin placement thoda weak hai."
```

The goal is not literal translation of one sentence into multiple languages.

The goal is natural language diversity.

---

# 11. Custom RMC Domain Coverage

The custom corpus should contain realistic discussion of topics such as:

```text
ACADEMICS
FACULTY
PLACEMENTS
HOSTEL
INFRASTRUCTURE
FEES
ADMINISTRATION
CAMPUS_LIFE
ADMISSIONS
MULTI_TOPIC
```

Risk labels should appear inside realistic college-review contexts where
possible.

Example:

A toxicity example should ideally look like college-review toxicity rather
than an unrelated generic social-media insult.

---

# 12. Custom RMC Label Support

The custom corpus is expected to provide the strongest domain-specific
coverage for:

```text
spam
toxicity
advertising
off_topic
pii
```

Deception is different.

---

# 13. Deception in Custom RMC

Ordinary human-written college reviews should generally use:

```text
deception = -1
```

unless reliable provenance establishes the truth status.

Do not label a review deceptive merely because it:

- appears exaggerated
- is unusually positive
- is unusually negative
- sounds polished
- looks machine-generated
- seems suspicious

For known deception examples, use controlled research design or another
source with defensible provenance.

---

# 14. Controlled RMC Research Samples

Controlled samples may be created to establish known target conditions.

Examples:

```text
known deceptive task
known advertising task
known off-topic task
known spam-pattern task
known toxicity task
```

Each controlled sample must record:

```text
source_type = CONTROLLED_RESEARCH
```

or the equivalent approved provenance value.

---

# 15. Human-Written Research Samples

Participants may be asked to write realistic college-review examples.

These samples should be marked:

```text
source_type = HUMAN_WRITTEN_RESEARCH
```

They must not be represented as genuine platform reviews.

---

## Appropriate Uses

Human-written research examples are especially useful for:

- Hinglish
- Roman Hindi
- Indian English
- college terminology
- borderline annotation cases
- natural student phrasing

---

# 16. Synthetic Data

Synthetic data may be used carefully for:

- controlled perturbations
- rare-label augmentation research
- robustness experiments
- pilot development

It must remain identifiable as:

```text
source_type = SYNTHETIC_RESEARCH
```

---

## Synthetic Data Must NOT

Synthetic data must not:

- dominate the final evaluation set
- masquerade as genuine student reviews
- create duplicate templates across splits
- be used to manufacture high performance
- automatically receive deception labels
- replace real domain evaluation

---

# 17. Translation Policy

Naively translating an English dataset into Hinglish does not create a
genuine Hinglish dataset.

Translation may be useful for controlled experiments, but translated examples
must remain identifiable as derived data.

Example provenance relationship:

```text
original English sample
        ↓
controlled Hinglish transformation
        ↓
same source group
```

The original and transformed version must not appear in different dataset
splits.

---

# 18. Perturbation Data

Robustness variants may be generated from existing samples.

Example:

```text
good faculty
        ↓
goood faculty

best college
        ↓
b3st college
```

These are derived samples.

They must remain grouped with their parent record during dataset splitting.

Otherwise the model may effectively see the same sample during training and
testing.

---

# 19. Source-to-Label Mapping

Initial source policy:

| Source                             |                 Spam |                           Deception |         Toxicity |      Advertising |        Off-topic |              PII |
| ---------------------------------- | -------------------: | ----------------------------------: | ---------------: | ---------------: | ---------------: | ---------------: |
| Deceptive Opinion Spam Corpus      |          Not assumed |      Yes, where provenance supports |               No |               No |               No |               No |
| Jigsaw Toxic Comments              |                   No |                                  No |              Yes |               No |               No |               No |
| Jigsaw Unintended Bias             |                   No |                                  No |              Yes |               No |               No |               No |
| YelpCHI / related review-spam data | Possible after audit |                         Not assumed |               No |               No |               No |               No |
| Human-written RMC                  |     Yes if annotated |                     Usually UNKNOWN |              Yes |              Yes |              Yes |              Yes |
| Controlled RMC                     |                  Yes | Yes where experimentally controlled |              Yes |              Yes |              Yes |              Yes |
| Synthetic RMC                      |    Experimental only |  Only if controlled design warrants |     Experimental |     Experimental |     Experimental |     Experimental |
| Real college reviews               |     Yes if annotated |                     Usually UNKNOWN | Yes if annotated | Yes if annotated | Yes if annotated | Yes if annotated |

This mapping may only change after documented source review.

---

# 20. Source Provenance Must Remain Separate From Model Input

A dangerous example would be:

```text
source_type = JIGSAW
toxicity = 1
```

If the model receives `source_type`, it could learn:

```text
JIGSAW
  ↓
toxicity
```

instead of learning toxic language.

Therefore:

```text
source_type
dataset_name
dataset_version
original_label
annotation_origin
```

must remain research metadata unless a separately approved experiment
requires otherwise.

---

# 21. Source Fingerprint Leakage

Different datasets may have identifiable writing patterns.

Examples include:

- punctuation conventions
- HTML remnants
- formatting
- sentence lengths
- special tokens
- source-specific boilerplate

A classifier might learn these artifacts instead of the intended risk
concept.

Before training, source-specific artifacts must be investigated.

Cleaning must be reproducible and must not erase meaningful language
characteristics without justification.

---

# 22. Dataset Balancing

Do not force every source to contribute the same number of samples.

Do not fabricate examples only to create perfectly balanced labels.

Instead:

- measure natural class distributions
- report imbalance
- use modeling techniques where justified
- perform controlled sampling experiments if needed

Dataset balance is an experimental decision, not an annotation decision.

---

# 23. RMC Test-Set Principle

The most important final evaluation should measure the target problem:

```text
English / Hinglish / Roman-Hindi
college-review risk detection
```

Therefore the main final RMC test set should emphasize in-domain college
reviews rather than being dominated by external generic datasets.

External datasets may maintain their own benchmark evaluation tracks.

---

# 24. Recommended Evaluation Separation

Conceptually maintain two evaluation categories.

## In-Domain RMC Evaluation

Measures performance on college-review language.

This is the primary evidence for RateMyCollezzz.

---

## External Benchmark Evaluation

Measures performance on external datasets according to their own task
definitions.

This helps compare behaviour with established research datasets.

The two types of results must not be silently merged into one metric.

---

# 25. Train/Validation/Test Source Separation

The split strategy must consider source identity.

A naive random row split can cause leakage when:

- multiple samples come from the same template
- transformed versions share a parent
- synthetic examples share prompts
- duplicates exist
- external datasets contain repeated reviews

Where appropriate, related samples must be grouped before splitting.

---

# 26. Parent-Child Grouping

Derived records should preserve a parent relationship internally.

Example:

```text
RMC-000123
    ├── spelling perturbation
    ├── Roman-Hindi variation
    └── repeated-character variation
```

All members of this family should normally remain in the same split.

The final schema may introduce a grouping identifier when the derived-data
pipeline is implemented.

---

# 27. Test-Set Protection

Once the final test set is frozen:

- do not train on it
- do not tune thresholds on it
- do not repeatedly inspect failures to modify the model
- do not augment from it into training
- do not modify it to improve results

Development decisions should primarily use training and validation data.

---

# 28. Source Licensing

Before a dataset is downloaded, redistributed, published, or bundled with
RMC, its license or terms must be verified from an authoritative source.

For every external dataset, record:

```text
license status
redistribution status
citation requirement
commercial/research restrictions
derivative-data restrictions
```

If the permission status is unclear:

```text
do not redistribute the raw dataset
```

until resolved.

---

# 29. Copyright and Publication

The RMC research repository does not automatically gain redistribution rights
for third-party review text.

Where redistribution is prohibited or uncertain, the project may instead
store:

- retrieval instructions
- dataset identifiers
- processing scripts
- derived statistics

where legally permitted.

Exact handling must follow the verified terms of each source.

---

# 30. Privacy and PII

External or real-world sources must be inspected for privacy risk.

Unnecessary real personal identifiers must not be retained merely because PII
is one of the research labels.

For PII training examples, prefer:

- controlled examples
- safely constructed examples
- appropriately de-identified text

where possible.

---

# 31. College-Review Domain Realism

The custom RMC corpus should avoid becoming a collection of simplistic
sentences such as:

```text
college good
college bad
faculty bad
hostel good
```

Realistic examples should contain:

- mixed opinions
- context
- informal expressions
- domain vocabulary
- variable length
- nuanced sentiment
- English/Hinglish variation

Risk classification must not collapse into sentiment classification.

---

# 32. Sentiment Is Not a Risk Label

Examples:

```text
"The placements are terrible and the hostel is overpriced."
```

may legitimately have:

```text
spam = 0
toxicity = 0
advertising = 0
off_topic = 0
pii = 0
deception = -1
```

Negative sentiment does not imply harmful review behaviour.

Similarly, positive sentiment does not imply authenticity.

---

# 33. Source Diversity

RMC should eventually contain variation in:

- writing length
- college topics
- language mix
- spelling style
- positive/negative sentiment
- formal/informal language
- clean/noisy text
- risk combinations

However, diversity must be measured rather than assumed.

---

# 34. Pilot Before Scale

Do not begin by collecting tens of thousands of RMC samples.

First build a small pilot corpus.

The pilot should test whether:

- schema works
- annotation rules work
- source tracking works
- Hinglish categories are usable
- labels are understandable
- deception UNKNOWN behaves correctly
- multi-label examples are practical
- duplicates can be detected
- source leakage can be controlled

Only after the pilot is reviewed should collection scale.

---

# 35. Initial Pilot Source Composition

Exact sample counts are not locked yet.

The pilot should contain a small but diverse mixture of:

```text
clean college reviews
spam-like college reviews
advertising reviews
toxicity cases
off-topic cases
PII cases
multi-label cases
English reviews
Hinglish reviews
Roman-Hindi reviews
ambiguous cases
controlled deception examples
```

The purpose is annotation validation rather than model performance.

---

# 36. External Data Does Not Become RMC Automatically

External datasets should initially remain logically distinct.

Conceptually:

```text
external/
    Deception dataset
    Jigsaw datasets
    Yelp-related dataset

RMC/
    college-domain research corpus
```

This is a conceptual separation.

Do not create these folders yet unless the implementation plan explicitly
approves them.

Experiments may combine sources through reproducible preprocessing later.

---

# 37. Source Transformation Logging

Any transformation applied to source data must be reproducible.

Examples include:

- Unicode normalization
- HTML cleanup
- masking
- de-identification
- language tagging
- deduplication
- controlled perturbation

The raw-source identity must remain traceable through research metadata where
permitted.

---

# 38. No Silent Relabeling

Original dataset labels must not silently be reinterpreted.

Example:

If an external dataset's original label means:

```text
platform-filtered review
```

do not silently convert it to:

```text
confirmed deception
```

Instead document:

```text
original_label
        ↓
RRM mapping decision
        ↓
limitations
```

---

# 39. Source Quality Levels

Sources may later be assigned research-quality categories such as:

```text
STRONG_GROUND_TRUTH
CONTROLLED_GROUND_TRUTH
HUMAN_ANNOTATED
WEAK_LABEL
UNLABELED
```

These categories must describe evidence quality, not model confidence.

The exact implementation is deferred until ingestion design.

---

# 40. RMC Corpus Ownership

The term:

```text
RMC
```

should primarily refer to the project's own curated college-review research
corpus and its controlled derivatives.

Third-party datasets remain third-party datasets even when used in RRM
experiments.

This prevents ambiguity in the research paper.

---

# 41. Dataset Source Strategy Completion Criteria

RRM 3.2C is complete when:

- source classes are defined
- candidate external datasets are identified
- each source has a limited legitimate RRM role
- external-domain limitations are explicit
- deception ground truth is protected
- custom RMC is identified as the main domain corpus
- English/Hinglish/Roman-Hindi coverage is required
- synthetic data limitations are explicit
- transformation leakage is addressed
- provenance is separated from model input
- licensing verification is mandatory
- privacy requirements are defined
- in-domain and external evaluation are separated
- pilot-before-scale policy is established

After this strategy is locked, proceed to:

**RRM 3.2D — Pilot Dataset Design**
