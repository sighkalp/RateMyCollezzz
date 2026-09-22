# RateMyCollezzz RRM — Pilot Dataset Design

## 1. Purpose

This document defines the first controlled pilot for the RateMyCollezzz
Review Model Corpus (RMC).

The pilot exists to validate:

- dataset schema
- annotation definitions
- English/Hinglish/Roman-Hindi handling
- multi-label behaviour
- deception UNKNOWN handling
- annotator agreement
- source tracking
- privacy handling
- duplicate/leakage controls

The pilot is NOT intended to train the final RRM.

It is also NOT intended to produce publishable model-performance claims.

---

# 2. Pilot Objective

The pilot answers one primary question:

> Can we consistently create and annotate realistic English, Hinglish,
> and Roman-Hindi college-review examples using the current RMC schema
> and annotation rules?

The pilot should expose weaknesses before large-scale dataset creation begins.

---

# 3. Pilot Size

Initial target:

```text
150 unique base review records
```

This number is intentionally small enough for manual inspection while large
enough to expose annotation problems.

The pilot size may change after inspection, but expansion requires a
documented reason.

---

# 4. Language Composition

Target composition:

| Language Type | Target Records |
| ------------- | -------------: |
| English       |             50 |
| Hinglish      |             60 |
| Roman Hindi   |             40 |
| **Total**     |        **150** |

This distribution is a pilot design target rather than a claim about the
true population distribution of Indian college reviews.

The purpose is to ensure sufficient exposure to all three primary language
styles.

---

# 5. Why Hinglish Receives the Largest Pilot Share

Hinglish is especially important to the custom RRM because it introduces:

- code mixing
- Roman-script Hindi
- English technical vocabulary
- informal student expressions
- inconsistent spelling
- flexible grammar
- character variation

Example:

```text
Faculty supportive hai but placement scene thoda weak hai.
```

A generic English-only classifier may not represent such language optimally.

The pilot therefore deliberately includes strong Hinglish coverage.

---

# 6. Scenario Composition

The 150 records should be constructed from diverse scenario groups.

Initial collection targets:

| Scenario Group                             | Approximate Count |
| ------------------------------------------ | ----------------: |
| Normal / low-risk college reviews          |                35 |
| Spam-oriented cases                        |                20 |
| Advertising-oriented cases                 |                20 |
| Toxicity-oriented cases                    |                20 |
| Off-topic cases                            |                15 |
| PII-oriented cases                         |                15 |
| Controlled deception cases                 |                10 |
| Ambiguous / borderline / multi-label cases |                15 |
| **Total**                                  |           **150** |

These are collection scenarios.

They are NOT mutually exclusive final labels.

Example:

A review created under the advertising scenario may eventually receive:

```text
advertising = 1
spam = 1
off_topic = 1
```

if the annotation guide supports all three.

Final labels must come from annotation rather than from the scenario name.

---

# 7. Important Scenario-vs-Label Separation

The scenario used to construct or select a sample must not automatically
become its final annotation.

Conceptually:

```text
collection scenario
        ↓
review text
        ↓
independent annotation
        ↓
final labels
```

Not:

```text
scenario = advertising
        ↓
advertising = 1 automatically
```

except where the sample is part of an explicitly controlled task whose
ground truth defines that target.

---

# 8. Source Composition

The first RMC pilot should primarily use project-controlled sources.

Recommended composition:

```text
HUMAN_WRITTEN_RESEARCH
CONTROLLED_RESEARCH
SYNTHETIC_RESEARCH
```

External datasets should not yet be mixed into this first RMC pilot unless
their source audit has been completed.

Real public college reviews should also not enter the pilot until:

- terms are reviewed
- privacy handling is defined
- redistribution policy is understood
- source-specific approval is complete

---

# 9. Human-Written Research Samples

Target approximately:

```text
90–100 records
```

These should resemble realistic college-review writing without pretending to
be genuine public student reviews.

They should include:

- positive experiences
- negative experiences
- mixed experiences
- short reviews
- medium reviews
- longer reviews
- formal language
- informal language
- Hinglish
- Roman Hindi
- spelling variation
- normal student vocabulary

---

# 10. Controlled Research Samples

Target approximately:

```text
30–40 records
```

Controlled samples are especially important where known ground truth is
required.

Possible controlled tasks include:

- intentionally promotional content
- intentionally off-topic content
- intentionally toxic content
- intentionally spam-like content
- intentionally deceptive review writing

Controlled provenance must remain recorded.

---

# 11. Synthetic / Derived Samples

Keep synthetic or derived content limited during the pilot.

Target:

```text
no more than approximately 20% of pilot records
```

unless later experiments explicitly justify more.

Synthetic examples may help test:

- rare cases
- PII patterns
- spelling corruption
- repeated characters
- character substitutions
- robustness

Synthetic content must not dominate the pilot.

---

# 12. Deception Pilot Design

Deception receives special treatment.

Most ordinary college-review examples should use:

```text
deception = -1
```

unless trustworthy provenance establishes their truth status.

The pilot should include approximately:

```text
10 controlled deception-positive examples
```

where participants or controlled generation are explicitly instructed to
write a fabricated experience.

Where possible, include controlled genuine examples as:

```text
deception = 0
```

with defensible provenance.

Do not manufacture deception labels from writing style.

---

# 13. Normal / Low-Risk Examples

The pilot must contain meaningful normal reviews.

Examples should discuss legitimate college topics without forcing a risk
label.

Example:

```text
The faculty is generally supportive and the labs are decent,
but the hostel maintenance needs improvement.
```

Possible labels:

```text
spam = 0
deception = -1
toxicity = 0
advertising = 0
off_topic = 0
pii = 0
```

The model must eventually learn what ordinary criticism looks like.

---

# 14. Mixed-Sentiment Examples

Include reviews containing both praise and criticism.

Example:

```text
Campus kaafi accha hai and teachers helpful hain,
but placement opportunities average hain.
```

Mixed sentiment helps prevent the RRM from learning simplistic shortcuts such
as:

```text
negative sentiment = risky review
```

---

# 15. Spam Cases

Spam-oriented examples should include variation such as:

- repeated text
- repetitive promotion
- low-information repetition
- template-like messages
- meaningless flooding patterns

Do not make every spam example identical.

---

# 16. Advertising Cases

Advertising examples should include different forms:

- consultancy promotion
- coaching promotion
- channel promotion
- website promotion
- referral promotion
- commercial services

Some should overlap with spam.

Others should test cases where:

```text
advertising = 1
spam = 0
```

if justified by annotation rules.

---

# 17. Toxicity Cases

Include toxicity at different linguistic styles.

Examples should include:

- English insults
- Hinglish insults
- Roman-Hindi insults
- hostile but non-toxic criticism
- borderline institutional criticism

The pilot should intentionally test the boundary between:

```text
negative criticism
```

and:

```text
toxicity
```

---

# 18. Off-topic Cases

Include clearly unrelated content and borderline student-life context.

Clearly off-topic example:

```text
Kal cricket match bahut mast tha.
```

Borderline relevant example:

```text
College ke aas paas transport options kaafi ache hain.
```

The second may still be relevant to student experience.

These cases test annotation consistency.

---

# 19. PII Cases

PII examples should prefer controlled or fictional identifiers.

Examples may contain fictional patterns representing:

- phone numbers
- emails
- addresses
- account identifiers

Do not insert real private information into the pilot merely to test PII
detection.

---

# 20. Multi-label Cases

At least:

```text
15–25% of the pilot
```

should contain meaningful multi-label possibilities.

Example:

```text
Guaranteed admission ke liye ABC Consultancy ko call karo at 98XXXXXXXX.
```

Potential labels:

```text
spam = 1
advertising = 1
off_topic = 1
pii = 1
deception = -1
toxicity = 0
```

Exact labels must still be determined through annotation.

---

# 21. Borderline Cases

At least:

```text
15 records
```

should deliberately test difficult boundaries.

Examples:

- harsh criticism vs toxicity
- enthusiastic review vs spam
- useful service mention vs advertising
- surrounding-area discussion vs off-topic
- public institutional contact vs personal PII
- suspicious wording vs deception

Borderline samples are valuable because they expose weaknesses in the
annotation guide.

---

# 22. Review Length Variation

The pilot should not contain only one-sentence examples.

Target a mixture of:

```text
SHORT
MEDIUM
LONG
```

Conceptually:

- Short: one or two short sentences
- Medium: a normal review paragraph
- Long: multi-topic detailed review

Exact token thresholds should not be locked until corpus statistics are
observed.

---

# 23. Natural Language Rule

Human-written examples must sound like plausible student writing.

Avoid constructing the entire dataset from simplistic patterns such as:

```text
faculty good
hostel bad
placement good
college bad
```

Include realistic variation.

Example:

```text
First year mein faculty ka support kaafi acha tha,
especially programming subjects mein. Hostel facilities average hain
and mess food could definitely improve.
```

---

# 24. Do Not Artificially Perfect the Language

Do not automatically correct:

- grammar
- capitalization
- punctuation
- spelling
- Roman-Hindi variants

unless preprocessing experiments explicitly require it.

The raw language is part of the research problem.

---

# 25. Language Diversity

Hinglish and Roman-Hindi records should not all follow one template.

Possible variants:

```text
accha
acha
achha

hai
h
hain

placement
placements
placement scene
placement ka scene
```

Such natural variation is valuable for tokenizer and character-level
experiments.

---

# 26. Topic Diversity

The pilot should cover:

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

Do not let the corpus become almost entirely placement reviews.

---

# 27. Annotation Workflow

Each pilot record should follow:

```text
record creation/source selection
        ↓
anonymous review_id assignment
        ↓
language tagging
        ↓
topic tagging
        ↓
independent risk annotation
        ↓
disagreement review
        ↓
adjudication where needed
        ↓
final pilot record
```

---

# 28. Double-Annotation Subset

At least:

```text
60 of the 150 records
```

should be independently annotated by two annotators if practical.

This subset should deliberately include:

- English
- Hinglish
- Roman Hindi
- clean examples
- risky examples
- multi-label examples
- borderline examples

The purpose is to measure annotation consistency.

---

# 29. Remaining Pilot Records

The remaining records may initially use:

```text
one primary annotation
+
later quality review
```

if annotation resources are limited.

The pilot should not pretend that single-annotator labels have the same
reliability as adjudicated multi-annotator labels.

---

# 30. Annotator Independence

For the double-annotated subset:

Annotator A and Annotator B should make their first decisions independently.

They should not see each other's labels before initial annotation.

This reduces conformity bias.

---

# 31. Agreement Analysis

After annotation, measure disagreement by label.

At minimum inspect:

```text
spam disagreement
deception disagreement
toxicity disagreement
advertising disagreement
off-topic disagreement
pii disagreement
```

Also inspect disagreement by:

```text
language_mix
college_category
```

---

# 32. Agreement Metric

For the pilot, begin with simple raw agreement percentages per label.

If the pilot demonstrates sufficient volume and annotation quality,
additional statistics such as:

```text
Cohen's kappa
```

may be calculated for two-annotator binary labels.

Do not blindly report agreement metrics without understanding their
assumptions.

---

# 33. High-Disagreement Rule

If a label shows substantial disagreement:

```text
do not scale annotation immediately
```

Instead:

```text
inspect disagreement
        ↓
identify ambiguous rule
        ↓
improve ANNOTATION_GUIDE.md
        ↓
re-test
```

The pilot exists specifically to find these problems.

---

# 34. Annotation Status

Pilot samples may progress through:

```text
UNANNOTATED
IN_PROGRESS
ANNOTATED
ADJUDICATION_REQUIRED
FINAL
EXCLUDED
```

Only appropriate final records should later enter model experiments.

---

# 35. Pilot Exclusion Reasons

Possible reasons include:

- empty text
- corrupted text
- unusable provenance
- privacy issue
- duplicate sample
- incomprehensible language
- annotation policy cannot resolve case

Exclusion must be documented.

---

# 36. Duplicate Control

Before pilot finalization, check:

- exact duplicate text
- whitespace-normalized duplicates
- case-normalized duplicates
- obvious template duplicates

Do not treat copied variants as independent evidence.

---

# 37. Derived-Variant Grouping

If multiple perturbations originate from one parent review:

```text
parent review
  ├── spelling variant
  ├── repeated-character variant
  └── Hinglish variant
```

record that relationship internally.

These variants must later remain in the same split.

---

# 38. Dataset Split Policy for Pilot

The first 150-record pilot does NOT need to become the final
train/validation/test split.

Its primary purpose is:

```text
annotation validation
```

not:

```text
final model benchmarking
```

Therefore do not prematurely freeze a final test set from this tiny pilot.

---

# 39. Pilot Model Usage

The pilot may later support:

- schema validation scripts
- preprocessing tests
- tokenizer inspection
- tiny development experiments

It must not be used to claim final RRM performance.

---

# 40. No Performance Optimization During Pilot

Do not change labels or sample composition because a model performs poorly.

The correct process is:

```text
dataset problem?
        ↓
fix dataset methodology

model problem?
        ↓
fix model later
```

Do not modify ground truth to fit model predictions.

---

# 41. Pilot Quality Report

After the pilot is complete, produce a factual report containing:

- total record count
- language distribution
- topic distribution
- label distribution
- multi-label frequency
- deception UNKNOWN frequency
- annotation-status distribution
- annotator agreement
- exclusion count
- duplicate count
- major ambiguity patterns

No model-performance claims are required at this stage.

---

# 42. Pilot Success Criteria

The pilot is successful if:

- schema is practical
- annotation rules are understandable
- English/Hinglish/Roman-Hindi records can be represented
- annotators can distinguish normal criticism from toxicity
- advertising/spam boundaries are workable
- deception UNKNOWN is used correctly
- PII policy is workable
- multi-label annotation is practical
- disagreements can be explained
- provenance remains intact
- duplicate handling works

---

# 43. Pilot Failure Is Useful

If the pilot reveals:

- high disagreement
- unclear labels
- unusable fields
- poor language categories
- confusing source metadata

that is not a failed research project.

That is the reason the pilot exists.

The methodology should be corrected before scaling.

---

# 44. Explicit Non-Goals

The pilot does not determine:

- final dataset size
- final class distribution
- final model performance
- final tokenizer vocabulary
- final Transformer configuration
- final train/test percentages
- production moderation thresholds
- Layer-4 Trust decisions

---

# 45. Pilot Design Completion Criteria

RRM 3.2D is complete when:

- pilot size is defined
- language composition is defined
- scenario composition is defined
- source composition rules are defined
- deception handling is defined
- multi-label coverage is required
- borderline cases are required
- annotation workflow is defined
- double-annotation subset is defined
- disagreement handling is defined
- duplicate handling is defined
- pilot split policy is defined
- quality-report requirements are defined

After this design is locked, proceed to:

**RRM 3.2E — Pilot Dataset Build**

---

## Pilot Outcome — RMC v0.1

The initial dataset-foundation pilot is complete.

### Seed Corpus

- 150 synthetic RMC seed reviews
- English, Hinglish, and Roman Hindi coverage
- Normal, boundary, multi-label, and noisy-language examples
- Synthetic provenance explicitly recorded
- Seed corpus is development material, not final training or evaluation evidence

### Human Annotation Pilot

- Annotator A1: 40 reviews
- Annotator A2: 20 reviews
- Shared double-annotated reviews: 20
- Duplicate annotator-review pairs: 0
- Blind deception annotations remained UNKNOWN

### Round-1 Agreement

| Field | Raw Agreement | Cohen's Kappa |
| --- | ---: | ---: |
| Spam | 55% | 0.100 |
| Toxicity | 50% | 0.029 |
| Advertising | 70% | 0.400 |
| Off-topic | 45% | -0.058 |
| PII | 45% | -0.100 |
| Language Mix | 30% | 0.122 |
| College Category | 25% | 0.178 |

The results are treated as pilot diagnostics rather than final reliability estimates.

Round-1 disagreement demonstrated that several operational definitions were too ambiguous, particularly off-topic, PII, language classification, and college-category assignment.

### Guide Revision

Annotation Guide v0.2 was introduced in response to the Round-1 findings.

The revision clarifies:

- spam versus advertising
- criticism versus toxicity
- student-life relevance versus off-topic content
- personal PII versus institutional contact information
- English versus Hinglish versus Roman Hindi
- single-topic versus multi-topic college categorization
- independent evaluation of each risk label

### Research Interpretation

The synthetic seed corpus is retained for:

- schema validation
- annotation workflow validation
- deterministic-rule development
- robustness examples
- test fixtures
- later tokenizer and character-level experiments

It must not be represented as:

- a naturally collected student-review corpus
- a final human-annotated dataset
- model-performance evidence
- proof of real-world prevalence
- real-world deception ground truth

The Round-1 annotations and disagreements are preserved rather than overwritten.

Future annotation reliability should be re-evaluated on approved real or independently collected research data using the revised annotation guide.
