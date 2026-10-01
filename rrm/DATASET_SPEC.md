# RateMyCollezzz RRM — Dataset Specification

## 1. Purpose

This document defines the structure, semantics, and research rules for the
RateMyCollezzz Review Model Corpus (RMC).

The dataset is intended to support research on review-risk detection for
college reviews containing:

- English
- Hinglish
- Roman Hindi
- Indian English
- informal student language
- spelling variation
- college-domain terminology

The production annotation and metadata contract is defined in:

```text
rrm/PRODUCTION_ANNOTATION_CONTRACT.md
```

That document is the canonical Gate B frozen contract.

This specification defines what each dataset field means.

It does not define the detailed human annotation procedure.

That belongs in `ANNOTATION_GUIDE.md`.

---

## 2. Dataset Principles

The RMC dataset must follow these principles:

1. Dataset provenance must be preserved.
2. Model inputs must be separated from research metadata.
3. Labels must not be guessed when reliable evidence is unavailable.
4. Deception requires defensible ground truth.
5. Multiple risk labels may apply to one review.
6. Train, validation, and test leakage must be prevented.
7. Personal or sensitive information must be handled carefully.
8. Dataset versions and splits must be reproducible.
9. Synthetic data must remain identifiable as synthetic.
10. Metadata must not silently become a model feature.

---

## 3. Unit of Analysis

The primary unit of analysis is one review text.

Conceptually:

```text
One row
   =
One review
   +
Research metadata
   +
Risk labels
   +
Annotation metadata
```

---

## 4. Six Task Labels

Canonical order:

```text
("spam", "deception", "toxicity", "advertising", "off_topic", "pii")
```

Allowed values:

| Label | Allowed values |
|-------|---------------|
| `spam` | 0, 1 |
| `deception` | -1, 0, 1 |
| `toxicity` | 0, 1 |
| `advertising` | 0, 1 |
| `off_topic` | 0, 1 |
| `pii` | 0, 1 |

Only `deception` supports -1.

-1 means defensible truth status is not known.

UNKNOWN is not negative.

UNKNOWN must not become 0.

UNKNOWN is masked during supervised training.

All six labels are supervised training targets.

They are NOT neural inputs.

The sole neural input is review_text.

---

## 5. Text Fields

### 5.1 source_text_raw

Requirement: CONDITIONAL

Purpose: secure/restricted source retention where governance permits

Rules:

- never public
- never neural input
- applicable to Human-Written RMC only where consent permits

### 5.2 review_text

Requirement: REQUIRED

Purpose: canonical safe text for annotation, training, and runtime

Rules:

- sole neural input
- must contain no real private value
- preserve natural language, spelling, casing, Hinglish/Roman Hindi
- no grammar or semantic rewriting

### 5.3 export_text_redacted

Requirement: OPTIONAL

Purpose: public/export-safe text

Rules:

- never neural input
- may be more aggressively redacted than review_text

---

## 6. PII Policy

Real private PII must never be publicly redistributed.

PII evidence metadata must never store the original private value.

If real PII appears in source_text_raw and the record is retained for
training, transform it to a SAFE TYPE-PRESERVING NON-REAL SURROGATE.

pii = 1 only when review_text still contains a safe PII-like instance
consistent with the target.

If safe transformation cannot preserve target validity, the record must
not be used as a positive PII supervised example.

Controlled/safe research examples are the preferred source of PII-positive
training examples.

PII evidence metadata fields (pii_detected, pii_categories,
pii_redaction_status, pii_evidence_id) contain type/category/status
information only.

They never contain the original private value.

---

## 7. Language Metadata

language_mix:

- REQUIRED
- single-valued

Allowed values:

```text
ENGLISH
HINGLISH
ROMAN_HINDI
OTHER
MIXED_OTHER
```

Common English college-domain words (college, faculty, hostel, placement)
do not automatically make Roman Hindi into Hinglish.

Do not use OTHER or MIXED_OTHER because of spelling mistakes,
abbreviations, or informal writing.

---

## 8. College-Topic Metadata

college_category:

- OPTIONAL
- nullable
- single-valued

Allowed non-null values:

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

college_category = null when no legitimate college topic applies,
including when a review is fully off-topic.

off_topic is a six-task risk label.

It is NOT a college_category value.

MULTI_TOPIC handles reviews with multiple substantial college topics.

---

## 9. Annotation Workflow

Production policy: 100% independent double annotation.

Minimal status lifecycle:

```text
UNANNOTATED -> ANNOTATING -> ADJUDICATION_REQUIRED -> FINAL
                                                    -> EXCLUDED
```

| Status | Meaning |
|--------|---------|
| `UNANNOTATED` | Record created, annotation not started |
| `ANNOTATING` | Annotation in progress |
| `ADJUDICATION_REQUIRED` | Disagreement detected, awaiting adjudication |
| `FINAL` | Adjudication complete, record finalized |
| `EXCLUDED` | Record excluded |

Only FINAL records enter model experiments.

---

## 10. Adjudication

Any risk-label disagreement: route to adjudication.

Adjudicator: trained senior annotation lead who was not Annotator A or B.

Guide ambiguity: flag for revision, re-annotate affected records.

Deception: provenance overrides voting. If truth unavailable: deception = -1.

No numeric kappa acceptance threshold.

Agreement metrics may be reported descriptively.

---

## 11. Source Types

Production source_type values:

```text
HUMAN_WRITTEN_RMC
CONTROLLED_RMC
SYNTHETIC_DERIVED_RMC
```

---

## 12. Source-Specific Fields

### 12.1 Human-Written RMC

- consent_status (CONDITIONAL; CONSENTED for includable records)
- contributor_pseudonym (CONDITIONAL)
- collection_method (CONDITIONAL)

Requires explicit digital research/data-use consent before collection.

No paper signature required by default.

No real name required.

### 12.2 Controlled RMC

- experiment_id (CONDITIONAL)
- controlled_targets (CONDITIONAL)
- deception_truth (CONDITIONAL)
- control_protocol_id (CONDITIONAL)

Controlled truth comes from the experiment protocol, not annotator voting.

### 12.3 Synthetic / Derived RMC

- parent_review_id (CONDITIONAL)
- derivation_type (CONDITIONAL)
- generation_method (CONDITIONAL)

Same split group as parent under Gate E.

No fixed production proportion.

---

## 13. Audit Fields

| Field | Requirement | Type / Allowed Values | Neural Input | Training Target | Owning Gate | Purpose |
|-------|------------|----------------------|-------------|-----------------|-------------|---------|
| created_at | REQUIRED | datetime (ISO 8601) | NO | NO | C | Record creation timestamp |
| finalized_at | CONDITIONAL | datetime (ISO 8601); present when annotation_status = FINAL | NO | NO | C | Finalization timestamp |

---

## 14. Later-Gate Fields

These fields are POPULATED_LATER and owned by later gates.

They are NOT REQUIRED during annotation.

| Field | Requirement | Type / Allowed Values | Neural Input | Training Target | Owning Gate | Purpose |
|-------|------------|----------------------|-------------|-----------------|-------------|---------|
| dataset_version | POPULATED_LATER | str | NO | NO | D | Frozen dataset version identifier |
| split_membership | POPULATED_LATER | str | NO | NO | E | Train / validation / test assignment |
| split_group_id | POPULATED_LATER | str | NO | NO | E | Leakage-safe grouping identifier |

split_seed is Gate E manifest metadata, not a per-record field.

Tokenizer-related fields belong to Gate F / Gate G.

---

## 15. Multi-Label Rules

One review may contain more than one risk type.

Labels are independent binary labels, not mutually exclusive.

Each field must be judged independently.

Every positive label requires its own evidence.

---

## 16. Neural Input Isolation

The only normal neural input is review_text.

All six task labels, all metadata fields, all provenance fields,
all annotation metadata, and all audit fields are NOT neural inputs.

They are NOT training targets (except the six labels).

Research metadata must not silently become a model feature.

---

## 17. Evidence vs Decision

This specification defines evidence production.

It does NOT define moderation decisions.

The RRM produces risk signals and evidence.

Layer 4 (Trust) makes final moderation decisions.

Similarity and duplicate evidence are separate from the six neural heads.
