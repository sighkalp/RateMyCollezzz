# RMC Production Annotation + Metadata Contract

## Document Header

Title: RMC Production Annotation + Metadata Contract

Contract version: 1.0

Gate: Gate B — Production Annotation + Metadata Contract

Status: FROZEN

Basis: Gate A CLOSED; Gate B Phase 1 approved

---

## 1. Purpose

This document defines the production annotation and metadata contract for the
RateMyCollezzz Review Model Corpus (RMC).

It is the canonical reference for:

- the six task labels and their allowed values
- the neural input contract
- the text/PII field design
- the language metadata policy
- the college-topic metadata policy
- the annotation workflow
- the double-annotation requirement
- the adjudication policy
- the source-specific rules for Human-Written, Controlled, and Synthetic/Derived RMC
- the annotation metadata fields
- the provenance fields
- the audit fields

This document does NOT:

- collect production data
- download datasets
- train any model
- implement Gate C data collection
- define train/validation/test splits
- freeze dataset versions
- train the production tokenizer
- make moderation decisions
- implement Trust-layer policy

Those responsibilities belong to later gates.

---

## 2. Six Task Labels

Canonical order:

```text
("spam", "deception", "toxicity", "advertising", "off_topic", "pii")
```

Allowed values:

| Label | Allowed values | Notes |
|-------|---------------|-------|
| `spam` | 0, 1 | Binary |
| `deception` | -1, 0, 1 | -1 = UNKNOWN |
| `toxicity` | 0, 1 | Binary |
| `advertising` | 0, 1 | Binary |
| `off_topic` | 0, 1 | Binary |
| `pii` | 0, 1 | Binary |

Only `deception` supports -1.

-1 means:

Defensible truth status is not known.

UNKNOWN is not negative.

UNKNOWN must not become 0.

UNKNOWN is masked during supervised training.

All six labels are:

Neural Input = NO

Training Target = YES

---

## 3. Neural Input Contract

The canonical neural input is:

review_text

review_text is:

- REQUIRED
- string
- Neural Input = YES
- Training Target = NO
- preserve natural language
- preserve spelling and casing
- preserve English / Hinglish / Roman Hindi / Indian English variation
- no grammar rewriting
- no semantic rewriting

Research metadata must not silently enter the neural model.

Fields that are explicitly NOT neural inputs include:

- all six task labels
- language_mix
- college_category
- source_type and all provenance fields
- all annotation metadata
- all audit/version fields
- all PII evidence fields (type/category/status only, never raw values)
- source_text_raw
- export_text_redacted

---

## 4. Text Fields and PII Contract

### 4.1 source_text_raw

Requirement: CONDITIONAL

Purpose: secure/restricted source retention where governance permits

Rules:

- never public
- never neural input
- may contain original source text only under valid governance
- applicable only to Human-Written RMC where consent permits

### 4.2 review_text

Requirement: REQUIRED

Purpose: canonical safe text used for annotation, training, and runtime

Rules:

- sole neural input
- must contain no real private value
- if real PII appears in source_text_raw and the record is retained for
  training, transform it to a SAFE TYPE-PRESERVING NON-REAL SURROGATE
  where feasible

Safe transformation examples (conceptual):

- real email -> non-real test email
- real phone -> non-real phone-like test value
- real identifier -> non-real format-preserving research surrogate

The actual private value must never be preserved in review_text.

### 4.3 export_text_redacted

Requirement: OPTIONAL

Purpose: public/export-safe text

Rules:

- never neural input
- may be more aggressively redacted than review_text

### 4.4 PII Evidence Metadata

PII evidence metadata may contain:

- pii_detected (boolean)
- pii_categories (list of PII type/category labels)
- pii_redaction_status (redaction/transformation status)
- pii_evidence_id (rule or evidence identifier)

PII evidence metadata must NOT contain the original private value.

### 4.5 PII Target/Input Consistency

- pii = 1 is valid only when review_text still contains a safe PII-like
  instance consistent with the target.
- If safe transformation cannot preserve target validity, the record must
  not be used as a positive PII supervised example.
- Controlled/safe research examples are the preferred source of PII-positive
  training examples.

---

## 5. Language Metadata

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

No UNSPECIFIED value.

Workflow status handles records not yet annotated.

LOCKED rule:

Common English college-domain words (college, faculty, hostel, placement)
do not automatically make Roman Hindi into Hinglish.

Do not use OTHER or MIXED_OTHER because of spelling mistakes,
abbreviations, or informal writing.

---

## 6. College-Topic Metadata

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

Do NOT use:

- OTHER
- OFF_TOPIC
- UNSPECIFIED
- NOT_APPLICABLE

off_topic is the six-task risk label.

It is NOT a college_category value.

MULTI_TOPIC handles reviews with multiple substantial college topics.

---

## 7. Annotation Workflow

Production policy: 100% independent double annotation.

Annotator B must not see Annotator A's initial labels before submitting.

Minimal lifecycle:

```
UNANNOTATED -> ANNOTATING -> ADJUDICATION_REQUIRED -> FINAL
                                                    -> EXCLUDED
```

| Status | Meaning |
|--------|---------|
| `UNANNOTATED` | Record created, annotation not started |
| `ANNOTATING` | Annotation in progress |
| `ADJUDICATION_REQUIRED` | Disagreement detected, awaiting adjudication |
| `FINAL` | Adjudication complete, record finalized |
| `EXCLUDED` | Record excluded (empty, duplicate, corrupted, etc.) |

No READY, ANNOTATED_A, ANNOTATED_B, or FROZEN states.

FROZEN belongs to Gate D.

---

## 8. Adjudication Policy

Any risk-label disagreement: route to adjudication.

Adjudicator:

- trained senior annotation lead
- must not be Annotator A or B for that record

Guide ambiguity:

- flag for guide revision
- affected records require re-annotation after revision

Deception:

- provenance rule overrides subjective voting
- if defensible truth unavailable: deception = -1

No numeric kappa acceptance threshold in Gate B.

Agreement metrics may be reported descriptively.

---

## 9. Source-Specific Rules

### 9.1 Human-Written RMC

source_type: HUMAN_WRITTEN_RMC

Role: primary planned in-domain corpus

Rules:

- explicit digital research/data-use consent before collection
- exact UI/form implementation belongs to Gate C
- no paper signature required by default
- institutional rules may impose additional requirements later
- use pseudonymous contributor identifier
- no real name required in dataset metadata
- preserve natural English / Hinglish / Roman Hindi / Indian English
- no grammar/style normalization
- deception defaults to -1 absent defensible truth provenance
- source_text_raw may be retained only where consent/governance permits

Fields where applicable:

- consent_status (CONDITIONAL; CONSENTED for includable records)
- contributor_pseudonym (CONDITIONAL)
- collection_method (CONDITIONAL)

### 9.2 Controlled RMC

source_type: CONTROLLED_RMC

Role: controlled-ground-truth supplement

Fields:

- experiment_id (CONDITIONAL)
- controlled_targets (CONDITIONAL)
- deception_truth (CONDITIONAL)
- control_protocol_id (CONDITIONAL)

Rules:

- controlled truth comes from experimental protocol, not annotator voting
- only labels explicitly established by the experiment are authoritative
- annotators may still annotate other perceptual labels
- deception = 0/1 only when experiment establishes truthful/deceptive status
- otherwise deception = -1

### 9.3 Synthetic / Derived RMC

source_type: SYNTHETIC_DERIVED_RMC

Role: conditional augmentation / robustness data

Fields:

- parent_review_id (CONDITIONAL)
- derivation_type (CONDITIONAL)
- generation_method (CONDITIONAL)

Rules:

- always identifiable as synthetic/derived
- preserve parent provenance
- labels inherited only when defensible
- synthetic generation alone does not establish deception
- same split group as parent later under Gate E
- never substitute for real-domain final evaluation
- must not dominate merely to inflate metrics
- NO fixed percentage at Gate B

---

## 10. Source Types

Production source_type values:

```text
HUMAN_WRITTEN_RMC
CONTROLLED_RMC
SYNTHETIC_DERIVED_RMC
```

External Gate A HOLD sources do NOT enter the production enum yet.

---

## 11. Annotation Metadata

| Field | Requirement | Type / Allowed Values | Neural Input | Training Target | Owning Gate | Purpose |
|-------|------------|----------------------|-------------|-----------------|-------------|---------|
| annotation_status | REQUIRED | str: UNANNOTATED, ANNOTATING, ADJUDICATION_REQUIRED, FINAL, EXCLUDED | NO | NO | B contract / C annotation | Workflow state |
| annotation_guide_version | REQUIRED | str (semantic version, e.g. "1.0") | NO | NO | B contract / C annotation | Guide version used for final annotation |
| annotator_A_id | REQUIRED | str (pseudonymous) | NO | NO | B contract / C annotation | First annotator identifier |
| annotator_B_id | REQUIRED | str (pseudonymous) | NO | NO | B contract / C annotation | Second annotator identifier |
| adjudicator_id | CONDITIONAL | str (pseudonymous); present when adjudication occurred | NO | NO | B contract / C annotation | Adjudicator identifier |
| annotation_notes | OPTIONAL | str (free text) | NO | NO | B contract / C annotation | Annotator or adjudicator notes |

annotation_guide_version production value: "1.0"

---

## 12. Provenance and Source Fields

| Field | Requirement | Type / Allowed Values | Neural Input | Training Target | Owning Gate | Purpose |
|-------|------------|----------------------|-------------|-----------------|-------------|---------|
| source_type | REQUIRED | str | NO | NO | B contract / C collection | Source class identifier |
| consent_status | CONDITIONAL | str; HUMAN_WRITTEN_RMC only; CONSENTED for includable records | NO | NO | B contract / C collection | Consent state |
| contributor_pseudonym | CONDITIONAL | str (pseudonymous); HUMAN_WRITTEN_RMC only | NO | NO | B contract / C collection | Pseudonymous contributor identifier |
| collection_method | CONDITIONAL | str; HUMAN_WRITTEN_RMC only | NO | NO | B contract / C collection | How the review was collected |
| experiment_id | CONDITIONAL | str; CONTROLLED_RMC only | NO | NO | B contract / C collection | Controlled experiment identifier |
| controlled_targets | CONDITIONAL | str or list; CONTROLLED_RMC only | NO | NO | B contract / C collection | Labels established by experiment protocol |
| deception_truth | CONDITIONAL | str; CONTROLLED_RMC only | NO | NO | B contract / C collection | How deception truth was established |
| control_protocol_id | CONDITIONAL | str; CONTROLLED_RMC only | NO | NO | B contract / C collection | Protocol identifier |
| parent_review_id | CONDITIONAL | str; SYNTHETIC_DERIVED_RMC only | NO | NO | B contract / C collection | Parent review identifier |
| derivation_type | CONDITIONAL | str; SYNTHETIC_DERIVED_RMC only | NO | NO | B contract / C collection | How the derivative was created |
| generation_method | CONDITIONAL | str; SYNTHETIC_DERIVED_RMC only | NO | NO | B contract / C collection | Generation method |

---

## 13. Audit Fields

| Field | Requirement | Type / Allowed Values | Neural Input | Training Target | Owning Gate | Purpose |
|-------|------------|----------------------|-------------|-----------------|-------------|---------|
| created_at | REQUIRED | datetime (ISO 8601) | NO | NO | C | Record creation timestamp |
| finalized_at | CONDITIONAL | datetime (ISO 8601); present when annotation_status = FINAL | NO | NO | C | Finalization timestamp |

---

## 14. Later-Gate Fields

The following fields are part of the future-compatible schema but are
POPULATED_LATER and owned by later gates.

They must NOT be treated as REQUIRED during annotation.

| Field | Requirement | Type / Allowed Values | Neural Input | Training Target | Owning Gate | Purpose |
|-------|------------|----------------------|-------------|-----------------|-------------|---------|
| dataset_version | POPULATED_LATER | str | NO | NO | D | Frozen dataset version identifier |
| split_membership | POPULATED_LATER | str | NO | NO | E | Train / validation / test assignment |
| split_group_id | POPULATED_LATER | str | NO | NO | E | Leakage-safe grouping identifier |

split_seed is Gate E manifest metadata.

It is NOT a per-record field.

Tokenizer-related fields belong to Gate F / Gate G.

They are NOT part of the Gate B annotation schema.

---

## 15. Multi-Label Rules

- One review may contain more than one risk type.
- Labels are independent binary labels, not mutually exclusive.
- Each field must be judged independently.
- Never assume off_topic implies spam, advertising implies spam,
  negative implies toxic, positive implies genuine,
  or mention of phone-related words implies PII.
- Every positive label requires its own evidence.

---

## 16. Boundary Examples and Ambiguity Rules

### 16.1 Spam vs Advertising

- Spam: repetitive, flooding, templated, low-value, manipulative posting.
- Advertising: content primarily attempting to promote a commercial service.
- A review may be both spam and advertising.
- Do not mark something as spam merely because it is negative, toxic,
  off-topic, advertising, short, or highly positive.

### 16.2 Negative Criticism vs Toxicity

- "The hostel food is terrible." is negative criticism, not toxicity.
- Toxicity requires abusive, insulting, threatening, hateful, or degrading
  language toward a person or group.

### 16.3 Off-topic Boundary

- College surroundings, transport, food options, and student-life information
  remain relevant to student experience.
- Only mark off_topic when content is materially unrelated to college or
  student experience.

### 16.4 Deception Default

- Ordinary human-written college reviews: deception = -1
- Deception = 0/1 only when defensible ground truth exists.
- Do not label a review deceptive because it appears exaggerated, unusually
  positive, unusually negative, polished, machine-generated, or suspicious.

### 16.5 PII Boundary

- Official institutional contact information is not treated as personal PII.
- Only actual personal identifiers count: personal phone, personal email,
  home address, government identifier, private account identifier.

### 16.6 Ambiguity Handling

Borderline cases should be flagged rather than forced.

If guide ambiguity causes disagreement:

- mark the guide section for revision
- re-annotate affected records after revision

---

## 17. Research Metadata Isolation

The following are research and governance metadata.

They are NOT neural model inputs.

They are NOT training targets.

They exist for audit, provenance, research analysis, and dataset governance:

- source_type
- consent_status
- contributor_pseudonym
- collection_method
- experiment_id
- controlled_targets
- deception_truth
- control_protocol_id
- parent_review_id
- derivation_type
- generation_method
- language_mix
- college_category
- annotation_status
- annotation_guide_version
- annotator_A_id
- annotator_B_id
- adjudicator_id
- annotation_notes
- created_at
- finalized_at
- pii_categories
- pii_redaction_status
- pii_evidence_id
- dataset_version (POPULATED_LATER)
- split_membership (POPULATED_LATER)
- split_group_id (POPULATED_LATER)

---

## 18. Evidence vs Decision Distinction

This contract defines evidence production.

It does NOT define moderation decisions.

The RRM produces risk signals and evidence.

Layer 4 (Trust) makes final moderation decisions.

Similarity and duplicate evidence are separate from the six neural heads.

They are NOT included in the six-task macro score.

---

## 19. Contract Versioning

This contract is version 1.0.

Future versions require explicit approval.

Changes to:

- label definitions
- allowed label values
- neural input contract
- PII target/input policy
- source-type enum

require explicit re-approval.

Changes to:

- annotation guide details
- adjudication procedure refinements
- metadata field additions that do not alter existing contracts

may be version-bumped without reopening the core contract.
