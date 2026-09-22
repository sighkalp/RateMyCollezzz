# RateMyCollezzz RRM — Annotation Guide

## 1. Purpose

This document defines how human annotators should assign RMC review-risk labels.

It operationalizes the six primary labels:

- spam
- deception
- toxicity
- advertising
- off_topic
- pii

This guide is intended to reduce subjective interpretation and improve annotation consistency.

It must be used together with:

- `RESEARCH_CONTRACT.md`
- `DATASET_SPEC.md`

---

# 2. Core Annotation Principles

Annotators must follow these rules:

1. Annotate only what is supported by the text and available ground truth.
2. Do not guess intent without evidence.
3. Multiple labels may be positive at the same time.
4. Negative criticism is not automatically toxic.
5. Advertising is not automatically spam.
6. Suspicious writing is not automatically deceptive.
7. Unknown deception must remain UNKNOWN.
8. Personal opinions are not factual proof.
9. Borderline cases should be flagged rather than forced.
10. Annotation decisions must be reproducible.

---

# 3. Label Encoding

For most labels:

```text
0 = absent
1 = present
```

For deception:

```text
-1 = UNKNOWN / insufficient ground truth
 0 = defensibly non-deceptive
 1 = defensibly deceptive
```

---

# RMC Annotation Guide v0.2 Clarifications

**Current guide version:** RMC-v0.2

RMC-v0.2 was introduced after Round-1 pilot agreement analysis showed substantial disagreement in several labels and metadata fields.

## Spam
Set `spam = 1` only for clear repetitive, flooding, templated, or manipulative posting behaviour.

Do not mark something as spam merely because it is negative, toxic, off-topic, advertising, short, or highly positive.

Advertising and spam are independent labels.

## Toxicity
Set `toxicity = 1` for abusive, insulting, threatening, hateful, or degrading language toward a person or group.

Normal criticism is not toxicity.

Examples of non-toxic criticism:
- "The professor explains concepts poorly."
- "Administration ka process bahut slow hai."
- "Placement support was disappointing."

## Advertising
Set `advertising = 1` when the text promotes a commercial service, consultancy, coaching service, referral, product, paid channel, or similar promotion.

Normal references to official college resources are not advertising.

## Off-topic
Set `off_topic = 1` only when the content is materially unrelated to college or student experience.

College surroundings, metro access, transport, food options, hostel surroundings, and similar student-life information remain relevant.

## PII
Set `pii = 1` when an actual personal identifier appears, such as:
- personal phone number
- personal email
- private home address
- government identifier
- private account identifier

Official institutional contact information is not treated as personal PII in the current pilot.

## Language

### ENGLISH
Predominantly English.

### HINGLISH
English and Hindi both make meaningful contributions to the sentence.

### ROMAN_HINDI
Predominantly Hindi written in the Latin/Roman alphabet.

Common words such as college, faculty, hostel, and placement do not automatically make Roman Hindi into Hinglish.

### OTHER
Predominantly a language other than English or Hindi.

### MIXED_OTHER
Meaningful mixing involving another language outside English/Hindi.

Do not use OTHER or MIXED_OTHER because of spelling mistakes, abbreviations, or informal writing.

## College Category
Choose one category when one topic clearly dominates.

Use `MULTI_TOPIC` only when two or more substantial college topics are genuinely discussed.

Use `OTHER` when no legitimate college topic applies.

## Independent-label rule

Each field must be judged independently.

Never assume:

- off-topic means spam
- advertising means spam
- negative means toxic
- positive means genuine
- mention of phone-related words means PII

Every positive label requires its own evidence.

