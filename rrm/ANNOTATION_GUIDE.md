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
