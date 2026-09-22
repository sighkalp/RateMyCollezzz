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
