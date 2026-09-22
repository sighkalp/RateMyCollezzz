# RateMyCollezzz RRM — Research Contract

## 1. Purpose

The RateMyCollezzz Review Risk Model (RRM) is a specialized review-intelligence
system for analyzing student-written college reviews.

The RRM is designed for language commonly found in Indian college reviews,
including:

- English
- Hinglish
- Roman Hindi
- Indian English
- informal student language
- spelling variation
- repeated characters
- college-domain terminology

The RRM produces review-level evidence and risk signals.

It does NOT make final moderation decisions.

Core architectural rule:

**RRM UNDERSTANDS.**

**TRUST DECIDES.**

---

# 2. Research Problem

Online college-review platforms may contain content that is:

- spam
- deceptive
- abusive
- promotional
- irrelevant
- privacy-sensitive
- duplicated or coordinated

Generic text classifiers may not fully capture:

- Hinglish
- Roman Hindi
- Indian-English phrasing
- informal spelling
- repeated characters
- obfuscated words
- college-specific terminology

The research problem is therefore:

> Can a compact domain-aware multi-task model combining subword-level
> semantic representations and character-level representations provide
> useful review-risk detection for English/Hinglish college reviews while
> remaining smaller and more efficient than large pretrained baselines?

This is a research question.

It is NOT a claim that the custom RRM is already superior.

---

# 3. Core RRM Scope

The RRM subsystem may contain:

1. input validation
2. deterministic review checks
3. exact duplicate detection
4. near-duplicate / similarity signals
5. custom tokenization
6. semantic Transformer representation
7. Character CNN representation
8. feature fusion
9. multi-task risk heads
10. review-risk inference

The RRM does not own:

- user bans
- account reputation
- campaign detection
- coordinated-user graph analysis
- final moderation
- publish/remove decisions

Those belong primarily to Layer 4 — Trust & Safety.

---

# 4. Primary Label Taxonomy

The RRM uses a multi-label formulation.

One review may contain more than one risk type.

Example:

A review may simultaneously be:

- advertising
- spam
- off-topic

Therefore the primary labels are independent binary labels rather than one
single mutually exclusive class.

---

## 4.1 Spam

### Definition

Content whose primary behaviour resembles unwanted, repetitive, low-value,
or manipulative posting rather than a genuine useful college review.

Possible examples include:

- repeated promotional messages
- copied review templates
- meaningless repeated text
- mass-posted content
- review flooding
- obvious unrelated spam

### Important distinction

Spam does not automatically mean deception.

A review may be spam without making a false factual claim.

---

## 4.2 Deception

### Definition

A review deliberately presented as genuine experience when reliable ground
truth indicates that the claimed experience is fabricated or intentionally
misrepresented.

### Critical research restriction

Deception is difficult to infer from text alone.

The project must NOT label ordinary reviews as deceptive merely because they:

- sound overly positive
- sound overly negative
- use unusual vocabulary
- appear AI-like
- contain strong opinions

A deception label requires defensible ground truth.

Acceptable research sources may include:

- established deception datasets
- intentionally generated experimental samples
- controlled annotation scenarios with known provenance

Unverified real student reviews must not be casually labeled deceptive.

---

## 4.3 Toxicity

### Definition

Content containing abusive, insulting, threatening, hateful, or seriously
hostile language directed at a person or group.

Toxicity is about harmful language.

It is not the same as criticism.

Example:

> "The hostel food is terrible."

is negative criticism but is not automatically toxic.

---

## 4.4 Advertising

### Definition

Content primarily attempting to promote:

- a commercial service
- coaching
- admissions consultancy
- paid product
- website
- social channel
- unrelated business
- referral or promotional offer

Advertising may overlap with spam.

The two labels remain separate so the model can distinguish promotional intent
from broader spam behaviour.

---

## 4.5 Off-topic

### Definition

Content that does not materially relate to:

- the college
- student experience
- academics
- faculty
- placements
- infrastructure
- hostel
- campus life
- fees
- administration
- other legitimate college-review topics

A review can be well-written but still be off-topic.

---

## 4.6 PII

### Definition

Content containing personally identifiable or sensitive information that
should not normally appear publicly in a college review.

Possible examples include:

- personal phone numbers
- personal email addresses
- home addresses
- government identifiers
- private account identifiers

PII detection may combine:

- deterministic rules
- pattern recognition
- learned contextual signals

Rule-derived and model-derived PII evidence must remain distinguishable.

---

# 5. Secondary Signals

These are useful RRM outputs but are NOT necessarily neural classification
heads.

---

## 5.1 Exact Duplicate

Detect reviews whose normalized text is identical.

This should primarily use deterministic comparison or hashing.

It does not require a neural network.

---

## 5.2 Near-Duplicate / Similarity

Estimate whether a review is unusually similar to another review.

Possible methods may include:

- character similarity
- TF-IDF similarity
- embedding similarity

The initial implementation should remain simple.

Collective coordination analysis belongs to Layer 4.

---

## 5.3 Review Quality

Review quality may include signals such as:

- informativeness
- sufficient context
- meaningful content
- excessive repetition

However, "quality" is subjective.

Therefore review quality must NOT become a supervised neural target until it
has a precise annotation definition and measurable ground truth.

---

## 5.4 AI-like Writing

AI-like writing detection is optional and experimental.

It must NOT be treated as proof that a review was generated by AI.

It must NOT independently cause:

- removal
- banning
- moderation punishment

If studied, it is only an additional weak signal.

---

# 6. Rule Signals vs Learned Signals

The RRM is a hybrid system.

Not every problem should be solved with neural ML.

---

## Deterministic / Rule-Oriented Signals

Examples:

- empty input
- malformed input
- exact duplicates
- obvious URLs
- obvious phone-number patterns
- obvious email patterns
- repeated-character limits
- basic PII patterns

---

## Learned Signals

Possible learned tasks:

- spam
- deception
- toxicity
- advertising
- off-topic
- contextual PII

---

## Similarity Signals

May come from:

- exact comparison
- lexical similarity
- vector similarity

---

## Important Rule

The system must preserve signal provenance.

Conceptual example:

```json
{
  "toxicity_score": 0.81,
  "advertising_score": 0.07,
  "rule_flags": ["PHONE_NUMBER_PRESENT"],
  "similarity_signal": 0.13
}
```
