# RateMyCollezzz RRM — Production Dataset Source & Provenance Decision Packet

## 1. Document Status

STATUS:
FINAL — GATE A SOURCE DECISIONS RECORDED

PACKET STATUS:
FINAL — GATE A CLOSED

GATE A STATUS:
CLOSED — USER SOURCE DECISIONS RECORDED

Sources 5–7 are approved with conditions for downstream planning.
Sources 1–4 and 8 remain on HOLD and are not approved for production
ingestion/training.
No source is unconditionally approved.
Gate A remains a dataset-governance decision, not legal advice.

Basis branch:
main

Basis commit:
d6f62ef

---

## 2. Gate A Purpose and Non-Goals

This packet audits the identity, provenance, license, and label-mapping
suitability of candidate data sources for the RateMyCollezzz Review Model
Corpus (RMC).

Gate A answers:

- What is each source?
- What was it originally built to study?
- What are its actual labels?
- What are its rights and redistribution terms?
- Which RRM labels may it defensibly support?
- Which mappings are explicitly forbidden?

Gate A does NOT:

- download datasets
- collect reviews
- create corpus rows
- annotate records
- freeze language_mix
- freeze college_category
- freeze split ratios
- freeze grouping policy
- choose near-duplicate thresholds
- train tokenizer
- run pretraining
- train supervised RRM
- define Trust policy
- approve sources for the user
- provide legal advice

---

## 3. Evidence Standard and Source Hierarchy

Three evidence states apply to every factual claim:

VERIFIED: supported by an authoritative original source.

UNVERIFIED: authoritative evidence has not established the claim.

AMBIGUOUS: rights, terms, or evidence contain multiple layers or do not
support one unambiguous conclusion.

Authoritative source priority:

1. original paper / publisher
2. official author-maintained dataset page
3. official competition/data page
4. official dataset-specific repository
5. institutional page

Never transfer:

- paper license to dataset license
- related-repository license to another dataset license
- public download availability to redistribution permission

Internal project policy markers (LOCKED, PROPOSED, UNRESOLVED) are NOT
evidence states and must not appear in the external evidence table.

---

## 4. Canonical RRM Source-to-Label Contract

Locked canonical label order:

```
spam
deception
toxicity
advertising
off_topic
pii
```

A source may supervise an RRM task only when its real ground truth supports
that task. Unsupported labels remain UNKNOWN or unsupported.

### Global Deception Rule

Text alone does not establish deception. Ordinary annotator judgment does not
establish deception.

deception = 0/1 only when:

- independent provenance establishes truth status, OR
- a controlled experimental protocol establishes truth status

Otherwise:

deception = UNKNOWN (-1).

This rule applies to Human-Written RMC, Synthetic RMC, Real Platform reviews,
Controlled RMC, and all other sources.

---

## 5. External Evidence Verification

| Source | Claim Category | Authoritative Location | Finding | Status | Remaining Uncertainty |
|--------|---------------|----------------------|---------|--------|----------------------|
| Source 1 | Identity | ACL Anthology P11-1032 | Deceptive Opinion Spam Corpus (Ott et al., ACL 2011) | VERIFIED | None |
| Source 1 | Original task | ACL Anthology P11-1032 | Hotel-review deception detection | VERIFIED | None |
| Source 1 | Original labels | ACL Anthology P11-1032 | truthful / deceptive | VERIFIED | None |
| Source 1 | Dataset size | ACL Anthology P11-1032 | 800 positive reviews (400 truthful + 400 deceptive) | VERIFIED | None |
| Source 1 | Dataset composition | ACL Anthology P11-1032 | 20 Chicago hotels, 20 truthful + 20 deceptive per hotel | VERIFIED | None |
| Source 1 | Deceptive construction | ACL Anthology P11-1032 | Controlled Mechanical Turk | VERIFIED | None |
| Source 1 | Truthful source | ACL Anthology P11-1032 | TripAdvisor positive reviews | VERIFIED | None |
| Source 1 | Negative extension | ACL Anthology N13-1053 | 800 negative reviews | VERIFIED | None |
| Source 1 | Combined corpus size | ACL Anthology P11-1032 and N13-1053 | 1,600 reviews | VERIFIED | None |
| Source 1 | Dataset redistribution license | Not located in authoritative sources | No explicit dataset redistribution terms found | UNVERIFIED | Redistribution terms unconfirmed |
| Source 1 | Commercial/product use | Not located in authoritative sources | No explicit commercial-use grant or restriction found | UNVERIFIED | Commercial terms unconfirmed |
| Source 2 | Identity | Official Kaggle Toxic Comment Classification DATA/RULES pages | Jigsaw Toxic Comment Classification Challenge | VERIFIED | None |
| Source 2 | Original task | Official Kaggle DATA page | Toxic comment detection | VERIFIED | None |
| Source 2 | Source text | Official Kaggle DATA page | Wikipedia talk-page comments | VERIFIED | None |
| Source 2 | Original labels | Official Kaggle DATA page | toxic, severe_toxic, obscene, threat, insult, identity_hate | VERIFIED | None |
| Source 2 | Annotation method | Official Kaggle DATA page | Human raters | VERIFIED | None |
| Source 2 | Competition dataset license | Official Kaggle DATA page | CC0 | VERIFIED | Layered with underlying text license |
| Source 2 | Underlying text license | Official Kaggle DATA page | CC-SA-3.0 | VERIFIED | Competition rules are a separate access/use layer |
| Source 2 | Access/use rules | Official Kaggle RULES page | Subject to Competition Rules | VERIFIED | Non-commercial research restriction |
| Source 2 | Redistribution | Official Kaggle RULES page | Prohibited to parties who have not accepted rules | AMBIGUOUS | Redistribution scope depends on rule interpretation |
| Source 2 | Commercial/product use | Official Kaggle RULES page | Limited to non-commercial research and education | AMBIGUOUS | Commercial-use status depends on rule interpretation |
| Source 3 | Identity | Official Kaggle Unintended Bias DATA page | Jigsaw Unintended Bias in Toxicity Classification Challenge | VERIFIED | None |
| Source 3 | Source text | Official Kaggle competition OVERVIEW | Civil Comments | VERIFIED | None |
| Source 3 | Scale | Official Kaggle competition OVERVIEW | Approximately 2 million public comments | VERIFIED | Approximate figure from official source |
| Source 3 | Annotation method | Official Kaggle DATA page | Human raters | VERIFIED | None |
| Source 3 | Target/attribute values | Official Kaggle DATA page | Fraction of human raters applying the attribute | VERIFIED | None |
| Source 3 | Original task | Official Kaggle DATA page | Toxicity detection with identity-bias analysis | VERIFIED | None |
| Source 3 | Original labels | Official Kaggle DATA page | toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit | VERIFIED | None |
| Source 3 | Competition dataset license | Official Kaggle DATA page | CC0 | VERIFIED | Layered with underlying text license |
| Source 3 | Underlying text license | Official Kaggle DATA page | CC0 | VERIFIED | Competition rules are a separate access/use layer |
| Source 3 | Access/use rules | Official Kaggle competition terms | Subject to Competition Rules | VERIFIED | Non-commercial research restriction |
| Source 3 | Redistribution | Official Kaggle RULES page | Prohibited to parties who have not accepted rules | AMBIGUOUS | Redistribution scope depends on rule interpretation |
| Source 3 | Commercial/product use | Official Kaggle RULES page | Limited to non-commercial research and education | AMBIGUOUS | Commercial-use status depends on rule interpretation |
| Source 4 | Identity | Shebuti Rayana official YelpCHI maintainer page | YelpCHI dataset | VERIFIED | None |
| Source 4 | Original task | Rayana and Akoglu, KDD 2015 | Spam detection and reviewer-behavior analysis | VERIFIED | None |
| Source 4 | Original labels | Rayana and Akoglu, KDD 2015; YelpCHI maintainer page | Filtered reviews, spammer / non-spammer | VERIFIED | Label semantics require careful mapping |
| Source 4 | Dataset size | YelpCHI maintainer page | 67,395 reviews | VERIFIED | None |
| Source 4 | Business count | YelpCHI maintainer page | 201 hotels and restaurants | VERIFIED | None |
| Source 4 | Reviewer count | YelpCHI maintainer page | 38,063 reviewers | VERIFIED | None |
| Source 4 | Filtered review rate | YelpCHI maintainer page | 13.23% filtered reviews | VERIFIED | None |
| Source 4 | Spammer rate | YelpCHI maintainer page | 20.33% spammers | VERIFIED | None |
| Source 4 | Papers | YelpCHI maintainer page; ICWSM 2013; KDD 2015 | Mukherjee et al. (ICWSM 2013); Rayana and Akoglu (KDD 2015) | VERIFIED | None |
| Source 4 | Ground-truth quality | YelpCHI maintainer page | Near-ground-truth, not perfect truth | VERIFIED | Conditional mapping required |
| Source 4 | Dataset redistribution license | Not confirmed | YelpCHI-specific redistribution terms not confirmed | UNVERIFIED | Redistribution terms unconfirmed |
| Source 4 | Commercial/product use | Not confirmed | No explicit commercial-use grant or restriction found | UNVERIFIED | Commercial terms unconfirmed |

---

## 6. Summary Decision Matrix

| Source | Source Class | Verified Potential RRM Labels | Forbidden Automatic Mappings | Recommended Experimental Role | License Status | Redistribution Status | Commercial/Product Use Status | Privacy Risk | Domain Match | Agent Recommendation | Approval Status |
|--------|-------------|------------------------------|------------------------------|------------------------------|---------------|----------------------|------------------------------|-------------|-------------|---------------------|----------------|
| Source 1 | External — deception-specific | deception | spam, toxicity, advertising, off_topic, pii | Research reference | UNVERIFIED | UNVERIFIED | UNVERIFIED | Low | Low (hotel reviews, not college) | HOLD_NEEDS_REVIEW | HOLD |
| Source 2 | External — toxicity-specific | toxicity | spam, deception, advertising, off_topic, pii | Auxiliary training / external benchmark | AMBIGUOUS / LAYERED | AMBIGUOUS | AMBIGUOUS | Medium | Low (Wikipedia, not college reviews) | HOLD_NEEDS_REVIEW | HOLD |
| Source 3 | External — toxicity and bias | toxicity | spam, deception, advertising, off_topic, pii | Robustness / bias-analysis experiments | Layered CC0 / Competition Rules | AMBIGUOUS | AMBIGUOUS | Medium | Low (Civil Comments, not college reviews) | HOLD_NEEDS_REVIEW | HOLD |
| Source 4 | External — spam-related proxy | spam (conditional near-ground-truth proxy) | deception, toxicity, advertising, off_topic, pii | Research reference / external benchmark | UNVERIFIED | UNVERIFIED | UNVERIFIED | Medium / REQUIRES AUDIT | Low-Medium (business reviews, not college) | HOLD_NEEDS_REVIEW | HOLD |
| Source 5 | Internal — college-specific | spam, toxicity, advertising, off_topic, pii; deception UNKNOWN by default | deception unless independent truth provenance exists | Primary in-domain training corpus | UNRESOLVED — project control; contributor consent/release/governance not yet frozen | UNRESOLVED | UNRESOLVED | Low | High | APPROVE_WITH_CONDITIONS | APPROVE_WITH_CONDITIONS |
| Source 6 | Internal — controlled ground truth | Any canonical task individually when experimentally established | any label not established by the experiment; deception unless truthful/deceptive status is experimentally established | Auxiliary training controlled supplement | Project-controlled | Project-controlled | Project-controlled | Low | High | APPROVE_WITH_CONDITIONS | APPROVE_WITH_CONDITIONS |
| Source 7 | Internal — programmatic | Any label where parent provenance supports it and transformation preserves it | deception unless parent supports it or controlled generation establishes it; any label not supported by parent provenance | Robustness / augmentation | Project-generated synthetic: project-controlled; derived records: inherit parent-source restrictions | Project-generated synthetic: project-controlled; derived records: inherit parent-source restrictions | Project-generated synthetic: project-controlled; derived records: inherit parent-source restrictions | Depends on parent; requires PII inheritance/scrubbing audit | High | APPROVE_WITH_CONDITIONS | APPROVE_WITH_CONDITIONS |
| Source 8 | External — unverified | spam, toxicity, advertising, off_topic, pii; deception UNKNOWN by default | deception without independent truth provenance | Research reference / future domain realism | UNVERIFIED | UNVERIFIED | UNVERIFIED | High | High | HOLD_NEEDS_REVIEW | HOLD |

---

## 7. Detailed Source Audits

### Source 1 — Deceptive Opinion Spam Corpus

| Field | Value |
|-------|-------|
| source_name | Deceptive Opinion Spam Corpus |
| source_version | Not versioned |
| source_url_or_authoritative_location | ACL Anthology P11-1032; ACL Anthology N13-1053 |
| paper_reference | Ott et al., ACL 2011, "Finding Deceptive Opinion Spam by Any Stretch of the Imagination"; Ott et al., NAACL-HLT 2013, "Negative Deceptive Opinion Spam" |
| original_task | Hotel-review deception detection |
| original_labels | truthful / deceptive |
| language | English |
| domain | Hotel reviews |
| license_status | UNVERIFIED |
| license_evidence_source | No explicit dataset redistribution terms located in authoritative sources |
| redistribution_allowed | UNVERIFIED |
| citation_requirements | ACL Anthology paper citation |
| commercial_product_restrictions | UNVERIFIED |
| derivative_data_restrictions | UNVERIFIED |
| pii_risk | Low |
| duplicate_template_risk | Medium |
| documented_potential_rrm_labels | deception |
| forbidden_rrm_labels | spam, toxicity, advertising, off_topic, pii |
| proposed_experimental_role | Research reference; possible later auxiliary training after rights clarification and user decision |
| split_constraints | GATE-E RECOMMENDATION: no cross-split copies, paraphrases, derived variants, or translated versions |
| transformation_parent_child_constraints | GATE-E RECOMMENDATION: transformed versions remain grouped with originals |
| source_fingerprint_shortcut_risk | Medium — well-known academic dataset |
| production_ingestion_recommendation | HOLD_NEEDS_REVIEW |
| approval_status | HOLD |
| notes | Combined positive and negative corpus contains 1,600 reviews |
| limitations | Domain is hotel reviews, not college reviews. Performance on this dataset must not be presented as proof of college-domain deception detection. |

#### Identity and Evidence

Authoritative locations: ACL Anthology P11-1032 (positive corpus, 2011) and
ACL Anthology N13-1053 (negative extension, 2013).

2011 positive corpus: 400 truthful + 400 deceptive = 800 total, across 20
Chicago hotels, 20 truthful + 20 deceptive per hotel.

Deceptive reviews: controlled Mechanical Turk construction.

Truthful reviews: original TripAdvisor reviews sampled under the study design.

2013 negative extension: 800 negative reviews.

Combined positive + negative: 1,600 reviews.

#### Original Task and Labels

Original task: hotel-review deception detection.

Original labels:

- truthful
- deceptive

#### RRM-Compatible Use

May support:

```
deception
```

where the original provenance genuinely supports the truthful/deceptive
distinction.

May also be useful for:

- deception baseline experiments
- representation experiments
- domain-transfer analysis

#### Forbidden Use

Do not automatically derive:

```
spam
toxicity
advertising
off_topic
pii
```

unless those targets are independently annotated.

#### Domain Limitations

Domain is hotel reviews, not Indian college reviews.

```
useful deception evidence
    !=
college-domain representativeness
```

#### License / Terms

Dataset redistribution license: UNVERIFIED.

No explicit dataset redistribution terms located in authoritative sources.

Do not infer dataset rights from ACL paper publication licensing.

#### Redistribution / Product-Use Status

Redistribution: UNVERIFIED.

Commercial/product use: UNVERIFIED.

No explicit commercial-use grant or restriction found in authoritative sources.

#### Privacy / PII

Low risk. Reviews were collected through controlled academic research
protocols. No known PII exposure in published dataset.

#### Duplicate / Template / Shortcut Risk

Medium. Well-known academic dataset. Models may learn dataset-specific
artifacts rather than general deception patterns.

#### Source-Specific Future Split Recommendations

GATE-E RECOMMENDATION: records must not leak into RMC evaluation through
copies, paraphrases, derived variants, or translated versions. Transformed
versions must remain grouped with originals during splitting.

#### Agent Recommendation

HOLD_NEEDS_REVIEW.

#### Remaining Questions

- Dataset redistribution license terms
- Commercial-use status
- Suitability for college-domain experiments

#### Authoritative References

- Ott et al., ACL 2011, ACL Anthology P11-1032
- Ott et al., NAACL-HLT 2013, ACL Anthology N13-1053

---

### Source 2 — Jigsaw Toxic Comment Classification Challenge

| Field | Value |
|-------|-------|
| source_name | Jigsaw Toxic Comment Classification Challenge |
| source_version | Competition dataset (Kaggle) |
| source_url_or_authoritative_location | Official Kaggle competition DATA and RULES pages |
| paper_reference | Kaggle competition page; related work on toxic comment classification |
| original_task | Toxic comment classification |
| original_labels | toxic, severe_toxic, obscene, threat, insult, identity_hate |
| language | English |
| domain | Wikipedia talk-page comments |
| license_status | AMBIGUOUS / LAYERED |
| license_evidence_source | Official Kaggle DATA page (CC0 competition dataset); Official Kaggle DATA page (CC-SA-3.0 underlying Wikipedia text) |
| redistribution_allowed | AMBIGUOUS |
| citation_requirements | Kaggle competition citation requirements |
| commercial_product_restrictions | AMBIGUOUS |
| derivative_data_restrictions | Subject to Competition Rules |
| pii_risk | Medium |
| duplicate_template_risk | Low |
| documented_potential_rrm_labels | toxicity |
| forbidden_rrm_labels | spam, deception, advertising, off_topic, pii |
| proposed_experimental_role | Auxiliary training or external benchmark for approved research use |
| split_constraints | GATE-E RECOMMENDATION: subject to Kaggle competition rules |
| transformation_parent_child_constraints | GATE-E RECOMMENDATION: none specific beyond general leakage controls |
| source_fingerprint_shortcut_risk | Low |
| production_ingestion_recommendation | HOLD_NEEDS_REVIEW |
| approval_status | HOLD |
| notes | Competition data is CC0; underlying Wikipedia text is CC-SA-3.0. Competition rules are a separate access/use layer. |
| limitations | Domain is Wikipedia talk pages, not college reviews. Competition rules restrict use to non-commercial research. |

#### Identity and Evidence

Authoritative locations: official Kaggle competition DATA and RULES pages.

Source: Wikipedia talk-page comments.

Human-rater annotations across six toxicity labels.

#### Original Task and Labels

Original task: toxic comment classification.

Original labels:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

#### RRM-Compatible Use

May support:

```
toxicity
```

for approved research use.

May be useful for:

- auxiliary toxicity training
- toxicity baseline experiments
- external toxicity benchmark

#### Forbidden Use

Do not derive:

```
spam
deception
advertising
off_topic
pii
```

from toxicity annotations.

#### Domain Limitations

Domain is Wikipedia talk pages, not Indian college reviews.

#### License / Terms

License status: AMBIGUOUS / LAYERED.

Competition dataset: CC0 (official Kaggle DATA page).

Underlying Wikipedia text: CC-SA-3.0 (official Kaggle DATA page).

Access and use: subject to Competition Rules.

Do not simplify to unrestricted use.

#### Redistribution / Product-Use Status

Redistribution: AMBIGUOUS.

Competition rules prohibit providing data to parties who have not accepted
the rules.

Commercial/product use: AMBIGUOUS.

Competition rules limit use to non-commercial research, education, and
Kaggle forums.

#### Privacy / PII

Medium risk. Wikipedia talk-page comments may contain personal information
or references to real individuals.

#### Duplicate / Template / Shortcut Risk

Low. Large-scale dataset with diverse comment content.

#### Source-Specific Future Split Recommendations

GATE-E RECOMMENDATION: subject to Kaggle competition rules for access and
use.

#### Agent Recommendation

HOLD_NEEDS_REVIEW.

#### Remaining Questions

- Precise redistribution scope under competition rules
- Commercial-use determination
- Suitability for college-domain toxicity experiments

#### Authoritative References

- Official Kaggle Toxic Comment Classification Challenge DATA page
- Official Kaggle Toxic Comment Classification Challenge RULES page

---

### Source 3 — Jigsaw Unintended Bias in Toxicity Classification

| Field | Value |
|-------|-------|
| source_name | Jigsaw Unintended Bias in Toxicity Classification |
| source_version | Competition dataset (Kaggle) |
| source_url_or_authoritative_location | Official Kaggle competition DATA page |
| paper_reference | Kaggle competition OVERVIEW and DATA pages |
| original_task | Toxicity detection with identity-bias analysis |
| original_labels | toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit; identity attributes |
| language | English |
| domain | Civil Comments (public online comments) |
| license_status | Layered CC0 / Competition Rules |
| license_evidence_source | Official Kaggle DATA page |
| redistribution_allowed | AMBIGUOUS |
| citation_requirements | Kaggle competition citation requirements |
| commercial_product_restrictions | AMBIGUOUS |
| derivative_data_restrictions | Subject to Competition Rules |
| pii_risk | Medium |
| duplicate_template_risk | Medium — different labels/subgroups may apply to the exact same comment text |
| documented_potential_rrm_labels | toxicity |
| forbidden_rrm_labels | spam, deception, advertising, off_topic, pii |
| proposed_experimental_role | Robustness / bias-analysis experiments; auxiliary toxicity training |
| split_constraints | GATE-E RECOMMENDATION: subject to Kaggle competition rules |
| transformation_parent_child_constraints | GATE-E RECOMMENDATION: none specific beyond general leakage controls |
| source_fingerprint_shortcut_risk | Medium — identity/domain artifacts and repeated comment text may become model shortcuts |
| production_ingestion_recommendation | HOLD_NEEDS_REVIEW |
| approval_status | HOLD |
| notes | Identity mention does not imply toxicity. Target and attribute values are fractions of human raters applying the attribute. |
| limitations | Domain is Civil Comments, not college reviews. Competition rules restrict use. |

#### Identity and Evidence

Authoritative location: official Kaggle competition DATA page.

Source: Civil Comments.

Scale: approximately 2 million public comments.

Annotations: human raters.

Target and attribute values: fraction of human raters applying the attribute.

#### Original Task and Labels

Original task: toxicity detection with identity-bias analysis.

Original labels:

- toxicity
- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Identity attributes: identity mentioned in text.

Identity mention != toxicity.

#### RRM-Compatible Use

May support:

```
toxicity
```

May be useful for:

- robustness analysis
- bias/error analysis
- toxicity training
- false-positive analysis

#### Forbidden Use

Do not derive:

```
spam
deception
advertising
off_topic
pii
```

without independent labels.

#### Domain Limitations

Domain is Civil Comments (general public comments), not Indian college reviews.

#### License / Terms

License status: Layered CC0 / Competition Rules.

Competition dataset: CC0 (official Kaggle DATA page).

Underlying Civil Comments text: CC0 (official Kaggle DATA page).

Access and use: subject to Competition Rules.

#### Redistribution / Product-Use Status

Redistribution: AMBIGUOUS.

Competition rules restrict data sharing.

Commercial/product use: AMBIGUOUS.

Competition rules limit use to non-commercial research.

#### Privacy / PII

Medium risk. Public comments may contain personal information.

#### Duplicate / Template / Shortcut Risk

Medium. The same comment text may carry different target/subgroup labels
under the same identity mentioned condition. Large and diverse corpus, but
exact-text duplicates can occur with different labels.

#### Source-Specific Future Split Recommendations

GATE-E RECOMMENDATION: prevent exact-text duplicates from leaking across
future splits. Group records with identical text and different labels/
subgroups together so they stay in the same split. Subject to Kaggle
competition rules for access and use.

#### Agent Recommendation

HOLD_NEEDS_REVIEW.

#### Remaining Questions

- Precise redistribution scope under competition rules
- Commercial-use determination
- Suitability for college-domain toxicity robustness experiments

#### Authoritative References

- Official Kaggle Unintended Bias in Toxicity Classification Challenge DATA page
- Official Kaggle competition OVERVIEW page

---

### Source 4 — YelpCHI

| Field | Value |
|-------|-------|
| source_name | YelpCHI |
| source_version | Not versioned |
| source_url_or_authoritative_location | Shebuti Rayana official YelpCHI maintainer page |
| paper_reference | Mukherjee et al., ICWSM 2013; Rayana and Akoglu, KDD 2015 |
| original_task | Spam detection and reviewer-behavior analysis |
| original_labels | Filtered reviews (near-ground-truth); spammer / non-spammer |
| language | English |
| domain | Yelp hotel and restaurant reviews |
| license_status | UNVERIFIED |
| license_evidence_source | YelpCHI-specific redistribution terms not confirmed |
| redistribution_allowed | UNVERIFIED |
| citation_requirements | Original paper citations |
| commercial_product_restrictions | UNVERIFIED |
| derivative_data_restrictions | UNVERIFIED |
| pii_risk | Medium / REQUIRES AUDIT |
| duplicate_template_risk | Low |
| documented_potential_rrm_labels | spam (conditional near-ground-truth proxy) |
| forbidden_rrm_labels | deception, toxicity, advertising, off_topic, pii |
| proposed_experimental_role | Research reference or external benchmark for anomaly research |
| split_constraints | GATE-E RECOMMENDATION: verify redistribution terms before use |
| transformation_parent_child_constraints | GATE-E RECOMMENDATION: none specific beyond general leakage controls |
| source_fingerprint_shortcut_risk | Low |
| production_ingestion_recommendation | HOLD_NEEDS_REVIEW |
| approval_status | HOLD |
| notes | Yelp filtering signal is near-ground-truth, not perfect truth. Conditional spam mapping only. |
| limitations | Domain is Yelp business reviews, not college reviews. Original label semantics must not be renamed to deception. |

#### Identity and Evidence

Authoritative location: Shebuti Rayana official YelpCHI maintainer page.

Verified quantities:

- 67,395 reviews
- 38,063 reviewers
- 201 hotels and restaurants
- 13.23% filtered reviews
- 20.33% spammers

Related papers:

- Mukherjee et al., ICWSM 2013, "What Yelp Fake Review Filter Might Be Doing?"
- Rayana and Akoglu, KDD 2015, "Collective Opinion Spam Detection: Bridging Review Networks and Metadata"

#### Original Task and Labels

Original task: spam detection and reviewer-behavior analysis.

Original labels:

- filtered / non-filtered reviews (near-ground-truth)
- spammer / non-spammer

#### RRM-Compatible Use

May support:

```
spam (conditional near-ground-truth proxy)
```

as a research reference or external benchmark for anomaly research.

#### Forbidden Use

Do not map original labels to:

```
deception
toxicity
advertising
off_topic
pii
```

#### Domain Limitations

Domain is Yelp hotel and restaurant reviews, not Indian college reviews.

#### License / Terms

License status: UNVERIFIED.

YelpCHI-specific redistribution terms not confirmed.

#### Redistribution / Product-Use Status

Redistribution: UNVERIFIED.

Commercial/product use: UNVERIFIED.

#### Privacy / PII

Medium / REQUIRES AUDIT. User-authored review text may contain personal
information. Privacy and PII exposure must be audited before any product
ingestion.

#### Duplicate / Template / Shortcut Risk

Low.

#### Source-Specific Future Split Recommendations

GATE-E RECOMMENDATION: verify redistribution terms before use.

#### Agent Recommendation

HOLD_NEEDS_REVIEW.

#### Remaining Questions

- YelpCHI redistribution terms
- Commercial-use status
- Ground-truth audit of original label semantics
- Suitability for college-domain spam experiments

#### Authoritative References

- Shebuti Rayana official YelpCHI maintainer page
- Mukherjee et al., ICWSM 2013
- Rayana and Akoglu, KDD 2015

---

### Source 5 — Human-Written RMC Research Data

| Field | Value |
|-------|-------|
| source_name | Human-Written RMC Research Data |
| source_version | PLANNED / NOT YET VERSIONED |
| source_url_or_authoritative_location | Project-defined (internal) |
| paper_reference | Project-defined |
| original_task | College-review risk-label research corpus |
| original_labels | spam, toxicity, advertising, off_topic, pii (where defensibly annotated); deception = UNKNOWN by default |
| language | English, Hinglish, Roman Hindi, Indian English |
| domain | Indian college reviews |
| license_status | UNRESOLVED — project control; contributor consent/release/governance not yet frozen |
| license_evidence_source | Project-defined |
| redistribution_allowed | UNRESOLVED — pending contributor consent/release and project governance |
| citation_requirements | Project-defined |
| commercial_product_restrictions | UNRESOLVED — pending contributor rights/consent and project governance |
| derivative_data_restrictions | UNRESOLVED — derivative/redistribution policy to be defined by project governance |
| pii_risk | Low |
| duplicate_template_risk | Low |
| documented_potential_rrm_labels | spam, toxicity, advertising, off_topic, pii |
| forbidden_rrm_labels | deception unless independent truth provenance exists |
| proposed_experimental_role | Primary in-domain training corpus |
| split_constraints | GATE-E RECOMMENDATION: train/validation/test leakage must be prevented |
| transformation_parent_child_constraints | N/A |
| source_fingerprint_shortcut_risk | Low |
| production_ingestion_recommendation | APPROVE_WITH_CONDITIONS |
| approval_status | APPROVE_WITH_CONDITIONS |
| notes | Deception remains UNKNOWN (-1) by default unless independent provenance establishes truth status. |
| limitations | Requires consent, PII safeguards, annotation protocol, provenance tracking, dataset governance, and leakage controls before production use. |

#### Identity and Evidence

Project-defined internal corpus. College-review examples written specifically
for RMC research.

Coverage:

- English
- Hinglish
- Roman Hindi
- Indian English
- informal student language
- spelling variation
- college-domain terminology

#### Original Task and Labels

Original task: provide domain coverage for English, Hinglish, Roman Hindi,
Indian English, college terminology, and realistic student phrasing.

Independently annotatable labels:

- spam
- toxicity
- advertising
- off_topic
- pii

deception: UNKNOWN by default.

#### RRM-Compatible Use

May support:

```
spam
toxicity
advertising
off_topic
pii
```

with defensible annotation.

Primary role: in-domain training corpus.

#### Forbidden Use

Do not automatically map to deception unless independent truth provenance
exists.

#### Domain Limitations

Domain is Indian college reviews — highest domain match among all sources.

#### License / Terms

Project-controlled corpus. Contributor consent/release and project
governance must be established before production use.

#### Redistribution / Product-Use Status

Redistribution: UNRESOLVED. Consent/release and project governance must
permit redistribution before it is released.

#### Privacy / PII

Low risk. Written for research purposes. Requires PII safeguards in
production.

#### Duplicate / Template / Shortcut Risk

Low. Research-constructed with natural variation.

#### Source-Specific Future Split Recommendations

GATE-E RECOMMENDATION: train/validation/test leakage must be prevented.
Near-duplicate relationships must be audited.

#### Agent Recommendation

APPROVE_WITH_CONDITIONS.

#### Remaining Questions

- Consent procedures
- PII handling protocol
- Annotation protocol freeze
- Provenance metadata design
- Dataset governance framework

#### Authoritative References

- rrm/DATA_SOURCE_STRATEGY.md Section 10
- rrm/ANNOTATION_GUIDE.md
- rrm/PILOT_DATASET_DESIGN.md

---

### Source 6 — Controlled RMC Research Data

| Field | Value |
|-------|-------|
| source_name | Controlled RMC Research Data |
| source_version | Project-defined |
| source_url_or_authoritative_location | Project-defined (internal) |
| paper_reference | Project-defined |
| original_task | Controlled ground-truth experiments for RMC labels |
| original_labels | Any canonical task individually when experimentally established; deception = 0/1 only when truthful/deceptive status is actually established |
| language | English, Hinglish, Roman Hindi (per experiment design) |
| domain | College reviews (per experiment design) |
| license_status | Self-generated |
| license_evidence_source | Project-defined |
| redistribution_allowed | Self-generated |
| citation_requirements | Project-defined |
| commercial_product_restrictions | None — project-controlled license has no commercial restrictions |
| derivative_data_restrictions | None — project-controlled license allows derivative works |
| pii_risk | Low |
| duplicate_template_risk | Low |
| documented_potential_rrm_labels | Any canonical task when experimentally established |
| forbidden_rrm_labels | any label not established by the experiment; deception unless truthful/deceptive status is experimentally established |
| proposed_experimental_role | Auxiliary training controlled ground-truth supplement |
| split_constraints | GATE-E RECOMMENDATION: train/validation/test leakage must be prevented |
| transformation_parent_child_constraints | N/A |
| source_fingerprint_shortcut_risk | Low |
| production_ingestion_recommendation | APPROVE_WITH_CONDITIONS |
| approval_status | APPROVE_WITH_CONDITIONS |
| notes | A controlled record supports a label only when that experiment establishes the target. |
| limitations | Each experiment must individually establish ground truth for its target label. Deception requires controlled truthful/deceptive status establishment. |

#### Identity and Evidence

Project-defined internal corpus. Samples produced under conditions where the
true target is known.

#### Original Task and Labels

Original task: provide ground truth where ordinary text does not provide
sufficient annotation certainty.

A controlled record supports a label only when that experiment establishes
the target. Any canonical task may be supported individually when
experimentally established.

deception: 0/1 only when truthful/deceptive status is actually established;
otherwise UNKNOWN.

#### RRM-Compatible Use

May support any canonical RRM label individually when the controlled
experiment genuinely establishes that label's ground truth.

Role: auxiliary training controlled ground-truth supplement.

#### Forbidden Use

Do not claim label support from controlled generation unless the experiment
genuinely establishes that label.

#### Domain Limitations

Domain is college reviews per experiment design.

#### License / Terms

Project-controlled. No external license constraints.

#### Redistribution / Product-Use Status

Project-controlled.

#### Privacy / PII

Low risk. Controlled generation with no real personal data.

#### Duplicate / Template / Shortcut Risk

Low.

#### Source-Specific Future Split Recommendations

GATE-E RECOMMENDATION: train/validation/test leakage must be prevented.

#### Agent Recommendation

APPROVE_WITH_CONDITIONS.

#### Remaining Questions

- Specific controlled experiment designs
- Ground-truth establishment protocols per label

#### Authoritative References

- rrm/DATA_SOURCE_STRATEGY.md Section 3.3
- rrm/ANNOTATION_GUIDE.md

---

### Source 7 — Synthetic / Derived RMC Research Data

| Field | Value |
|-------|-------|
| source_name | Synthetic / Derived RMC Research Data |
| source_version | Project-defined |
| source_url_or_authoritative_location | Project-defined (internal) |
| paper_reference | Project-defined |
| original_task | Programmatic augmentation for RMC training and robustness testing |
| original_labels | Inherited from parent where parent provenance supports it and transformation preserves it; deception requires defensibly labeled parent or controlled generation establishing truth status |
| language | English, Hinglish, Roman Hindi (per generation design) |
| domain | College reviews (per generation design) |
| license_status | Project-generated synthetic: project-controlled; derived records: inherit parent-source restrictions |
| license_evidence_source | Project-defined |
| redistribution_allowed | Project-generated synthetic: project-controlled; derived records: inherit parent-source restrictions |
| citation_requirements | Project-defined |
| commercial_product_restrictions | Project-generated synthetic: project-controlled; derived records: inherit parent-source restrictions |
| derivative_data_restrictions | Derived records: inherit parent-source restrictions; synthetic identity preserved in provenance metadata |
| pii_risk | Depends on parent; requires PII inheritance/scrubbing audit |
| duplicate_template_risk | Medium — synthetic patterns may overlap with natural or other synthetic text |
| documented_potential_rrm_labels | Any label where parent provenance supports it and transformation preserves it |
| forbidden_rrm_labels | deception unless parent provenance supports it or controlled generation establishes truth status; any label not supported by parent provenance |
| proposed_experimental_role | Robustness / augmentation |
| split_constraints | GATE-E RECOMMENDATION: synthetic parent/child grouped together; train/validation/test leakage must be prevented |
| transformation_parent_child_constraints | Parent/child must remain grouped during splitting. Synthetic identity must be preserved in provenance metadata. |
| source_fingerprint_shortcut_risk | Medium — synthetic patterns may be learnable |
| production_ingestion_recommendation | APPROVE_WITH_CONDITIONS |
| approval_status | APPROVE_WITH_CONDITIONS |
| notes | Synthetic generation does not create ground truth automatically. Must not be represented as real reviews. Must not be a replacement for real final evaluation. Must not be used as performance evidence. |
| limitations | Cannot inherit labels that parent does not support. Deception requires specific conditions. Not a substitute for real data. |

#### Identity and Evidence

Project-defined internal corpus. Data generated or derived programmatically
for RMC research.

#### Original Task and Labels

Original task: augment training data, test robustness, or provide edge-case
coverage where natural examples are scarce.

A label may be inherited only if:

- parent provenance supports it, AND
- the transformation preserves it.

For deception:

- inherit only from a defensibly labeled parent, OR
- controlled generation genuinely establishes truthful/deceptive status.

Otherwise UNKNOWN.

#### RRM-Compatible Use

May support any RRM label where:

- parent provenance supports the label, AND
- the transformation preserves the label.

Role: robustness testing and limited augmentation.

#### Forbidden Use

Do not use synthetic data as:

- a replacement for real final evaluation
- performance evidence
- representation of real reviews

Do not inherit deception without meeting the conditions above.

#### Domain Limitations

Domain is college reviews per generation design.

#### License / Terms

Project-controlled license for synthetic data.
Derived records inherit parent-source license constraints.

#### Redistribution / Product-Use Status

Project-controlled. Synthetic records: project owns license.
Derived records: inherit parent-source redistribution status.

#### Privacy / PII

Low risk. Programmatic generation with no real personal data.

#### Duplicate / Template / Shortcut Risk

Medium. Synthetic patterns may overlap with natural text or other synthetic
data. Model may learn synthetic artifacts.

#### Source-Specific Future Split Recommendations

GATE-E RECOMMENDATION: parent/child synthetic records must remain grouped
together during splitting. Synthetic identity preserved in provenance metadata.

#### Agent Recommendation

APPROVE_WITH_CONDITIONS.

#### Remaining Questions

- Parent provenance tracking design
- Synthetic identity metadata schema
- Augmentation volume limits

#### Authoritative References

- rrm/DATA_SOURCE_STRATEGY.md Section 3.4
- rrm/DATASET_SPEC.md Principle 9

---

### Source 8 — Real Platform / Public College Reviews

| Field | Value |
|-------|-------|
| source_name | Real Platform / Public College Reviews |
| source_version | UNVERIFIED — no specific source selected |
| source_url_or_authoritative_location | UNVERIFIED — no specific source approved |
| paper_reference | Not yet established |
| original_task | Not yet established |
| original_labels | No original RRM labels yet; future annotation may support spam, toxicity, advertising, off_topic, pii; deception UNKNOWN by default |
| language | Not yet established |
| domain | College reviews |
| license_status | UNVERIFIED |
| license_evidence_source | UNVERIFIED |
| redistribution_allowed | UNVERIFIED |
| citation_requirements | UNVERIFIED |
| commercial_product_restrictions | UNVERIFIED |
| derivative_data_restrictions | UNVERIFIED |
| pii_risk | High |
| duplicate_template_risk | Not yet assessed |
| documented_potential_rrm_labels | spam, toxicity, advertising, off_topic, pii (subject to annotation quality) |
| forbidden_rrm_labels | deception without independent truth provenance |
| proposed_experimental_role | Research reference / future domain-realism source |
| split_constraints | GATE-E RECOMMENDATION: all standard leakage controls apply once source is approved |
| transformation_parent_child_constraints | N/A |
| source_fingerprint_shortcut_risk | Not yet assessed |
| production_ingestion_recommendation | HOLD_NEEDS_REVIEW |
| approval_status | HOLD |
| notes | No platform currently approved. Collection method not yet defined. |
| limitations | Requires full source audit before any use. No public review source should be imported until reviewed and approved. |

#### Identity and Evidence

No source has been identified or reviewed.

Collection method: SOURCE-SPECIFIC — NOT YET DEFINED.

#### Original Task and Labels

Not yet established.

Potentially annotatable labels:

- spam
- toxicity
- advertising
- off_topic
- pii

deception: UNKNOWN by default.

#### RRM-Compatible Use

May support:

```
spam
toxicity
advertising
off_topic
pii
```

if defensibly annotated.

deception remains UNKNOWN unless independent truth provenance exists.

Role: research reference / future domain-realism source.

#### Forbidden Use

Do not map to deception without independent truth provenance.

#### Domain Limitations

Domain is college reviews — highest potential domain match.

#### License / Terms

License status: UNVERIFIED.

#### Redistribution / Product-Use Status

Redistribution: UNVERIFIED.

Commercial/product use: UNVERIFIED.

#### Privacy / PII

High risk. Real public college reviews may contain personal information,
student names, instructor names, or other sensitive content.

#### Duplicate / Template / Shortcut Risk

Not yet assessed.

#### Source-Specific Future Split Recommendations

GATE-E RECOMMENDATION: all standard leakage controls apply once source is
approved.

#### Agent Recommendation

HOLD_NEEDS_REVIEW.

#### Remaining Questions

- Source identity
- Terms of use
- Copyright status
- Authorized access method
- Privacy assessment
- PII handling protocol
- Redistribution rights
- Provenance tracking
- Annotation feasibility

#### Authoritative References

- rrm/DATA_SOURCE_STRATEGY.md Section 3.4

---

## 8. Locked Leakage Principles vs Gate-E Decisions

### LOCKED NOW

The following principles are locked per DATASET_SPEC and project policy:

- train/validation/test leakage prevented
- exact/near-duplicate relationships audited
- transformed/derived children remain with parent during splitting
- final test protected from training influence
- dataset/split versions reproducible
- source identity/artifacts investigated
- training/tuning does not inspect final test outcomes
- synthetic data remains identifiable as synthetic
- metadata must not silently become a model feature

### NOT LOCKED UNTIL GATE E

The following decisions are deferred to Gate E:

- reviewer grouping
- author grouping
- college grouping
- hotel/business grouping
- time grouping
- exact split ratios
- stratification policy
- exact near-duplicate threshold
- exact grouping-identifier schema

Per-source suggestions in this packet are labeled GATE-E RECOMMENDATION.

---

## 9. User Source Decisions

| Source | Agent Recommendation | Recommended Role | Conditions | User Decision | User Notes |
|---------|---------------------|-----------------|------------|---------------|------------|
| Source 1 — Deceptive Opinion Spam Corpus | HOLD_NEEDS_REVIEW | Research reference | Rights clarification required | HOLD | |
| Source 2 — Jigsaw Toxic Comment Classification Challenge | HOLD_NEEDS_REVIEW | Auxiliary training / external benchmark | License and redistribution confirmation required | HOLD | |
| Source 3 — Jigsaw Unintended Bias in Toxicity Classification | HOLD_NEEDS_REVIEW | Robustness / bias-analysis experiments | Competition rules compliance required | HOLD | |
| Source 4 — YelpCHI | HOLD_NEEDS_REVIEW | Research reference / external benchmark | Ground-truth audit and rights confirmation required | HOLD | |
| Source 5 — Human-Written RMC Research Data | APPROVE_WITH_CONDITIONS | Primary in-domain training corpus | Consent, PII safeguards, annotation protocol, provenance, governance, leakage controls | APPROVE_WITH_CONDITIONS | |
| Source 6 — Controlled RMC Research Data | APPROVE_WITH_CONDITIONS | Auxiliary training controlled supplement | Ground-truth establishment protocol per experiment | APPROVE_WITH_CONDITIONS | |
| Source 7 — Synthetic / Derived RMC Research Data | APPROVE_WITH_CONDITIONS | Robustness / augmentation | Synthetic identity preserved, parent/child grouped, not performance evidence | APPROVE_WITH_CONDITIONS | |
| Source 8 — Real Platform / Public College Reviews | HOLD_NEEDS_REVIEW | Research reference / future domain realism | Full source audit required before any use | HOLD | |

No source is approved for production ingestion by this document.

Gate A closes only after the user records decisions for all eight sources.

---

## 10. Downstream Gate Architecture

Gate A — Dataset Source & Provenance Plan — CURRENT

Gate B — Production Annotation + Metadata Contract

Gate C — Production Corpus Collection + Annotation

Gate D — Dataset Version Freeze

Gate E — Split Policy + Manifest + Leakage Audit

Gate F — Production Tokenizer Corpus + Training

Gate G — Tokenizer Freeze

Gate H — RRM 3.6 Pretraining Execution

Gate I — Semantic Checkpoint Freeze

Gate J — Supervised Six-Task Training

Gate K — Three-Seed Neural Experiments

Gate L — Ablation + Fair Baseline Evaluation

Gate M — Final Scientific Evaluation

Gate N — Production Model Freeze

RRM 3.1–3.11 implementation infrastructure is already locked.

Gate H later EXECUTES the already-defined RRM 3.6 protocol; it does not
reimplement RRM 3.6.

Gate B resolves:

- production language_mix taxonomy
- production college_category taxonomy
- annotation-guide freeze
- adjudication protocol
- annotation-quality process
- required annotation metadata
- required provenance/grouping metadata

Gate C handles actual corpus collection and annotation.

---

## 11. Gate A Exit Contract

Gate A is complete when:

- all eight sources have an approval status other than PENDING_USER_DECISION
- user source decisions are recorded in Section 9
- conditions and holds are documented
- a separate Gate A closeout is performed

Gate A does not close on a timer. It closes only when the user completes
source decisions and unresolved claims are resolved or accepted.

Current state: CLOSED — USER SOURCE DECISIONS RECORDED.

All eight user decisions are recorded.
Conditions and holds are documented.
Gate A decision requirements are satisfied.
Next gate is Gate B.

---

## 12. Document Control

Version: 1.0

Basis branch: main

Basis commit: d6f62ef

Packet status: FINAL — GATE A CLOSED

Gate A status: CLOSED — USER SOURCE DECISIONS RECORDED

Next gate: Gate B — Production Annotation + Metadata Contract
