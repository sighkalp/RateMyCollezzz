\# RateMyCollezzz



RateMyCollezzz is a student-first college discovery, review, community,

and trust platform.



The project uses a controlled five-layer architecture designed to keep the

codebase compact, understandable, research-grounded, and maintainable.



\---



\# Architecture



\## Layer 1 — Experience



Folder:



`experience/`



Owns:



\- Discover UI

\- search

\- filters

\- college details

\- reviews UI

\- map interaction

\- 2D map navigation

\- 3D perspective navigation

\- gallery

\- compare

\- saved colleges

\- community experience



\---



\## Layer 2 — Platform Core



Folder:



`platform/`



Primary technology:



Django + Python



Owns:



\- accounts

\- colleges

\- reviews

\- ratings

\- communities

\- platform APIs

\- permissions

\- business logic



\---



\## Layer 3 — Review Intelligence / RRM



Folder:



`rrm/`



Owns review understanding and risk-signal generation.



Current research direction combines:



\- SentencePiece tokenization

\- compact Transformer encoder

\- Character CNN

\- feature fusion

\- multi-task prediction



Target review-risk signals include:



\- spam

\- deception

\- toxicity

\- advertising

\- off-topic

\- PII



The current custom-model size target is approximately 22–30M trainable

parameters.



This is a design target, not a proven optimum.



\---



\## Layer 4 — Trust \& Safety



Folder:



`trust/`



Consumes model and platform evidence.



Future responsibilities include:



\- behavioural analysis

\- coordination analysis

\- risk aggregation

\- moderation decisions

\- campaign detection



Core principle:



\*\*Layer 3 understands.\*\*



\*\*Layer 4 decides.\*\*



The RRM does not autonomously delete reviews.



\---



\## Layer 5 — Data / Security / Operations



Folder:



`ops/`



Owns operational infrastructure including:



\- PostgreSQL

\- Redis

\- security

\- audit

\- backups

\- monitoring

\- deployment

\- model storage



\---



\# Review Risk Model



The RRM is being designed for student-review language including:



\- English

\- Hinglish

\- Roman Hindi

\- Indian English

\- college-specific terminology



Conceptual model:



Review

↓

Validation

↓

Tokenizer

↓

Transformer Encoder + Character CNN

↓

Feature Fusion

↓

Multi-task Risk Heads

↓

Risk Signals



The model is evaluated scientifically against simpler and pretrained

baselines before custom-model claims are made.



\---



\# Research Principles



The project does not fabricate:



\- dataset sizes

\- model results

\- F1 scores

\- accuracy

\- latency

\- robustness improvements

\- novelty claims



Design targets remain design targets until experiments provide evidence.



The custom RRM must justify its complexity through:



\- baseline comparison

\- ablation

\- robustness testing

\- efficiency testing

\- error analysis



\---



\# UI Direction



The existing working UI has been preserved as legacy migration material.



Future Layer-1 development will preserve working behaviour while improving

architecture and visual quality.



The map experience supports:



\- 2D navigation

\- 3D perspective navigation

\- college-logo location pins

\- selected-pin emphasis

\- high map clarity



3D perspective is for navigation.



It does not fabricate college buildings or campus infrastructure.



\---



\# Repository Structure



Only five architectural root folders are used:



```text

RateMyCollezzz/

├── experience/

├── platform/

├── rrm/

├── trust/

└── ops/

