# Bias, fairness and privacy

Hiring is a **high-risk** use of AI (in the EU it is explicitly classed as high-risk, and data-protection law such as GDPR applies to resumes).
This project is a technical demonstration with safeguards, **not** legal advice or a compliance certificate. Keep a human in the loop and follow the
rules that apply where you operate.

## What the project does
| Risk | Safeguard |
|---|---|
| Names, e-mail, phone, URLs leak into embeddings / LLM prompts | `anonymize()` redacts them before embedding and again before any LLM call |
| Gendered pronouns and titles act as signals | He/She/Mr/Mrs -> neutral forms |
| Lines that state protected characteristics (date of birth, marital status, gender, nationality, religion, father's name ...) | whole-line and inline redaction |
| LLM guesses or mentions demographics | system prompt forbids it; candidates are anonymous (A/B/C) |
| LLM invents facts | evidence-only prompt + exact quotes verified against the evidence; ungrounded quotes are flagged |
| The ranking depends on gender cues | **bias audit**: counterfactual swap + qualified-selection parity (`python -m talent_rag bias-audit`) |
| Raw personal data in the repository | `data/raw/` and built indexes are git-ignored |

## How the audit works (and what it cannot do)
1. **Counterfactual test** - names/e-mail/pronouns of gender-coded candidates are swapped; the same fitted embedder embeds the modified resumes; the same
   queries are run. Anonymisation ON must give *exactly* zero change (verified in the tests); OFF shows how much the raw text would matter.
2. **Qualified-selection parity** - among candidates the benchmark marks relevant, compare how often each group reaches the top-k (disparate-impact ratio,
   "four-fifths" rule of thumb, Fisher exact test).

Limits: gender is **inferred from first names** with a small, culturally incomplete lexicon (many names are "unknown"); samples are small; other protected
attributes (age, ethnicity, disability, caste, religion) are not audited unless you supply group labels; **proxies** (employer, college, location, writing style)
can still carry bias even after redaction; and a clean audit is not proof of fairness. Use `--embedder` real models and your real data and review the report
with a person who understands the context.

## Recommended practice
* Use the tool to **shortlist and explain**, never to automatically reject.
* Prefer a local LLM (Ollama) for real candidate data; document consent and retention.
* Re-run the audit whenever the embedder, chunking, re-ranking or data changes.
* Give candidates a way to request human review.
