# Dataset: Resume Entities for NER (Kaggle)

The brief recommends the Kaggle dataset **"Resume Entities for NER"** (originally published by Dataturks): a few hundred
(about 220) real-world resumes, each annotated with named entities. We use the annotations in two ways: to build structured
"profile" text for retrieval and to generate an **annotation-derived retrieval benchmark** (`docs/METHODOLOGY.md`).

## Get it
1. Sign in to Kaggle and search for **"Resume Entities for NER"** (or "Entity Recognition in Resumes").
2. Download and place the JSON file in `data/raw/`. The file is usually named `Entity Recognition in Resumes.json`
   (the Makefile and the CLI default use that name; if yours differs, pass `--data "<path>"`).
3. Check it:
```bash
python -m talent_rag stats --data "data/raw/Entity Recognition in Resumes.json"
```

## Format (JSON lines: one resume per line)
```json
{"content": "Name\nDesignation\n...resume text...",
 "annotation": [{"label": ["Skills"], "points": [{"start": 1295, "end": 1621, "text": "..."}]},
                {"label": ["Name"],   "points": [{"start": 0,    "end": 11,   "text": "Jane Doe"}]}],
 "extras": null}
```
Labels: `Name`, `Designation`, `Companies worked at`, `Skills`, `Years of Experience`, `Degree`, `College Name`,
`Graduation Year`, `Location`, `Email Address`. `end` is inclusive; the loader accepts inclusive and exclusive offsets and
tolerates missing / null annotations.

Other layouts also work: a JSON array of the same objects; a **CSV** with a text column (`Resume_str`, `Resume`, `text`,
`content`) and optional `Category`; or a **folder of `.txt` files** (then no entity annotations exist: skills are found with the
vocabulary instead, and the benchmark queries come from skill mentions only).

## Personal data - important
Resumes contain names, e-mail addresses, phone numbers and sometimes protected characteristics. The raw file is therefore
**git-ignored**, never commit it, and treat any resume collection you index as personal data (consent, retention, access).
The search engine redacts personal data before embedding and before anything is sent to an LLM (`docs/BIAS_AND_PRIVACY.md`).

## The bundled sample
`data/sample/resumes_sample.json` contains 45 **synthetic** resumes in this exact format (fictional people, employers and
colleges), so everything runs without the Kaggle file. It is not suitable for drawing conclusions.
