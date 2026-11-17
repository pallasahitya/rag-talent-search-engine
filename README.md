# RAG-Powered Talent Search Engine

**Maintained by Sivani Sahitya Palla**

An AI-powered talent discovery platform that helps recruiters find relevant candidates using natural language queries, Retrieval-Augmented Generation (RAG), vector databases, and Large Language Models.

Instead of manually going through hundreds of resumes, recruiters can simply describe the candidate they are looking for.

Example:

> Find me a Junior Data Analyst with SQL and Tableau skills.

The system retrieves relevant candidates, evaluates their skill matches, and generates evidence-grounded explanations.

**Note:** This repository is a modified version of the original project by [Salik Hussain](https://github.com/salikhussain71-code/rag-talent-search-engine). Original project attribution and license are retained.

---

## Key Features

- **Resume Parsing:** Processes resumes from supported JSON, CSV, and text formats.
- **RAG Pipeline:** Uses resume chunking, embeddings, and vector databases for retrieval.
- **Hybrid Search:** Combines semantic search and BM25 keyword search using Reciprocal Rank Fusion (RRF).
- **Requirement Matching:** Extracts skills, job roles, seniority, and experience requirements from natural language queries.
- **AI-Powered Explanations:** Generates explanations describing why candidates match a requirement.
- **Evidence Verification:** Checks generated quotes against the original resume text.
- **Privacy Protection:** Redacts personal information before embedding and LLM processing.
- **Bias Auditing:** Includes counterfactual testing and qualified-selection parity analysis.
- **Interactive Interface:** Provides a Streamlit-based recruiter chat interface.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| RAG | Retrieval-Augmented Generation |
| Vector Databases | FAISS, ChromaDB, NumPy |
| Search | Semantic Search, BM25, Hybrid Search |
| NLP & Embeddings | Hugging Face Transformers |
| LLM Integration | OpenAI-compatible APIs, Google Gemini, Ollama |
| Framework | LangChain |
| Interface | Streamlit |
| Testing | Pytest |

## Architecture

1. **Data Ingestion:** Load and process resume datasets.
2. **Anonymization:** Remove personal and sensitive information.
3. **Chunking:** Divide resumes into meaningful sections.
4. **Embedding:** Convert resume chunks into vector representations.
5. **Indexing:** Store embeddings in a vector database.
6. **Retrieval:** Retrieve relevant candidates using semantic, keyword, or hybrid search.
7. **Requirement Matching:** Evaluate candidates against requested skills and experience.
8. **Explanation:** Generate grounded candidate explanations.
9. **Evaluation:** Analyze retrieval performance and potential ranking bias.

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/pallasahitya/rag-talent-search-engine.git
cd rag-talent-search-engine
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

### 4. Run the demo

```bash
python -m talent_rag demo
```

### 5. Launch the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

The bundled demo uses synthetic sample resumes and an offline explanation fallback. Real embedding models and external LLM providers require their respective setup.

## Example Queries

```text
Find a Junior Data Analyst who knows SQL and Tableau.
```

```text
Find a Senior DevOps Engineer with 6+ years of experience,
AWS and Kubernetes skills.
```

The search engine identifies relevant candidates, evaluates requirement coverage, and provides supporting evidence.

## LLM Integrations

The project supports multiple explanation backends:

- Ollama for locally hosted models such as Llama 3.
- OpenAI-compatible API providers.
- Google Gemini.
- Offline template-based explanations.

Refer to `docs/LLM_SETUP.md` for configuration instructions.

## Privacy and Responsible AI

Recruitment is a high-impact application of AI. This project includes privacy and fairness mechanisms such as:

- Personal information redaction.
- Evidence-grounded candidate explanations.
- Counterfactual bias testing.
- Qualified-selection parity analysis.

These mechanisms help identify potential risks but do not eliminate bias. Human review should always remain part of recruitment decisions.

**Never upload real candidate resumes, personal information, or API keys to a public repository.**

Read `docs/BIAS_AND_PRIVACY.md` for additional information.

## Testing

The original project documentation reports 129 automated tests passing in its Python 3.12 CPU test environment.

The documented test coverage includes data parsing, anonymization, vector stores, retrieval, LLM clients, explanation grounding, bias auditing, CLI functionality, and the Streamlit interface.

Real Kaggle dataset evaluation and live external model/API performance were not verified in that test environment.

## Project Structure

```text
rag-talent-search-engine/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── sample/
│
├── docs/
│
├── notebooks/
│
├── scripts/
│
├── src/
│   └── talent_rag/
│
├── tests/
│
├── pyproject.toml
├── requirements.txt
├── Makefile
├── README.md
└── LICENSE
```

## Future Improvements

- Improve retrieval quality using real-world evaluation datasets.
- Experiment with different embedding models.
- Enhance candidate ranking and requirement matching.
- Improve the recruiter interface and user experience.
- Explore scalable deployment options.

## References

## Original Author & License

This project is based on the original work by **Salik Hussain**.

- **Original Author:** [Salik Hussain](https://github.com/salikhussain71-code)
- **Original Repository:** [RAG-Powered Talent Search Engine](https://github.com/salikhussain71-code/rag-talent-search-engine)
- **License:** MIT License

The original author's copyright notice and MIT License are retained. Modifications and further development are maintained by **Sivani Sahitya Palla**.

See the [LICENSE](LICENSE) file for the full license terms.

---

**Maintainer:** Sivani Sahitya Palla

**Repository:** [rag-talent-search-engine](https://github.com/pallasahitya/rag-talent-search-engine)
