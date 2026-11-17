.PHONY: install test lint demo index search app evaluate bias

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

demo:
	python -m talent_rag demo

# needs the Kaggle file in data/raw/ (see docs/DATASET.md)
index:
	python -m talent_rag build-index --data "data/raw/Entity Recognition in Resumes.json" --out outputs/kaggle_index --embedder bge-small --store faiss

search:
	python -m talent_rag search "Find me a Junior Data Analyst who knows SQL and Tableau" --index outputs/kaggle_index

evaluate:
	python -m talent_rag evaluate --data "data/raw/Entity Recognition in Resumes.json" --embedders lsa minilm bge-small --compare-anonymization --out outputs/kaggle_evaluation

bias:
	python -m talent_rag bias-audit --data "data/raw/Entity Recognition in Resumes.json" --embedder bge-small --out outputs/kaggle_bias

app:
	TALENT_INDEX=outputs/kaggle_index streamlit run app/streamlit_app.py
