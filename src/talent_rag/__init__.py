"""talent_rag: a Retrieval-Augmented-Generation talent search engine for resumes.

Recruiters ask in plain language ("Find me a Junior Data Analyst who knows SQL and Tableau");
the system retrieves the best-matching resumes from a vector database, then an LLM (or an offline
template explainer) explains *why* the top candidates fit - grounded in the resume text, with
personal data redacted - and a bias audit checks whether the search favours demographic groups.
"""

__version__ = "1.0.0"
__author__ = "Salik Hussain"
