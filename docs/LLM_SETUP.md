# Connecting an LLM

The search works without any LLM (`--explain template`). To get natural-language explanations from a real model, choose a backend.
Model names change frequently: use the name shown in your provider's documentation.

## Local Llama 3 with Ollama (recommended for real resumes - nothing leaves your machine)
```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b            # or a smaller one for low RAM, e.g. llama3.2:3b
python -m talent_rag search "Find me a Junior Data Analyst who knows SQL and Tableau" --index outputs/demo/index --explain ollama:llama3.1:8b
```
A server on another machine: `OLLAMA_HOST=http://192.168.1.20:11434`.

## OpenAI
```bash
export OPENAI_API_KEY=sk-...                  # PowerShell: $env:OPENAI_API_KEY="sk-..."
python -m talent_rag search "..." --explain openai:gpt-4o-mini
```

## Google Gemini
```bash
export GEMINI_API_KEY=...                     # key from Google AI Studio
python -m talent_rag search "..." --explain gemini:gemini-2.5-flash
```

## Any OpenAI-compatible server (vLLM, LM Studio, Groq, Together, Ollama's /v1 ...)
```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=anything-or-the-real-key
python -m talent_rag search "..." --explain openai:<model-name>
```

## In Python
```python
from talent_rag.index import TalentIndex
from talent_rag.searcher import TalentSearch
from talent_rag.llm import OllamaClient

res = TalentSearch(TalentIndex.load("outputs/demo/index")).search("Senior DevOps engineer with AWS", explain=True, llm=OllamaClient("llama3.1:8b"))
print(res.explanation.overall, res.explanation.grounded_ratio)
```

## What is sent to the model
Only `Candidate A/B/C`, a few structured facts (role, years, requested skills found/missing) and short **redacted** resume excerpts.
No names, e-mails, phone numbers, URLs, pronouns or personal-detail lines. With a **cloud** LLM the skills/employer text still leaves your
machine - use a local model (Ollama) or obtain the necessary consent and agreements before sending real candidate data to a provider.

## Troubleshooting
* `LLM request ... failed - connection problem` - the server is not running / wrong host. The CLI falls back to the template explainer and says why.
* `HTTP 401` - wrong or missing API key. `HTTP 404` - wrong model name.
* "LLM answer was not valid JSON" - small local models sometimes ignore the JSON instruction; try a larger model.
