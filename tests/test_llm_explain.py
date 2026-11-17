import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from talent_rag.explain import (
    SYSTEM_PROMPT,
    build_prompt,
    candidate_payload,
    explain_candidates,
    parse_llm_json,
    template_explanation,
    verify_quotes,
)
from talent_rag.llm import CallableLLM, GeminiClient, LLMError, OllamaClient, OpenAICompatibleClient, make_llm
from talent_rag.searcher import TalentSearch

Q = "Find me a Junior Data Analyst who knows SQL and Tableau"


class FakeServer:
    """A real local HTTP server that records requests and replies with a canned JSON body."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                status, payload = outer.replies.pop(0) if len(outer.replies) > 1 else outer.replies[0]
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):  # silence
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    made = []

    def make(replies):
        s = FakeServer(replies)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


def test_openai_compatible_request_and_response(server):
    srv = server([(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})])
    client = OpenAICompatibleClient("gpt-test", base_url=srv.url + "/v1", api_key="sk-secret")
    assert client.generate("sys", "user") == '{"ok": true}'
    req = srv.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["headers"]["authorization"] == "Bearer sk-secret"
    assert req["body"]["model"] == "gpt-test" and req["body"]["temperature"] == 0
    assert req["body"]["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}]
    assert req["body"]["response_format"] == {"type": "json_object"}


def test_openai_base_url_can_come_from_the_environment(server, monkeypatch):
    srv = server([(200, {"choices": [{"message": {"content": "{}"}}]})])
    monkeypatch.setenv("OPENAI_BASE_URL", srv.url + "/v1/")
    assert make_llm("openai:local-model").generate("s", "u") == "{}"
    assert srv.requests[0]["path"] == "/v1/chat/completions"


def test_gemini_request_and_response(server):
    srv = server([(200, {"candidates": [{"content": {"parts": [{"text": '{"a":'}, {"text": " 1}"}]}}]})])
    client = GeminiClient("gem-test", api_key="g-secret", base_url=srv.url + "/v1beta")
    assert client.generate("sys", "user") == '{"a": 1}'
    req = srv.requests[0]
    assert req["path"] == "/v1beta/models/gem-test:generateContent" and req["headers"]["x-goog-api-key"] == "g-secret"
    assert req["body"]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert req["body"]["contents"][0]["parts"][0]["text"] == "user"
    assert req["body"]["generationConfig"] == {"temperature": 0, "responseMimeType": "application/json"}


def test_ollama_request_and_response(server):
    srv = server([(200, {"message": {"role": "assistant", "content": '{"x": 2}'}})])
    client = OllamaClient("llama-test", host=srv.url)
    assert client.generate("sys", "user") == '{"x": 2}'
    req = srv.requests[0]
    assert req["path"] == "/api/chat" and req["body"]["stream"] is False and req["body"]["format"] == "json"
    assert req["body"]["model"] == "llama-test" and req["body"]["options"] == {"temperature": 0}


def test_http_errors_become_llm_errors_without_leaking_the_key(server):
    srv = server([(401, {"error": "invalid key sk-secret"})])
    with pytest.raises(LLMError) as exc:
        OpenAICompatibleClient("m", base_url=srv.url, api_key="sk-secret").generate("s", "u")
    assert "401" in str(exc.value) and "sk-secret" not in str(exc.value)


def test_transient_errors_are_retried(server, monkeypatch):
    monkeypatch.setattr("talent_rag.llm.time.sleep", lambda s: None)
    srv = server([(503, {"error": "busy"}), (200, {"choices": [{"message": {"content": "{}"}}]})])
    assert OpenAICompatibleClient("m", base_url=srv.url).generate("s", "u") == "{}"
    assert len(srv.requests) == 2


def test_malformed_and_unreachable_responses(server):
    srv = server([(200, {"unexpected": 1})])
    with pytest.raises(LLMError, match="Unexpected"):
        OpenAICompatibleClient("m", base_url=srv.url).generate("s", "u")
    with pytest.raises(LLMError, match="connection"):
        OllamaClient("m", host="http://127.0.0.1:9", timeout=2).generate("s", "u")


def test_make_llm():
    assert make_llm("none") is None and make_llm("template") is None and make_llm(None) is None
    assert make_llm("openai:gpt-x").model == "gpt-x" and make_llm("gemini:g-x").model == "g-x"
    assert make_llm("ollama:llama3.1:8b").model == "llama3.1:8b"  # only the first colon splits provider/model
    with pytest.raises(ValueError):
        make_llm("skynet:1")


# ------------------------------------------------------------------ explanation
@pytest.fixture(scope="module")
def result(index):
    return TalentSearch(index).search(Q, 3, "hybrid", 0.5)


def good_reply(payloads, extra_quote=None):
    cands = []
    for p in payloads:
        quote = " ".join(p["evidence"].split()[:6])
        cands.append({"candidate_id": p["candidate_id"], "fit_score": 4, "summary": f"Fits because {p['facts']['role']}.", "matched_requirements": ["SQL"],
                      "gaps": [], "evidence_quotes": [quote] + ([extra_quote] if extra_quote and p["candidate_id"] == "A" else [])})
    return json.dumps({"candidates": cands, "overall": "A is the strongest."})


def test_prompt_contains_redacted_evidence_and_the_safety_rules(result):
    payloads = [candidate_payload(h, "ABC"[i], c) for i, (h, c) in enumerate(zip(result.hits, result.coverages))]
    prompt = build_prompt(Q, payloads)
    assert Q in prompt and "Candidate A" in prompt and "Candidate C" in prompt and "JSON" in prompt
    for h in result.hits:  # privacy: nothing identifying reaches the LLM
        assert h.resume.name not in prompt and h.resume.get("Email Address")[0] not in prompt and h.resume.id not in prompt.replace("resume_id", "")
    assert "protected characteristic" in SYSTEM_PROMPT and "ONLY the facts" in SYSTEM_PROMPT


def test_llm_prompt_is_redacted_even_if_the_index_was_built_on_raw_text(raw_index):
    res = TalentSearch(raw_index).search(Q, 3, "dense")
    seen = {}
    explain_candidates(Q, res.hits, res.spec, res.coverages, CallableLLM(lambda s, u: seen.setdefault("p", u) or "{}"))
    for h in res.hits:
        assert h.resume.name not in seen["p"] and h.resume.get("Email Address")[0] not in seen["p"]


def test_full_llm_flow_with_grounded_quotes(result):
    payloads = [candidate_payload(h, "ABC"[i], c) for i, (h, c) in enumerate(zip(result.hits, result.coverages))]
    llm = CallableLLM(lambda s, u: good_reply(payloads), "fake-llm")
    ex = explain_candidates(Q, result.hits, result.spec, result.coverages, llm)
    assert ex.backend == "fake-llm" and ex.fallback_reason is None and ex.grounded_ratio == 1.0
    assert [c.resume_id for c in ex.candidates] == [h.resume.id for h in result.hits]
    assert all(c.fit_score == 4 and not c.ungrounded_quotes for c in ex.candidates) and ex.overall == "A is the strongest."


def test_hallucinated_quotes_are_flagged(result):
    payloads = [candidate_payload(h, "ABC"[i], c) for i, (h, c) in enumerate(zip(result.hits, result.coverages))]
    llm = CallableLLM(lambda s, u: good_reply(payloads, extra_quote="Led a team of 40 engineers at Google"))
    ex = explain_candidates(Q, result.hits, result.spec, result.coverages, llm)
    a = ex.candidates[0]
    assert a.ungrounded_quotes == ["Led a team of 40 engineers at Google"] and 0 < ex.grounded_ratio < 1


def test_bad_json_and_llm_failures_fall_back_to_the_template(result):
    ex = explain_candidates(Q, result.hits, result.spec, result.coverages, CallableLLM(lambda s, u: "Sure! Here is my opinion."))
    assert ex.backend.startswith("template") and "not valid JSON" in ex.fallback_reason and len(ex.candidates) == 3

    def boom(s, u):
        raise LLMError("server down")

    ex2 = explain_candidates(Q, result.hits, result.spec, result.coverages, CallableLLM(boom))
    assert ex2.backend.startswith("template") and "server down" in ex2.fallback_reason
    ex3 = explain_candidates(Q, result.hits, result.spec, result.coverages, CallableLLM(lambda s, u: json.dumps({"candidates": [{"candidate_id": "Z"}]})))
    assert "no known candidates" in ex3.fallback_reason


def test_json_extraction_tolerates_fences_and_chatter():
    obj = {"candidates": [{"candidate_id": "A"}], "overall": "x"}
    assert parse_llm_json("```json\n" + json.dumps(obj) + "\n```") == obj
    assert parse_llm_json("Here you go: " + json.dumps(obj) + " Hope it helps!") == obj
    assert parse_llm_json("nope") is None and parse_llm_json("") is None and parse_llm_json('{"candidates": 3}') is None


def test_quote_verification():
    ev = "Skills: Excel, SQL, Tableau.\nBuilt dashboards in Tableau."
    good, bad = verify_quotes(["built dashboards in tableau", "Skills: Excel, SQL", "Managed 12 people", ""], ev)
    assert good == ["built dashboards in tableau", "Skills: Excel, SQL"] and bad == ["Managed 12 people", ""]


def test_template_explainer_is_deterministic_and_grounded(result):
    payloads = [candidate_payload(h, "ABC"[i], c) for i, (h, c) in enumerate(zip(result.hits, result.coverages))]
    a = template_explanation(result.spec, payloads, result.coverages)
    b = template_explanation(result.spec, payloads, result.coverages)
    assert a == b and a.backend == "template (no LLM)"
    assert a.candidates[0].fit_score == 5 and a.candidates[0].evidence_quotes
    for c, p in zip(a.candidates, payloads):
        assert not verify_quotes(c.evidence_quotes, p["evidence"])[1]  # every template quote is verbatim from the evidence


def test_singular_year_grammar(result):
    payloads = [candidate_payload(h, "ABC"[i], c) for i, (h, c) in enumerate(zip(result.hits, result.coverages))]
    text = " ".join(c.summary for c in template_explanation(result.spec, payloads, result.coverages).candidates)
    assert "1 years" not in text


def test_no_hits(result):
    ex = explain_candidates(Q, [], result.spec, [], None)
    assert ex.candidates == [] and "No candidates" in ex.overall
