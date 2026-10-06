"""End to end on a tiny fixture: dataset rows -> loader -> BM25 -> generator -> answer records ->
scoring -> refusal classification. Fakes stand in for OpenAI (HTTP transport) and the judge
(metric function), so the test needs no network, API key or Ollama. Unit tests check each
piece; this checks they fit together and that provenance survives the whole trip."""
import asyncio
import json

import httpx

from generation import Generator
from retrieval import BM25Index
from run_pipeline import answer_questions
from score_run import deterministic_scores, run_metric
from scorers import ANSWERED, REFUSED, classify_response
from tydiqa import load

DOC = "Café Ayre. The shipyard was founded by Amos Ayre in 1918.\nUnrelated: cheese comes from Somerset."


def span(sub):
    b, s = DOC.encode(), sub.encode()
    return b.index(s), b.index(s) + len(s)


def row(question, chosen, minimal):
    p0, p1 = span("Café Ayre."), span("The shipyard was founded by Amos Ayre in 1918.")
    p2 = span("Unrelated: cheese comes from Somerset.")
    starts = [minimal[0] if minimal else -1] * 3
    ends = [minimal[1] if minimal else -1] * 3
    return {"language": "english", "question_text": question, "document_title": "Burntisland",
            "document_url": "https://en.wikipedia.org/wiki/Burntisland", "document_plaintext": DOC,
            "passage_answer_candidates": {"plaintext_start_byte": [p0[0], p1[0], p2[0]],
                                          "plaintext_end_byte": [p0[1], p1[1], p2[1]]},
            "annotations": {"passage_answer_candidate_index": chosen, "minimal_answers_start_byte": starts,
                            "minimal_answers_end_byte": ends, "yes_no_answer": ["NONE"] * 3}}


ROWS = [row("Who founded the shipyard?", [1, 1, 1], span("Amos Ayre")),
        row("What is the population of Mars?", [-1, -1, -1], None)]


def fake_openai(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    user = body["messages"][1]["content"]
    text = "Amos Ayre founded it in 1918." if "shipyard" in user else "NOT_IN_CONTEXT"
    return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                                     "choices": [{"index": 0, "finish_reason": "stop",
                                                  "message": {"role": "assistant", "content": text}}],
                                     "usage": {"prompt_tokens": 100, "completion_tokens": 8, "total_tokens": 108}})


class Score:
    def __init__(self, value):
        self.value = value


async def fake_accuracy(user_input, response, reference):
    return Score(1.0 if reference.lower() in response.lower() else 0.0)


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    passages, questions = load(ROWS)
    gen = Generator("gpt-6-luna", prompt_version="v2-not-in-context", transport=httpx.MockTransport(fake_openai))
    rows = answer_questions(questions, BM25Index(passages), gen, tmp_path / "answers.jsonl", k=2)

    # the file on disk is what scoring reads, so check it rather than the returned objects
    saved = [json.loads(x) for x in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert saved == json.loads(json.dumps(rows, ensure_ascii=False))
    answerable, unanswerable = saved

    # provenance survives: the retrieved gold passage carries its title, URL and byte range
    assert answerable["gold_retrieved"] is True
    gold = next(p for p in answerable["retrieved"] if p["passage_id"] == answerable["gold_passage_id"])
    assert gold["title"] == "Burntisland" and gold["url"].endswith("/Burntisland")
    assert DOC.encode()[gold["start_byte"]:gold["end_byte"]].decode() == gold["text"]
    assert answerable["sent_settings"]["temperature"] == 0 and answerable["prompt_version"] == "v2-not-in-context"

    # scoring: string match and failure label from the saved record
    s = deterministic_scores(answerable)
    assert s["string_match"] is True and s["failure_label_string_match"] == "correct" and s["refusal"] == ANSWERED
    assert deterministic_scores(unanswerable)["refusal"] == REFUSED
    assert classify_response(unanswerable["answer"], unanswerable["prompt_version"])["method"] == "marker"

    # judge path with a fake metric: values and statuses flow through run_metric unchanged
    r = asyncio.run(run_metric(fake_accuracy, user_input=answerable["question"], response=answerable["answer"],
                               reference=answerable["gold_answers"][0]))
    assert r == {"value": 1.0, "status": "ok", "seconds": r["seconds"]}
