# Copyright (c) Microsoft. All rights reserved.

import json
from types import SimpleNamespace

import httpx
import pytest

from agentlightning.research.events import initialize
from examples.search_r1.agents import search_r1_agent as agent


@pytest.mark.asyncio
@pytest.mark.parametrize("reward_delivery_failure", [False, True])
async def test_official_multiturn_search_and_em_are_observed_without_sampling_changes(
    tmp_path, monkeypatch, reward_delivery_failure
):
    import openai

    initialize(
        tmp_path,
        {
            "model_version": "fixture-api",
            "dataset_version": "fixture",
            "sampling_parameters": {},
            "execution_budget": {},
            "seed": None,
            "fault_config": {"synthetic": True},
        },
    )
    for key, value in {
        "AGL_RESEARCH_RUN_DIR": str(tmp_path),
        "AGL_RESEARCH_ROLLOUT_ID": "search-fixture",
        "QUESTION": "What is the capital of France?",
        "GOLDEN_ANSWERS": '["Paris"]',
        "AGL_KEY": "fixture",
        "AGL_EVENT_URL": "http://fixture/reward",
        "AGL_OPENAI_BASE_URL": "http://fixture/v1",
        "SEARCH_R1_RETRIEVAL_URL": "http://fixture/retrieve",
    }.items():
        monkeypatch.setenv(key, value)
    requests = []

    async def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/v1/chat/completions":
            requests.append(body)
            assert body["max_tokens"] == 500 and body["temperature"] == 1
            assert request.headers["x-agl-research-call-id"]
            if len(requests) == 1:
                text = "<think>Need a source.</think><search>capital France</search>"
            else:
                assert "<information>Doc 1(Title: France) Capital: Paris" in body["messages"][-1]["content"]
                text = "<think>The source answers it.</think><answer>Paris</answer>"
            return httpx.Response(
                200,
                json={
                    "id": "fixture",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "fixture",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": text},
                            "token_ids": [200 + len(requests)],
                        }
                    ],
                    "prompt_token_ids": [100],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        if request.url.path == "/retrieve":
            assert body == {"queries": ["capital France"], "topk": 3, "return_scores": True}
            return httpx.Response(
                200, json={"result": [[{"document": {"title": "France", "text": "Capital: Paris"}, "score": 0.9}]]}
            )
        if request.url.path == "/reward":
            if reward_delivery_failure:
                raise httpx.ReadTimeout("fixture reward delivery timeout", request=request)
            assert body["data"]["value"] == 1 and body["data"]["reason"] == "em_match"
            return httpx.Response(200, json={})
        raise AssertionError(request.url)

    original_openai = openai.AsyncOpenAI

    def make_openai(**kwargs):
        return original_openai(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(openai, "AsyncOpenAI", make_openai)
    # Replace only this example's client factory; do not monkeypatch the HTTPX class used by the SDK.
    monkeypatch.setattr(
        agent,
        "httpx",
        SimpleNamespace(
            AsyncClient=lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)
        ),
    )
    if reward_delivery_failure:
        with pytest.raises(httpx.ReadTimeout):
            await agent.SearchR1Agent().run()
    else:
        await agent.SearchR1Agent().run()
    raw = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    finished = [e["data"] for e in raw if e["event_type"] == "call_finished"]
    assert [c["kind"] for c in finished] == ["llm_client", "tool", "llm_client", "evaluation"]
    tool = next(c for c in finished if c["kind"] == "tool")
    assert tool["response"]["result"][0][0]["score"] == 0.9
    assert finished[0]["tokens"]["response_token_ids"]["value"] == [201]
    assert finished[0]["tokens"]["sampling_log_probs"]["status"] == "unavailable"
    assert len(requests) == 2
    reward = next(e for e in raw if e["event_type"] == "reward_observed")
    assert reward["data"]["value"] == 1 and reward["data"]["success"] is True
