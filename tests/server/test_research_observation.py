# Copyright (c) Microsoft. All rights reserved.

import json

import httpx

from agentlightning.research.events import initialize
from agentlightning.server.routes.events import record_event
from tests.server.conftest import MODEL_NAME


def enable(monkeypatch, tmp_path):
    initialize(
        tmp_path,
        {
            "model_version": "fixture-v1",
            "dataset_version": "fixture",
            "sampling_parameters": {},
            "execution_budget": {},
            "seed": 42,
            "fault_config": {"synthetic": True},
        },
    )
    monkeypatch.setenv("AGL_RESEARCH_RUN_DIR", str(tmp_path))


def events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]


def test_raw_reward_and_duplicate_model_events_survive_views(client, auth_headers, monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    rid = client.post(
        "/api/rollouts", headers=auth_headers, json=[{"input": {"id": "task-1"}, "metadata": {"repetition": 2}}]
    ).json()[0]["rollout_id"]
    for value in [None, 0]:
        record_event(rid, "0", "reward", {"value": value, "reason": "em_miss"})
    for _ in range(2):
        record_event(
            rid, "0", "model_request", {"response": {"prompt_token_ids": [1], "choices": [{"token_ids": [2]}]}}
        )
    raw = events(tmp_path)
    values = [e["data"]["value"] for e in raw if e["event_type"] == "reward_observed"]
    assert values == [None, 0]
    assert (
        len(
            [
                e
                for e in raw
                if e["event_type"] == "framework_event" and e["data"]["framework_event_type"] == "model_request"
            ]
        )
        == 2
    )
    assert next(e for e in raw if e["event_type"] == "rollout_scheduled")["data"]["repetition"] == 2
    before = (tmp_path / "events.jsonl").read_bytes()
    client.get(f"/api/rollouts/{rid}/events", params={"format": "triplet"}, headers=auth_headers)
    assert (tmp_path / "events.jsonl").read_bytes() == before


def test_real_proxy_retry_loop_records_every_attempt_and_original_tokens(
    client, app, auth_headers, monkeypatch, tmp_path
):
    enable(monkeypatch, tmp_path)

    async def no_wait(**kwargs):
        pass

    monkeypatch.setattr("agentlightning.server.proxy._sleep_before_retry", no_wait)
    seen = []

    def upstream(request):
        seen.append(json.loads(request.content))
        status = 503 if len(seen) < 3 else 200
        return httpx.Response(
            status,
            json={"error": "fixture outage"}
            if status == 503
            else {
                "prompt_token_ids": [101, 102],
                "choices": [
                    {"token_ids": [201], "logprobs": {"token_logprobs": [-0.25]}, "message": {"content": "ok"}}
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    rid = client.post("/api/rollouts", headers=auth_headers, json=[{"input": {"id": "task"}}]).json()[0]["rollout_id"]
    client.post(
        "/api/models", headers=auth_headers, json=[{"model": MODEL_NAME, "endpoint": "http://fixture/v1", "version": 5}]
    )
    response = client.post(
        f"/proxy/rollout/{rid}/attempt/0/mode/train/openai/v1/chat/completions",
        headers={**auth_headers, "X-AGL-Research-Call-ID": "logical-1"},
        json={"messages": [{"role": "user", "content": "same prompt"}], "max_tokens": 20},
    )
    assert response.status_code == 200
    raw = events(tmp_path)
    calls = [e["data"] for e in raw if e["event_type"] == "call_finished" and e["data"]["kind"] == "llm"]
    assert len(calls) == 3
    assert [c["http_status"] for c in calls] == [503, 503, 200]
    assert len({c["call_id"] for c in calls}) == 3
    assert {c["retry_group_id"] for c in calls} == {"logical-1"}
    assert calls[1]["retry_of"] == calls[0]["call_id"]
    assert calls[2]["tokens"]["response_token_ids"]["value"] == [201]
    assert calls[2]["tokens"]["weight_version"]["value"] == 5
    assert calls[0]["tokens"]["response_token_ids"]["value"] is None
    assert seen[0] == seen[1] == seen[2]  # Observer does not alter sampling or retry requests.


def test_exhausted_timeouts_leave_raw_attempts_and_stack(client, app, auth_headers, monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)

    async def no_wait(**kwargs):
        pass

    monkeypatch.setattr("agentlightning.server.proxy._sleep_before_retry", no_wait)

    def upstream(request):
        raise httpx.ReadTimeout("injected upstream timeout", request=request)

    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    rid = client.post("/api/rollouts", headers=auth_headers, json=[{"input": {}}]).json()[0]["rollout_id"]
    client.post(
        "/api/models", headers=auth_headers, json=[{"model": MODEL_NAME, "endpoint": "http://fixture/v1", "version": 1}]
    )
    response = client.post(
        f"/proxy/rollout/{rid}/attempt/0/mode/train/openai/v1/chat/completions",
        headers=auth_headers,
        json={"messages": []},
    )
    assert response.status_code == 504
    calls = [e["data"] for e in events(tmp_path) if e["event_type"] == "call_finished" and e["data"]["kind"] == "llm"]
    assert len(calls) == 6
    assert all(c["error"]["type"] == "ReadTimeout" for c in calls)
    assert all("injected upstream timeout" in c["error"]["traceback"] for c in calls)
