# Copyright (c) Microsoft. All rights reserved.

import json
import os
import subprocess
import sys

import pytest

from agentlightning.research.events import Recorder, initialize, token_fields


def metadata():
    return {
        "model_version": None,
        "dataset_version": "fixture",
        "sampling_parameters": {},
        "execution_budget": {},
        "seed": None,
        "fault_config": None,
    }


def test_append_only_parallel_writers_and_immutable_metadata(tmp_path):
    initialize(tmp_path, metadata())
    with pytest.raises(FileExistsError):
        initialize(tmp_path, metadata())
    code = (
        "from agentlightning.research.events import Recorder;import sys;r=Recorder(sys.argv[1]);"
        '[r.emit("fixture", "r", index=i) for i in range(8)]'
    )
    workers = [subprocess.Popen([sys.executable, "-s", "-c", code, str(tmp_path)]) for _ in range(4)]
    assert all(p.wait(timeout=30) == 0 for p in workers)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(events) == 32
    assert len({e["event_id"] for e in events}) == 32
    assert len({e["event_seq"] for e in events}) == 32
    assert [e["event_seq"] for e in events] == sorted(e["event_seq"] for e in events)


def test_call_error_preserved_and_never_swallowed(tmp_path):
    initialize(tmp_path, metadata())
    recorder = Recorder(tmp_path)
    with pytest.raises(TimeoutError), recorder.call("tool", "r", request={"query": "same"}):
        raise TimeoutError("actual Python exception")
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[1]["data"]["error"]["type"] == "TimeoutError"
    assert "actual Python exception" in events[1]["data"]["error"]["traceback"]
    assert events[1]["data"]["response"] is None
    assert events[1]["data"]["duration_seconds"] >= 0


def test_original_token_ids_only_no_retokenization():
    fields = token_fields({"choices": [{"message": {"content": "hello"}}]})
    assert fields["response_token_ids"]["value"] is None
    assert fields["prompt_token_ids"]["status"] == "unavailable"
    assert fields["sampling_log_probs"]["status"] == "unavailable"
    actual = token_fields(
        {"prompt_token_ids": [91], "choices": [{"token_ids": [42], "logprobs": {"token_logprobs": [-0.2]}}]}, 7
    )
    assert actual["prompt_token_ids"]["value"] == [91]
    assert actual["response_token_ids"]["value"] == [42]
    assert actual["sampling_log_probs"]["value"] == [-0.2]
    assert actual["weight_version"]["value"] == 7


def test_observer_failure_is_visible_without_changing_call_result(tmp_path, monkeypatch):
    initialize(tmp_path, metadata())
    recorder = Recorder(tmp_path)
    from contextlib import contextmanager

    import agentlightning.research.events as module

    @contextmanager
    def fail_lock(path):
        raise OSError("injected disk failure")
        yield

    with monkeypatch.context() as patch:
        patch.setattr(module, "file_lock", fail_lock)
        with pytest.warns(RuntimeWarning, match="observation lost"):
            recorder.emit("lost", "r")
    recorder.emit("recovered", "r")
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["observer_lost_events_before_this_record"] == 1


def test_actual_external_sigterm_preserves_unfinished_attempt(tmp_path, monkeypatch):
    import signal
    import time

    initialize(tmp_path, metadata())
    env = dict(os.environ, AGL_RESEARCH_RUN_DIR=str(tmp_path), AGL_RESEARCH_ROLLOUT_ID="interrupted")
    recorder = Recorder(tmp_path)
    recorder.emit("rollout_scheduled", "interrupted", task_id="signal-test")
    code = (
        "from agentlightning.controller.local_reconciler import _run_local_reconciler_worker as run;"
        'raise SystemExit(run("tests.research.worker_fixture:WaitingAgent"))'
    )
    proc = subprocess.Popen([sys.executable, "-s", "-c", code], env=env)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            text = (tmp_path / "events.jsonl").read_text()
            if "call_started" in text:
                break
            if proc.poll() is not None:
                pytest.fail("Worker exited before starting fixture tool")
            time.sleep(0.03)
        else:
            pytest.fail("Worker did not start")
        proc.terminate()
        assert proc.wait(timeout=10) == -signal.SIGTERM
        recorder.emit(
            "termination_hint", "interrupted", reason="observer_sent_sigterm", category="external_interruption"
        )
        recorder.emit("rollout_finished", "interrupted", state="failed", process_exit_code=proc.returncode)
        from agentlightning.research.analysis import analyze

        report = analyze([tmp_path], tmp_path / "analysis", plots=False, n_bootstrap=10)
        assert report["counts"]["started"] == report["counts"]["ended"] == 1
        assert report["counts"]["valid_evaluations"] == 0
        assert report["termination_reasons"]["observer_sent_sigterm"] == 1
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
