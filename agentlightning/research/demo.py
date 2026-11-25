# Copyright (c) Microsoft. All rights reserved.

"""Small synthetic event fixture. No model, credentials, GPU, sampling job or training."""

from __future__ import annotations

import time
from pathlib import Path

from .events import Recorder, identifier, initialize


def create(directory: str | Path):
    initialize(
        directory,
        {
            "run_id": "synthetic-fixture-" + identifier(),
            "model_version": "synthetic-model-v1",
            "dataset_version": "six-task-fixture-v1",
            "harness_version": "agent-lightning-local-observer",
            "sampling_parameters": {"synthetic": True},
            "execution_budget": {"rounds": 3, "tokens": 64, "seconds": 1},
            "seed": 42,
            "fault_config": {"synthetic": True, "conditions": ["control", "fault"]},
            "pricing_usd_per_million": {"input": 0.5, "output": 1.0},
            "synthetic": True,
            "collection_capabilities": {"llm_client": True, "tools": True},
        },
    )
    recorder = Recorder(directory)
    cases = ["success", "failure", "missing_reward", "retry", "timeout", "external_interruption"]
    for index, scenario in enumerate(cases):
        for repetition, condition in enumerate(["control", "fault"]):
            rid = f"{scenario}-{condition}"
            recorder.emit("rollout_scheduled", rid, task_id=f"task-{index}", repetition=repetition, condition=condition)
            time.sleep(0.001)
            recorder.emit("rollout_started", rid, source="synthetic_fixture")
            recorder.emit("turn_entered", rid, round_index=1)
            with recorder.call("llm", rid, request={"prompt": scenario}, round_index=1) as model:
                model.result.update(response={"answer": "fixture"}, usage={"prompt_tokens": 10, "completion_tokens": 4})
            interrupted = scenario in {"timeout", "external_interruption"} and condition == "fault"
            if scenario == "retry" and condition == "fault":
                recorder.emit("turn_entered", rid, round_index=2)
                with recorder.call(
                    "tool", rid, tool_type="search", request={"query": "fixture"}, retry_group_id=rid, round_index=2
                ) as failed:
                    failed.result.update(
                        error={"type": "HTTP503", "category": "external_service"}, response={"error": "injected"}
                    )
                with recorder.call(
                    "tool",
                    rid,
                    tool_type="search",
                    request={"query": "fixture"},
                    retry_group_id=rid,
                    retry_of=failed.data["call_id"],
                    round_index=2,
                ) as recovered:
                    recovered.result.update(response={"results": ["fixture document"]})
                with recorder.call("llm", rid, request={"prompt": scenario}, round_index=2) as followup:
                    followup.result.update(
                        response={"answer": "fixture"}, usage={"prompt_tokens": 14, "completion_tokens": 3}
                    )
            if interrupted:
                reason = "time_budget_exhausted" if scenario == "timeout" else "external_signal"
                recorder.emit(
                    "termination_hint",
                    rid,
                    reason=reason,
                    category="budget_exhaustion" if scenario == "timeout" else "external_interruption",
                    budget_dimension="time" if scenario == "timeout" else None,
                )
                if scenario == "timeout":
                    try:
                        with recorder.call("tool", rid, tool_type="search", request={"query": "slow fixture"}):
                            raise TimeoutError("synthetic tool deadline")
                    except TimeoutError:
                        pass
            elif not (scenario == "missing_reward" and condition == "fault"):
                reward = 0 if scenario == "failure" and condition == "fault" else 1
                with recorder.call(
                    "evaluation", rid, tool_type="fixture_em", request={"answer": "fixture"}
                ) as evaluation:
                    evaluation.result.update(response={"reward": reward, "success": bool(reward)})
                recorder.emit(
                    "reward_observed",
                    rid,
                    value=reward,
                    reward_valid=True,
                    success=bool(reward),
                    source="synthetic_evaluator",
                )
                recorder.emit("termination_hint", rid, reason="completed", category="normal")
            else:
                recorder.emit("termination_hint", rid, reason="completed_without_reward", category="normal")
            recorder.emit(
                "rollout_finished",
                rid,
                state="failed" if interrupted else "succeeded",
                process_exit_code=-15 if interrupted else 0,
            )
    recorder.emit("rollout_scheduled", "queued-only", task_id="task-queued", repetition=0, condition="fault")
    recorder.emit("rollout_scheduled", "still-running", task_id="task-running", repetition=0, condition="fault")
    recorder.emit("rollout_started", "still-running", source="synthetic_fixture")
    recorder.emit("turn_entered", "still-running", round_index=1)
    recorder.emit(
        "call_started",
        "still-running",
        call_id="unfinished-call",
        kind="tool",
        tool_type="search",
        request={"query": "unfinished"},
        retry_group_id="unfinished-call",
    )
    recorder.emit("run_observation_ended", note="One ongoing rollout is intentionally right-censored")
    return Path(directory)
