# Copyright (c) Microsoft. All rights reserved.

import json

import pytest

from agentlightning.research.analysis import analyze, cluster_ci, interval_union, load_events
from agentlightning.research.demo import create
from agentlightning.research.events import Recorder


@pytest.fixture
def sample(tmp_path):
    folder = create(tmp_path / "run")
    return folder, analyze([folder], tmp_path / "analysis", n_bootstrap=50, plots=False)


def test_sample_denominators_zero_null_and_censoring(sample):
    folder, report = sample
    assert report["counts"] == {
        "scheduled": 14,
        "started": 13,
        "ended": 12,
        "valid_evaluations": 9,
        "confirmed_successes": 8,
        "unknown_successes": 4,
    }
    expected = {
        "valid_evaluation_rate": (9, 13),
        "confirmed_success_proportion": (8, 13),
        "evaluated_success_rate": (8, 9),
        "reward_missing_rate": (4, 13),
        "tool_error_rate": (2, 4),
        "retry_recovery_rate": (1, 1),
    }
    for key, (n, d) in expected.items():
        assert report["rates"][key]["numerator"] == n
        assert report["rates"][key]["denominator"] == d
        assert report["rates"][key]["value"] == pytest.approx(n / d)
    assert report["adapter"]["status"] == "unavailable_no_adapter_outputs"
    assert report["rates"]["time_budget_exhaustion_rate"]["value"] is None
    assert report["rates"]["time_budget_exhaustion_rate"]["confirmed_lower_bound"] == 1 / 13
    assert report["cost_per_confirmed_success"]["active_wall_seconds"]["value"] is None
    import csv

    with (folder.parent / "analysis/rollouts.csv").open(encoding="utf-8-sig") as f:
        rows = {r["rollout_id"]: r for r in csv.DictReader(f)}
    assert rows["failure-fault"]["reward"] == "0"
    assert rows["failure-fault"]["process_exited_normally"] == "True"
    assert rows["failure-fault"]["success"] == "False"
    assert rows["missing_reward-fault"]["reward"] == ""
    assert rows["missing_reward-fault"]["success"] == ""
    assert rows["still-running"]["ended"] == "False"
    assert rows["still-running"]["unfinished_calls"] == "1"


def test_round_risk_uses_actual_entry_and_task_pairing(sample):
    folder, _report = sample
    import csv

    with (folder.parent / "analysis/round_interruption_risk.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["at_risk_entered_round"] == "13"
    assert rows[0]["interrupted_here"] == "2"
    assert rows[0]["right_censored_here"] == "1"
    assert rows[1]["at_risk_entered_round"] == "1"
    with (folder.parent / "analysis/paired_conditions.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    success = next(r for r in rows if r["metric"] == "confirmed_success_proportion")
    assert success["paired_tasks"] == "6"
    assert float(success["mean_b_minus_a"]) == pytest.approx(-4 / 6)
    assert success["ci_method"] == "paired_task_cluster_bootstrap"


def test_parallel_intervals_are_not_wall_time():
    def span(a, b):
        return {
            "start": {"monotonic_ns": a * 10**9, "clock_id": "host:boot"},
            "end": {"monotonic_ns": b * 10**9, "clock_id": "host:boot"},
        }

    assert interval_union([span(0, 10), span(5, 15)]) == 15
    other = span(5, 15)
    other["start"]["clock_id"] = "another-host:boot"
    assert interval_union([span(0, 10), other]) is None


def test_task_cluster_ci_does_not_treat_repetitions_as_independent():
    rows = [{"task_id": "a", "v": 1}] * 30 + [{"task_id": "b", "v": 0}]
    ci = cluster_ci(rows, lambda r: r["v"], lambda r: 1, n_bootstrap=100, seed=7)
    assert ci["tasks"] == 2
    assert ci["low"] == 0 and ci["high"] == 1
    assert ci == cluster_ci(rows, lambda r: r["v"], lambda r: 1, n_bootstrap=100, seed=7)
    assert cluster_ci([{"task_id": None}], lambda r: 1, lambda r: 1)["low"] is None


def test_partial_tail_and_identity_only_deduplication(sample):
    folder, _ = sample
    path = folder / "events.jsonl"
    first = path.read_bytes().splitlines(keepends=True)[0]
    with path.open("ab") as f:
        f.write(first)
        f.write(b'{"partial":')
    events, health = load_events(folder)
    assert events
    assert health["truncated_tail"] and health["duplicate_event_ids"] == 1
    with path.open("ab") as f:
        f.write(b"\n{}\n")
    with pytest.raises(ValueError, match="Malformed"):
        load_events(folder)


def test_no_success_and_unpriced_costs_are_not_zero(sample, tmp_path):
    folder, _ = sample
    path = folder / "events.jsonl"
    events, _ = load_events(folder)
    for event in events:
        if event["event_type"] == "reward_observed":
            event["data"].update(value=0, success=False)
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    report = analyze([folder], tmp_path / "no-success", plots=False, n_bootstrap=10)
    assert report["cost_per_confirmed_success"]["generated_tokens"]["status"] == "unavailable_no_successes"
    assert report["cost_per_confirmed_success"]["generated_tokens"]["value"] is None


def test_observer_losses_are_accumulated_per_writer(tmp_path):
    events = [
        {
            "event_id": str(i),
            "clock_id": "machine:boot",
            "host": "machine",
            "pid": pid,
            "observer_lost_events_before_this_record": lost,
        }
        for i, (pid, lost) in enumerate([(11, 2), (11, 3), (22, 4), (22, 4)])
    ]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    _, integrity = load_events(tmp_path)
    assert integrity["observer_lost_events"] == 7


def test_adapter_baselines_use_observed_sample_weights(sample, tmp_path):
    folder, _ = sample
    recorder = Recorder(folder)
    for rid, count, reward, failures in [("success-control", 1, 1, 0), ("failure-control", 3, 0, 2)]:
        recorder.emit(
            "adapter_output",
            rid,
            level="trajectory",
            sample_count=count,
            prefix_comparisons=2,
            prefix_failures=failures,
            sample_rewards=[reward] * count,
            effective_rollout_reward=reward,
        )
    report = analyze([folder], tmp_path / "adapter", plots=False, n_bootstrap=10)
    assert report["adapter"]["complete_merge_ratio"] == 0.5
    assert report["adapter"]["prefix_merge_failure_rate"] == 0.5
    # No baseline can be manufactured without sample rewards.
    recorder.emit(
        "adapter_output",
        "success-fault",
        level="trajectory",
        sample_count=2,
        prefix_comparisons=None,
        prefix_failures=None,
    )
    analyze([folder], tmp_path / "missing-adapter", plots=False, n_bootstrap=10)


def test_duplicate_run_identity_rejected(sample, tmp_path):
    folder, _ = sample
    with pytest.raises(ValueError, match="Repeated run_id"):
        analyze([folder, folder], tmp_path / "twice", plots=False)


def test_uninstrumented_tools_and_rounds_are_unavailable_not_zero(tmp_path):
    from agentlightning.research.events import initialize

    initialize(
        tmp_path,
        {
            "model_version": None,
            "dataset_version": "unknown-harness",
            "sampling_parameters": {},
            "execution_budget": {},
            "seed": None,
            "fault_config": None,
        },
    )
    recorder = Recorder(tmp_path)
    recorder.emit("rollout_scheduled", "r", task_id="task")
    recorder.emit("rollout_started", "r")
    recorder.emit("rollout_finished", "r", process_exit_code=0)
    analyze([tmp_path], tmp_path / "analysis", plots=False, n_bootstrap=10)
    import csv

    with (tmp_path / "analysis/rollouts.csv").open(encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    assert row["tool_calls"] == row["llm_rounds"] == ""
    assert row["tool_calls_observed"] == "0"
    assert row["tool_call_coverage"] == "unknown_hidden_calls"


def test_confirmed_budget_rate_has_explicit_started_denominator(tmp_path):
    from agentlightning.research.events import initialize

    initialize(
        tmp_path,
        {
            "model_version": None,
            "dataset_version": "budget-fixture",
            "sampling_parameters": {},
            "execution_budget": {"seconds": 1},
            "seed": None,
            "fault_config": None,
        },
    )
    recorder = Recorder(tmp_path)
    for i in range(2):
        rid = str(i)
        recorder.emit("rollout_scheduled", rid, task_id=rid)
        recorder.emit("rollout_started", rid)
        recorder.emit(
            "termination_hint",
            rid,
            reason="time_budget_exhausted" if i else "completed",
            category="budget_exhaustion" if i else "normal",
            budget_dimension="time" if i else None,
        )
        recorder.emit("rollout_finished", rid, process_exit_code=1 if i else 0)
    report = analyze([tmp_path], tmp_path / "analysis", plots=False, n_bootstrap=10)
    rate = report["rates"]["time_budget_exhaustion_rate"]
    assert rate["numerator"] == 1 and rate["denominator"] == 2 and rate["value"] == 0.5
    assert rate["n_termination_unknown"] == 0


def test_sdk_gateway_retries_count_even_without_upstream_model_calls(tmp_path):
    from agentlightning.research.events import initialize

    initialize(
        tmp_path,
        {
            "model_version": None,
            "dataset_version": "sdk-fixture",
            "sampling_parameters": {},
            "execution_budget": {},
            "seed": None,
            "fault_config": {"synthetic": True},
        },
    )
    recorder = Recorder(tmp_path)
    recorder.emit("rollout_scheduled", "r", task_id="task")
    recorder.emit("rollout_started", "r")
    for i in range(3):
        with recorder.call(
            "gateway", "r", request={}, retry_group_id="sdk-operation", parent_call_id="sdk-operation"
        ) as call:
            call.result.update(
                response={"http_status": 503 if i < 2 else 200}, error={"type": "GatewayHTTPStatus"} if i < 2 else None
            )
    recorder.emit("rollout_finished", "r", process_exit_code=0)
    report = analyze([tmp_path], tmp_path / "analysis", plots=False, n_bootstrap=10)
    import csv

    with (tmp_path / "analysis/rollouts.csv").open(encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    assert row["retry_count"] == row["sdk_gateway_retry_count"] == "2"
    assert row["upstream_retry_count"] == "0"
    assert row["first_fault_kind"] == "gateway"
    assert report["rates"]["retry_recovery_rate"]["value"] == 1


def test_nested_upstream_and_sdk_retries_are_not_double_counted(tmp_path):
    from agentlightning.research.events import initialize

    initialize(
        tmp_path,
        {
            "model_version": None,
            "dataset_version": "nested-fixture",
            "sampling_parameters": {},
            "execution_budget": {},
            "seed": None,
            "fault_config": {"synthetic": True},
        },
    )
    recorder = Recorder(tmp_path)
    recorder.emit("rollout_scheduled", "r", task_id="task")
    recorder.emit("rollout_started", "r")
    for outer, inner_count in [(0, 2), (1, 1)]:
        with recorder.call("gateway", "r", request={}, retry_group_id="logical", parent_call_id="logical") as gateway:
            for inner in range(inner_count):
                with recorder.call(
                    "llm", "r", request={}, retry_group_id="logical", parent_call_id="logical", attempt_index=inner
                ) as model:
                    model.result["error"] = {"type": "HTTP503"} if outer == 0 else None
            gateway.result["error"] = {"type": "HTTP503"} if outer == 0 else None
    recorder.emit("rollout_finished", "r", process_exit_code=0)
    report = analyze([tmp_path], tmp_path / "analysis", plots=False, n_bootstrap=10)
    import csv

    with (tmp_path / "analysis/rollouts.csv").open(encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    assert row["retry_count"] == "2"
    assert row["sdk_gateway_retry_count"] == row["upstream_retry_count"] == "1"
    assert row["retry_episodes"] == "1"
    assert report["rates"]["retry_recovery_rate"]["value"] == 1
