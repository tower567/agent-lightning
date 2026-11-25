# Copyright (c) Microsoft. All rights reserved.

"""Offline descriptive statistics. Unknown observations remain unknown."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .events import valid_reward


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def distribution(values):
    known = [v for v in values if v is not None and math.isfinite(v)]
    return {
        "n_observed": len(known),
        "n_unavailable": len(values) - len(known),
        "mean": statistics.mean(known) if known else None,
        "median": statistics.median(known) if known else None,
        "p90": percentile(known, 0.9),
        "p95": percentile(known, 0.95),
    }


def elapsed(start, end):
    if not start or not end or start["clock_id"] != end["clock_id"]:
        return None
    duration = (end["monotonic_ns"] - start["monotonic_ns"]) / 1e9
    return duration if duration >= 0 else None


def load_events(directory: Path):
    path = directory / "events.jsonl"
    events = []
    integrity = {"truncated_tail": False, "duplicate_event_ids": 0, "observer_lost_events": 0}
    seen = {}
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Only a killed writer's incomplete final record can be ignored, with a visible warning.
                if not line.endswith("\n") and not stream.read():
                    integrity["truncated_tail"] = True
                    break
                raise ValueError(f"Malformed JSONL record: {path}:{index}") from None
            event_id = event["event_id"]
            if event_id in seen:
                if seen[event_id] != event:
                    raise ValueError(f"Conflicting event_id: {event_id}")
                integrity["duplicate_event_ids"] += 1
                continue  # Identity only; never deduplicate text, prompts or commands.
            seen[event_id] = event
            integrity["observer_lost_events"] = max(
                integrity["observer_lost_events"], event.get("observer_lost_events_before_this_record", 0)
            )
            events.append(event)
    return events, integrity


def call_records(events):
    starts = {e["data"]["call_id"]: e for e in events if e["event_type"] == "call_started"}
    finishes = {e["data"]["call_id"]: e for e in events if e["event_type"] == "call_finished"}
    records = []
    for cid, event in starts.items():
        finish = finishes.get(cid)
        data = {**event["data"], **(finish["data"] if finish else {})}
        if finish and "started_monotonic_ns" in data:
            event = {**event, "monotonic_ns": data["started_monotonic_ns"], "utc": data["started_utc"]}
            finish = {**finish, "monotonic_ns": data["finished_monotonic_ns"]}
        records.append(
            {
                **data,
                "start": event,
                "end": finish,
                "finished": finish is not None,
                "duration": elapsed(event, finish),
                "error": data.get("error"),
                "ok": None if not finish else data.get("error") is None and data.get("status") == "ok",
            }
        )
    return records


def interval_union(calls):
    if not calls:
        return 0.0
    if any(not c["end"] for c in calls) or len({c["start"]["clock_id"] for c in calls}) != 1:
        return None
    intervals = sorted((c["start"]["monotonic_ns"], c["end"]["monotonic_ns"]) for c in calls)
    total = 0
    lo, hi = intervals[0]
    for start, end in intervals[1:]:
        if start > hi:
            total += hi - lo
            lo, hi = start, end
        else:
            hi = max(hi, end)
    return (total + hi - lo) / 1e9


def token_count(call, kind):
    value = (call.get("usage") or {}).get(kind)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    name = "response_token_ids" if kind == "completion_tokens" else "prompt_token_ids"
    field = (call.get("tokens") or {}).get(name) or {}
    return len(field["value"]) if isinstance(field.get("value"), list) else None


def price(call, metadata):
    # This is a provider-price estimate, not a claim about the invoice or GPU/infrastructure cost.
    prices = metadata.get("pricing_usd_per_million") or {}
    inp, out = token_count(call, "prompt_tokens"), token_count(call, "completion_tokens")
    if inp is None or out is None or not all(valid_reward(prices.get(k)) for k in ("input", "output")):
        return None
    return (inp * prices["input"] + out * prices["output"]) / 1e6


def complete_sum(values):
    return sum(values) if all(v is not None for v in values) else None


def repeated_behavior(calls):
    counts = Counter()
    previous = None
    streak = maximum = 0
    for call in calls:
        key = json.dumps([call["kind"], call.get("tool_type"), call.get("request")], sort_keys=True, ensure_ascii=False)
        counts[key] += 1
        streak = streak + 1 if key == previous else 1
        maximum = max(maximum, streak)
        previous = key
    return sum(n - 1 for n in counts.values()), maximum


def spearman(pairs):
    if len(pairs) < 3:
        return None

    def ranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        ranked = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position + 1
            while end < len(order) and values[order[end]] == values[order[position]]:
                end += 1
            for index in order[position:end]:
                ranked[index] = (position + end - 1) / 2
            position = end
        return ranked

    x, y = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    return statistics.correlation(x, y) if len(set(x)) > 1 and len(set(y)) > 1 else None


def summarize_rollout(events, metadata):
    def of(kind):
        return [e for e in events if e["event_type"] == kind]

    scheduled = of("rollout_scheduled")
    starts = of("rollout_started")
    terminals = [
        e for e in of("rollout_state") if e["data"].get("state") in {"succeeded", "failed", "SUCCEEDED", "FAILED"}
    ]
    ends = of("rollout_finished") or terminals or of("worker_finished")
    start = starts[0] if starts else None
    end = ends[-1] if ends else None
    info = scheduled[0]["data"] if scheduled else {}
    calls = call_records(events)
    # Physical model attempts are authoritative for token/cost accounting. Client spans are NOT added again.
    models = [c for c in calls if c["kind"] == "llm"]
    clients = [c for c in calls if c["kind"] == "llm_client"]
    tools = [c for c in calls if c["kind"] == "tool"]
    llm_rounds = (
        len(clients)
        if clients
        else len({c.get("parent_call_id") or c.get("retry_group_id") or c["call_id"] for c in models})
    )
    rewards = of("reward_observed")
    reward = rewards[-1]["data"].get("value") if rewards else None
    evaluated = valid_reward(reward)
    success = rewards[-1]["data"].get("success") if evaluated else None
    if not isinstance(success, bool):
        success = None
    hints = of("termination_hint")
    hint = hints[-1]["data"] if hints else {}
    raw_error = (end or {}).get("data", {}).get("error_message") or (end or {}).get("data", {}).get("error")
    exits = of("process_exit_observed") or of("worker_finished") or ends
    exit_code = exits[-1]["data"].get("process_exit_code") if exits else None
    reason = hint.get("reason")
    category = hint.get("category")
    worker_errors = [e["data"]["error"] for e in of("worker_finished") if e["data"].get("error")]
    if worker_errors and category in {None, "normal"}:
        reason = "worker_exception"
        category = worker_errors[-1].get("category", "unknown")
    if reason is None:
        reason = (
            "completed"
            if end
            and not raw_error
            and (end["data"].get("state") in {"succeeded", "SUCCEEDED"} or end["data"].get("process_exit_code") == 0)
            else "unknown"
            if end
            else "not_ended"
        )
        category = "normal" if reason == "completed" else "unknown"
    round_entries = sorted({e["data"]["round_index"] for e in of("turn_entered")})
    cost_observations = models or clients
    known_tokens = [token_count(c, "completion_tokens") for c in cost_observations]
    known_prompt = [token_count(c, "prompt_tokens") for c in cost_observations]
    prices = [price(c, metadata) for c in cost_observations]
    capabilities = dict(metadata.get("collection_capabilities") or {})
    for event in of("capture_capabilities"):
        capabilities.update(event["data"])
    logical_rounds_known = (
        bool(clients) or capabilities.get("llm_client") or (models and all(c.get("parent_call_id") for c in models))
    )
    tools_complete = capabilities.get("tools") is True
    retry_groups = defaultdict(list)
    for c in calls:
        if c["kind"] in {"llm", "tool", "evaluation"}:
            retry_groups[(c["kind"], c.get("retry_group_id") or c["call_id"])].append(c)
    episodes = [group for group in retry_groups.values() if len(group) > 1]
    extra = [c for group in episodes for c in group[1:]]
    recovered = sum(group[0]["ok"] is False and any(c["ok"] is True for c in group[1:]) for group in episodes)
    recovery_eligible = sum(group[0]["ok"] is False for group in episodes)
    # Comparable requests are a behavior clue; no attempt is removed using their content.
    logical = clients or [group[0] for (kind, _), group in retry_groups.items() if kind == "llm"]
    repeated, consecutive = repeated_behavior(sorted(logical + tools, key=lambda c: c["start"]["event_seq"]))
    tool_repeated, tool_consecutive = repeated_behavior(tools)
    failures = [c for c in calls if c["kind"] in {"llm", "tool", "evaluation"} and c["ok"] is False]
    first_failure = failures[0] if failures else None
    fault_round = None
    if first_failure:
        if first_failure.get("round_index") is not None:
            fault_round = first_failure["round_index"]
        else:
            entries = [
                e["data"]["round_index"]
                for e in of("turn_entered")
                if e["event_seq"] < first_failure["start"]["event_seq"]
            ]
            fault_round = max(entries) if entries else None
    row = {
        "run_id": metadata["run_id"],
        "rollout_id": events[0]["rollout_id"],
        "task_id": info.get("task_id"),
        "repetition": info.get("repetition"),
        "condition": info.get("condition") or metadata.get("condition") or "unspecified",
        "scheduled": bool(scheduled),
        "started": bool(start),
        "ended": bool(end),
        "valid_evaluation": evaluated,
        "reward": reward,
        "success": success,
        "reward_missing": not evaluated,
        "reward_observation_count": len(rewards),
        "reward_conflict": len({json.dumps(e["data"].get("value"), sort_keys=True) for e in rewards}) > 1,
        "execution_status": end["data"].get("state", "process_ended")
        if end
        else "running_or_unobserved"
        if start
        else "not_started",
        "process_exit_code": exit_code,
        "process_exited_normally": exit_code == 0 if exit_code is not None else None,
        "termination_reason": reason,
        "termination_category": category,
        "raw_error": raw_error,
        "started_utc": start["utc"] if start else None,
        "ended_utc": end["utc"] if end else None,
        "active_wall_seconds": elapsed(start, end),
        "queue_seconds": elapsed(scheduled[0] if scheduled else None, start),
        "scheduled_to_end_seconds": elapsed(scheduled[0] if scheduled else None, end),
        "wall_observed_lower_bound_seconds": elapsed(start, events[-1]) if start else None,
        "llm_rounds": llm_rounds if logical_rounds_known else None,
        "llm_rounds_observed": llm_rounds,
        "tool_calls_observed": len(tools),
        "tool_call_coverage": "declared_complete" if tools_complete else "unknown_hidden_calls",
        "sdk_retry_linkage_available": bool(clients) or bool(models and all(c.get("parent_call_id") for c in models)),
        "llm_attempts": len(models),
        "tool_calls": len(tools) if tools_complete else None,
        "generated_tokens": complete_sum(known_tokens) if models else None,
        "generated_tokens_observed_lower_bound": sum(v for v in known_tokens if v is not None),
        "prompt_tokens": complete_sum(known_prompt) if models else None,
        "unmeasured_llm_attempts": sum(v is None for v in known_tokens) if models else None,
        "estimated_llm_cost_usd": complete_sum(prices) if models else None,
        "estimated_cost_observed_lower_bound_usd": sum(v for v in prices if v is not None),
        "tool_errors": sum(c["ok"] is False for c in tools),
        "unfinished_tool_calls": sum(not c["finished"] for c in tools),
        "retry_count": len(extra),
        "retry_episodes": len(episodes),
        "recovery_eligible_episodes": recovery_eligible,
        "recovered_retry_episodes": recovered,
        "retry_recovery_rate": ratio(recovered, recovery_eligible),
        "retry_extra_call_seconds": complete_sum([c["duration"] for c in extra]),
        "retry_extra_tokens": complete_sum([token_count(c, "completion_tokens") for c in extra if c["kind"] == "llm"]),
        "retry_extra_estimated_usd": complete_sum([price(c, metadata) for c in extra if c["kind"] == "llm"]),
        "retry_wait_seconds": complete_sum([c["duration"] for c in calls if c["kind"] == "retry_wait"]),
        "repeated_requests": repeated,
        "max_consecutive_identical_requests": consecutive,
        "repeated_tool_commands": tool_repeated,
        "max_consecutive_identical_tool_commands": tool_consecutive,
        "first_fault_call_id": first_failure["call_id"] if first_failure else None,
        "first_fault_kind": first_failure["kind"] if first_failure else None,
        "first_fault_round": fault_round,
        "first_fault_round_source": "explicit_or_preceding_recorded_round"
        if fault_round is not None
        else "unavailable",
        "first_fault_seconds": elapsed(start, first_failure["start"]) if first_failure else None,
        "fault_status": "observed_fault" if first_failure else "no_observed_fault",
        "retry_status": "retried" if extra else "no_observed_retry",
        "tool_types": sorted({c.get("tool_type") or "unknown" for c in tools}),
        "entered_rounds": round_entries,
        "interruption_round": max(round_entries)
        if end and category in {"external_interruption", "timeout", "budget_exhaustion"} and round_entries
        else None,
        "time_budget_exhausted": any(
            e["data"].get("budget_dimension") == "time" and e["data"].get("category") == "budget_exhaustion"
            for e in hints
        ),
        "turn_budget_exhausted": any(
            e["data"].get("budget_dimension") == "turns" and e["data"].get("category") == "budget_exhaustion"
            for e in hints
        ),
        "token_budget_exhausted": any(
            e["data"].get("budget_dimension") == "tokens" and e["data"].get("category") == "budget_exhaustion"
            for e in hints
        ),
        "budget_observation_available": bool(hints),
        "unfinished_calls": sum(not c["finished"] for c in calls),
    }
    for kind in ["llm", "tool", "evaluation"]:
        chosen = [c for c in calls if c["kind"] == kind]
        row[kind + "_call_seconds_sum"] = complete_sum([c["duration"] for c in chosen])
        row[kind + "_elapsed_seconds_union"] = interval_union(chosen)
    outputs = of("adapter_output")
    adapter = outputs[-1]["data"] if outputs else {}
    row.update(
        adapter_sample_count=adapter.get("sample_count"),
        adapter_level=adapter.get("level"),
        adapter_prefix_comparisons=adapter.get("prefix_comparisons"),
        adapter_prefix_failures=adapter.get("prefix_failures"),
        adapter_sample_rewards=adapter.get("sample_rewards"),
        adapter_effective_reward=adapter.get("effective_rollout_reward"),
    )
    return row


def cluster_ci(rows, numerator, denominator, *, seed=42, n_bootstrap=1000):
    if not rows or any(r.get("task_id") is None for r in rows):
        return {"low": None, "high": None, "method": "unavailable_missing_task_id"}
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    if len(groups) < 2:
        return {"low": None, "high": None, "method": "unavailable_fewer_than_two_tasks"}
    totals = [(sum(numerator(r) for r in group), sum(denominator(r) for r in group)) for group in groups.values()]
    rng = random.Random(seed)
    samples = []
    for _ in range(n_bootstrap):
        selected = rng.choices(totals, k=len(totals))
        n, d = map(sum, zip(*selected, strict=True))
        if d:
            samples.append(n / d)
    return {
        "low": percentile(samples, 0.025),
        "high": percentile(samples, 0.975),
        "method": "task_cluster_percentile_bootstrap",
        "tasks": len(groups),
        "replicates": len(samples),
        "seed": seed,
    }


def rate(rows, numerator, denominator, **kwargs):
    n, d = sum(numerator(r) for r in rows), sum(denominator(r) for r in rows)
    return {
        "numerator": n,
        "denominator": d,
        "value": ratio(n, d),
        "ci95": cluster_ci(rows, numerator, denominator, **kwargs),
    }


def write_csv(path, rows):
    rows = list(rows)
    columns = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            )


def grouped_table(rows, key, **kwargs):
    groups = defaultdict(list)
    for row in rows:
        values = row[key] if key == "tool_types" else [row[key]]
        for value in values:
            groups[value].append(row)
    result = []
    for value, group in sorted(groups.items(), key=lambda item: str(item[0])):
        statistic = rate(group, lambda r: int(r["success"] is True), lambda r: int(r["started"]), **kwargs)
        result.append(
            {
                key: value,
                "n_rollouts": len(group),
                "n_started": sum(r["started"] for r in group),
                "n_valid_evaluations": sum(r["valid_evaluation"] for r in group),
                "confirmed_success_proportion": statistic["value"],
                "ci95": statistic["ci95"],
                "evaluated_success_rate": ratio(
                    sum(r["success"] is True for r in group), sum(r["valid_evaluation"] for r in group)
                ),
                "cost_usd": distribution([r["estimated_llm_cost_usd"] for r in group]),
                "wall_seconds": distribution([r["active_wall_seconds"] for r in group]),
                "generated_tokens": distribution([r["generated_tokens"] for r in group]),
            }
        )
    return result


def paired_comparisons(rows, *, seed=42, n_bootstrap=1000):
    tasks = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["task_id"] is not None and row["started"]:
            tasks[row["task_id"]][row["condition"]].append(row)
    conditions = sorted({r["condition"] for r in rows})
    records, summaries = [], []
    for a, b in itertools.combinations(conditions, 2):
        for metric in [
            "confirmed_success_proportion",
            "reward_missing_rate",
            "active_wall_seconds",
            "generated_tokens",
        ]:
            differences = []
            for task, groups in tasks.items():
                if a not in groups or b not in groups:
                    continue

                def value(group, metric=metric):
                    if metric == "confirmed_success_proportion":
                        values = [float(r["success"] is True) for r in group]
                    elif metric == "reward_missing_rate":
                        values = [float(r["reward_missing"]) for r in group]
                    else:
                        values = [r[metric] for r in group]
                    # No unmatched complete-case dropping within a task/condition cost aggregate.
                    return statistics.mean(values) if values and all(v is not None for v in values) else None

                av, bv = value(groups[a]), value(groups[b])
                delta = bv - av if av is not None and bv is not None else None
                records.append(
                    {
                        "task_id": task,
                        "condition_a": a,
                        "condition_b": b,
                        "metric": metric,
                        "attempts_a": len(groups[a]),
                        "attempts_b": len(groups[b]),
                        "a": av,
                        "b": bv,
                        "b_minus_a": delta,
                        "pair_unit": "task_means_over_all_started_repetitions",
                    }
                )
                if delta is not None:
                    differences.append(delta)
            rng = random.Random(seed)
            bootstrap = (
                [statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(n_bootstrap)]
                if len(differences) >= 2
                else []
            )
            summaries.append(
                {
                    "condition_a": a,
                    "condition_b": b,
                    "metric": metric,
                    "paired_tasks": len(differences),
                    "mean_b_minus_a": statistics.mean(differences) if differences else None,
                    "ci95_low": percentile(bootstrap, 0.025),
                    "ci95_high": percentile(bootstrap, 0.975),
                    "ci_method": "paired_task_cluster_bootstrap" if bootstrap else "unavailable_fewer_than_two_pairs",
                }
            )
    return records, summaries


def analyze(directories, output, *, seed=42, n_bootstrap=1000, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, integrity, inputs, run_wall, exported_calls = [], [], [], [], []
    seen_runs = set()
    for directory in map(Path, directories):
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["run_id"] in seen_runs:
            raise ValueError("Repeated run_id across inputs; do not count an exported run twice")
        seen_runs.add(metadata["run_id"])
        events, health = load_events(directory)
        clocks = {e["clock_id"] for e in events}
        run_wall.append(
            {
                "run_id": metadata["run_id"],
                "clock_domains": len(clocks),
                "observed_elapsed_seconds": (
                    max(e["monotonic_ns"] for e in events) - min(e["monotonic_ns"] for e in events)
                )
                / 1e9
                if events and len(clocks) == 1
                else None,
                "scope": "first to last captured event; lower bound if collection/job is ongoing",
            }
        )
        integrity.append({"run_id": metadata["run_id"], **health})
        inputs.append(
            {
                "directory": str(directory),
                "metadata": metadata,
                "events_sha256": hashlib.sha256((directory / "events.jsonl").read_bytes()).hexdigest(),
            }
        )
        groups = defaultdict(list)
        for event in events:
            if event["rollout_id"] is not None:
                groups[event["rollout_id"]].append(event)
        rows.extend(summarize_rollout(group, metadata) for group in groups.values())
        for rid, group in groups.items():
            for call in call_records(group):
                exported_calls.append(
                    {
                        "run_id": metadata["run_id"],
                        "rollout_id": rid,
                        "call_id": call["call_id"],
                        "call_index": call.get("call_index"),
                        "kind": call["kind"],
                        "tool_type": call.get("tool_type"),
                        "retry_group_id": call.get("retry_group_id"),
                        "retry_of": call.get("retry_of"),
                        "parent_call_id": call.get("parent_call_id"),
                        "finished": call["finished"],
                        "ok": call["ok"],
                        "error": call.get("error"),
                        "duration_seconds": call["duration"],
                        "started_utc": call["start"]["utc"],
                        "ended_utc": call["end"]["utc"] if call["end"] else None,
                        "usage": call.get("usage"),
                        "token_availability": {
                            key: value.get("status") for key, value in (call.get("tokens") or {}).items()
                        },
                    }
                )
    started = [r for r in rows if r["started"]]
    options = {"seed": seed, "n_bootstrap": n_bootstrap}
    success_count = sum(r["success"] is True for r in started)
    report = {
        "counts": {
            "scheduled": sum(r["scheduled"] for r in rows),
            "started": len(started),
            "ended": sum(r["ended"] for r in rows),
            "valid_evaluations": sum(r["valid_evaluation"] for r in started),
            "confirmed_successes": success_count,
            "unknown_successes": sum(r["success"] is None for r in started),
        },
        "rates": {},
        "termination_reasons": dict(Counter(r["termination_reason"] for r in started)),
        "integrity": integrity,
        "wall_time_per_run": run_wall,
        "cost_scope": "physical LLM attempts; provider-price estimates exclude infrastructure",
        "interpretation": (
            "Association only. Final length is censored by interruption; "
            "risk uses observed round entry. Missing is not zero."
        ),
    }
    report["token_field_coverage"] = {}
    wire_calls = [call for call in exported_calls if call["kind"] == "llm"]
    for key in ["prompt_token_ids", "response_token_ids", "sampling_log_probs", "weight_version"]:
        observed = sum(call["token_availability"].get(key) == "observed" for call in wire_calls)
        report["token_field_coverage"][key] = {
            "observed_calls": observed,
            "physical_llm_calls": len(wire_calls),
            "unavailable_calls": len(wire_calls) - observed,
        }
    for name, numerator, denominator in [
        ("valid_evaluation_rate", lambda r: int(r["valid_evaluation"]), lambda r: 1),
        ("confirmed_success_proportion", lambda r: int(r["success"] is True), lambda r: 1),
        ("evaluated_success_rate", lambda r: int(r["success"] is True), lambda r: int(r["valid_evaluation"])),
        ("reward_missing_rate", lambda r: int(r["reward_missing"]), lambda r: 1),
        ("tool_error_rate", lambda r: r["tool_errors"], lambda r: r["tool_calls_observed"]),
        ("retry_recovery_rate", lambda r: r["recovered_retry_episodes"], lambda r: r["recovery_eligible_episodes"]),
    ]:
        report["rates"][name] = rate(started, numerator, denominator, **options)
    for dimension in ["time", "turn", "token"]:
        statistic = rate(started, lambda r, d=dimension: int(r[d + "_budget_exhausted"]), lambda r: 1, **options)
        unknown = sum(not r["ended"] or not r["budget_observation_available"] for r in started)
        statistic.update(
            n_termination_unknown=unknown,
            confirmed_lower_bound=statistic["value"],
            possible_upper_bound=ratio(statistic["numerator"] + unknown, len(started)),
        )
        if unknown:
            statistic.update(value=None, ci95={"low": None, "high": None, "method": "unavailable_unknown_termination"})
        report["rates"][dimension + "_budget_exhaustion_rate"] = statistic
    numeric = [
        "llm_rounds",
        "llm_attempts",
        "tool_calls",
        "generated_tokens",
        "generated_tokens_observed_lower_bound",
        "queue_seconds",
        "active_wall_seconds",
        "scheduled_to_end_seconds",
        "estimated_llm_cost_usd",
        "retry_count",
        "retry_extra_call_seconds",
        "retry_extra_tokens",
        "retry_extra_estimated_usd",
        "retry_wait_seconds",
        "repeated_requests",
        "max_consecutive_identical_requests",
        "repeated_tool_commands",
        "first_fault_round",
        "first_fault_seconds",
    ]
    numeric += [
        kind + suffix
        for kind in ["llm", "tool", "evaluation"]
        for suffix in ["_call_seconds_sum", "_elapsed_seconds_union"]
    ]
    report["distributions"] = {key: distribution([r[key] for r in started]) for key in numeric}
    report["cost_per_confirmed_success"] = {}
    for key in ["generated_tokens", "active_wall_seconds", "estimated_llm_cost_usd"]:
        total = complete_sum([r[key] for r in started])
        report["cost_per_confirmed_success"][key] = {
            "all_attempts_total": total,
            "success_denominator": success_count,
            "value": total / success_count if total is not None and success_count else None,
            "status": "unavailable_no_successes"
            if not success_count
            else "unavailable_incomplete_cost"
            if total is None
            else "computed",
        }
    bins = []
    for low, high, label in [(0, 1, "0"), (1, 2, "1"), (2, 4, "2-3"), (4, 8, "4-7"), (8, math.inf, "8+")]:
        group = [r for r in started if r["llm_rounds"] is not None and low <= r["llm_rounds"] < high]
        stat = rate(group, lambda r: int(r["reward_missing"]), lambda r: 1, **options)
        bins.append(
            {
                "observed_llm_rounds": label,
                "n": len(group),
                "not_ended": sum(not r["ended"] for r in group),
                "reward_missing_n": stat["numerator"],
                "reward_missing_rate": stat["value"],
                "ci95": stat["ci95"],
            }
        )
    unknown_length = [r for r in started if r["llm_rounds"] is None]
    if unknown_length:
        bins.append(
            {
                "observed_llm_rounds": "unavailable",
                "n": len(unknown_length),
                "not_ended": sum(not r["ended"] for r in unknown_length),
                "reward_missing_n": sum(r["reward_missing"] for r in unknown_length),
                "reward_missing_rate": ratio(sum(r["reward_missing"] for r in unknown_length), len(unknown_length)),
                "ci95": {"low": None, "high": None, "method": "unavailable_length"},
            }
        )
    risk = []
    for t in sorted({t for r in started for t in r["entered_rounds"]}):
        group = [r for r in started if t in r["entered_rounds"]]
        n = sum(r["interruption_round"] == t for r in group)
        stat = rate(group, lambda r, t=t: int(r["interruption_round"] == t), lambda r: 1, **options)
        risk.append(
            {
                "round": t,
                "at_risk_entered_round": len(group),
                "interrupted_here": n,
                "external_interrupted_here": sum(
                    r["interruption_round"] == t and r["termination_category"] == "external_interruption" for r in group
                ),
                "right_censored_here": sum(not r["ended"] and max(r["entered_rounds"], default=0) == t for r in group),
                "observed_interruption_risk": stat["value"],
                "ci95": stat["ci95"],
            }
        )
    pairs, pair_summary = paired_comparisons(started, **options)
    adapter_rows = [r for r in rows if r["adapter_sample_count"] is not None]
    comparisons = sum(r["adapter_prefix_comparisons"] or 0 for r in adapter_rows)
    report["adapter"] = {
        "status": "observed" if adapter_rows else "unavailable_no_adapter_outputs",
        "sample_splits": distribution([r["adapter_sample_count"] for r in adapter_rows]),
        "complete_merge_ratio": ratio(
            sum(r["adapter_sample_count"] == 1 for r in adapter_rows if r["adapter_level"] == "trajectory"),
            sum(r["adapter_level"] == "trajectory" for r in adapter_rows),
        ),
        "prefix_merge_failure_rate": ratio(sum(r["adapter_prefix_failures"] or 0 for r in adapter_rows), comparisons),
        "prefix_comparisons": comparisons,
    }
    baselines = []
    groups = defaultdict(list)
    for row in adapter_rows:
        if row["task_id"] is None:
            continue  # A per-task baseline cannot be inferred from unlabelled rollouts.
        groups[(row["run_id"], row["task_id"], row["condition"])].append(row)
    for (run, task, condition), group in groups.items():
        kept = [r for r in group if r["adapter_sample_count"] > 0]
        raw = [r["adapter_effective_reward"] for r in kept]
        samples = [value for r in kept for value in (r["adapter_sample_rewards"] or [])]
        known = (
            kept and all(valid_reward(v) for v in raw) and all(r["adapter_sample_rewards"] is not None for r in kept)
        )
        a = statistics.mean(raw) if known else None
        b = statistics.mean(samples) if known and samples else None
        baselines.append(
            {
                "run_id": run,
                "task_id": task,
                "condition": condition,
                "population": "adapter_rollouts_with_at_least_one_row",
                "rollouts": len(kept),
                "samples": len(samples),
                "rollout_mean": a,
                "sample_mean": b,
                "sample_minus_rollout": b - a if a is not None and b is not None else None,
            }
        )
    write_csv(output / "rollouts.csv", rows)
    write_csv(output / "calls.csv", exported_calls)
    write_csv(output / "metrics.csv", [{"metric": key, **value} for key, value in report["rates"].items()])
    write_csv(
        output / "distributions.csv", [{"metric": key, **value} for key, value in report["distributions"].items()]
    )
    write_csv(output / "length_reward_missing.csv", bins)
    write_csv(output / "round_interruption_risk.csv", risk)
    write_csv(output / "paired_tasks.csv", pairs)
    write_csv(output / "paired_conditions.csv", pair_summary)
    write_csv(
        output / "attempts_by_task.csv",
        sorted(started, key=lambda r: (str(r["task_id"]), r["condition"], str(r["repetition"]))),
    )
    write_csv(
        output / "first_fault_and_recovery.csv",
        [
            {
                k: r[k]
                for k in [
                    "run_id",
                    "task_id",
                    "rollout_id",
                    "condition",
                    "first_fault_round",
                    "first_fault_seconds",
                    "first_fault_kind",
                    "retry_count",
                    "recovered_retry_episodes",
                    "recovery_eligible_episodes",
                ]
            }
            for r in started
        ],
    )
    write_csv(output / "adapter_reward_baselines.csv", baselines)
    associations = []
    for population, chosen in [
        ("all_started_observed_length", started),
        ("ended_only_selected_population", [r for r in started if r["ended"]]),
    ]:
        for x, y in [
            ("llm_rounds", "reward_missing"),
            ("llm_rounds", "success"),
            ("retry_count", "success"),
            ("tool_errors", "active_wall_seconds"),
            ("retry_count", "generated_tokens"),
        ]:
            pairs_known = [(float(r[x]), float(r[y])) for r in chosen if r[x] is not None and r[y] is not None]
            associations.append(
                {
                    "population": population,
                    "x": x,
                    "y": y,
                    "n_observed": len(pairs_known),
                    "n_unavailable": len(chosen) - len(pairs_known),
                    "spearman_rho": spearman(pairs_known),
                    "interpretation": (
                        "descriptive_association_only; interruption_censors_length; complete_case_selection"
                    ),
                }
            )
    write_csv(output / "associations.csv", associations)
    grouped = {}
    for key in ["task_id", "tool_types", "termination_reason", "fault_status", "retry_status"]:
        grouped[key] = grouped_table(started, key, **options)
        write_csv(output / f"by_{key}.csv", grouped[key])
    report["analysis_parameters"] = {
        **options,
        "paired_population": "task intersection; all started repetitions per condition",
        "inputs": inputs,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )
    if plots:
        from .plots import plot_all

        plot_all(
            output,
            rows=started,
            bins=bins,
            risk=risk,
            grouped=grouped,
            pairs=pair_summary,
            synthetic=all(item["metadata"].get("synthetic") for item in inputs),
        )
    return report
