# Copyright (c) Microsoft. All rights reserved.

"""Thin opt-in integration boundaries; sampling/reward/retry logic stays upstream."""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
from typing import Any, cast

from agentlightning.research.events import (
    active_recorder,
    current_rollout,
    identifier,
    observe,
    observed_call,
    token_fields,
    valid_reward,
)

llm_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("research_llm", default=None)


def gateway_observer(function):
    @functools.wraps(function)
    async def wrapped(rollout_id, attempt_id, mode, upstream_path, request):
        if not active_recorder():
            return await function(rollout_id, attempt_id, mode, upstream_path, request)
        raw = await request.body()  # Starlette caches it; downstream sees the same bytes.
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"raw_utf8": raw.decode("utf-8", errors="replace")}
        parent = request.headers.get("x-agl-research-call-id")
        group = parent or identifier()
        token = llm_context.set(
            {"rollout_id": rollout_id, "attempt_id": attempt_id, "parent_call_id": parent, "retry_group_id": group}
        )
        try:
            with observed_call(
                "gateway",
                rollout_id,
                request=body,
                retry_group_id=group,
                parent_call_id=parent,
                attempt_id=attempt_id,
                sdk_retry_index=request.headers.get("x-stainless-retry-count"),
            ) as call:
                response = await function(rollout_id, attempt_id, mode, upstream_path, request)
                if call:
                    call.result.update(
                        response={"http_status": response.status_code},
                        error={"type": "GatewayHTTPStatus", "status": response.status_code, "category": "unknown"}
                        if response.status_code >= 400
                        else None,
                    )
                return response
        finally:
            llm_context.reset(token)

    cast(Any, wrapped).__signature__ = inspect.signature(function, eval_str=True)
    return wrapped


def search_call(kind: str, **data):
    return observed_call(kind, current_rollout(), **data)


def reward_observation(
    value: Any,
    *,
    reason: str | None,
    rollout_id: str | None = None,
    success: bool | None = None,
    source: str = "agent_evaluator",
) -> None:
    # Only a task-specific evaluator may define success. Do not infer reward > 0 generically.
    if success is None and reason in {"em_match", "em_miss", "invalid_action"}:
        success = reason == "em_match"
    observe(
        "reward_observed",
        rollout_id or current_rollout(),
        value=value,
        reward_valid=valid_reward(value),
        success=success,
        reason=reason,
        source=source,
    )


def adapter_input(rollouts, level: str, global_steps: int) -> None:
    if not active_recorder():
        return
    for rollout in rollouts:
        observe(
            "adapter_input",
            rollout.rollout_id,
            phase="before_reward_fill_sort_merge_filter",
            global_steps=global_steps,
            level=level,
            completed_rollout=rollout,
        )


def adapter_output(rollouts, batch, level: str, global_steps: int, reward_fillna_value: float) -> None:
    if not active_recorder():
        return
    ids = list(batch.non_tensor_batch.get("rollout_id_list", []))
    for rollout in rollouts:
        count = sum(str(rid) == rollout.rollout_id for rid in ids)
        indices = [i for i, rid in enumerate(ids) if str(rid) == rollout.rollout_id]
        scores = batch.batch.get("token_level_scores")
        sample_rewards = scores[indices].sum(-1).detach().cpu().tolist() if scores is not None else None
        # Optional analysis of prefix merges uses exact observed IDs only, never tokenizer round trips.
        triplets = rollout.triplets or []
        comparisons = failures = 0
        context = None
        for triplet in triplets:
            prompt = triplet.prompt.get("token_ids")
            response = triplet.response.get("token_ids")
            if not isinstance(prompt, list) or not isinstance(response, list):
                comparisons = failures = None
                break
            if context is not None:
                comparisons += 1
                failures += prompt[: len(context)] != context
            context = prompt + response
        observe(
            "adapter_output",
            rollout.rollout_id,
            global_steps=global_steps,
            level=level,
            sample_count=count,
            prefix_comparisons=comparisons,
            prefix_failures=failures,
            final_reward=rollout.final_reward,
            reward_valid=valid_reward(rollout.final_reward),
            effective_rollout_reward=rollout.final_reward if rollout.final_reward is not None else reward_fillna_value,
            sample_rewards=sample_rewards,
        )


__all__ = [
    "adapter_input",
    "adapter_output",
    "gateway_observer",
    "llm_context",
    "reward_observation",
    "search_call",
    "token_fields",
]
