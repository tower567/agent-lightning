# Copyright (c) Microsoft. All rights reserved.

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("verl")

import torch

from agentlightning.research.analysis import analyze
from agentlightning.research.events import Recorder, initialize
from agentlightning.verl.agl_rollout_manager import CompletedRollout, Triplet
from agentlightning.verl.rollout_adapter import RolloutAdapter


def test_actual_adapter_raw_reward_masks_and_sample_weighting(tmp_path, monkeypatch):
    initialize(
        tmp_path,
        {
            "model_version": None,
            "dataset_version": "adapter-fixture",
            "sampling_parameters": {},
            "execution_budget": {},
            "seed": 42,
            "fault_config": {"synthetic": True},
        },
    )
    monkeypatch.setenv("AGL_RESEARCH_RUN_DIR", str(tmp_path))
    recorder = Recorder(tmp_path)

    def triplet(prompt, response):
        return Triplet(
            prompt={"token_ids": prompt}, response={"token_ids": response, "log_probs": [-0.1] * len(response)}
        )

    rollouts = []
    for i, (reward, second_prompt) in enumerate([(1.0, [10, 20, 30]), (None, [90])]):
        rid = f"r{i}"
        recorder.emit("rollout_scheduled", rid, task_id="same-task", repetition=i, condition="adapter-test")
        rollouts.append(
            CompletedRollout(
                rollout_id=rid,
                data_id="same-task",
                step=0,
                sample_idx_in_step=i,
                enqueue_time=0.0,
                final_reward=reward,
                triplets=[triplet([10], [20]), triplet(second_prompt, [40])],
            )
        )
    adapter = RolloutAdapter(
        max_prompt_length=8,
        max_response_length=8,
        device=torch.device("cpu"),
        pad_token_id=0,
        reward_fillna_value=0.0,
        trace_aggregator_level="trajectory",
    )
    batch, _ = adapter.get_train_data_batch(rollouts)
    assert batch.batch["response_mask"][0, :3].tolist() == [1, 0, 1]
    raw = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    inputs = [e for e in raw if e["event_type"] == "adapter_input"]
    assert inputs[1]["data"]["completed_rollout"]["final_reward"] is None
    outputs = [e["data"] for e in raw if e["event_type"] == "adapter_output"]
    assert [o["sample_count"] for o in outputs] == [1, 2]
    assert [o["prefix_failures"] for o in outputs] == [0, 1]
    assert outputs[1]["sample_rewards"] == [0.0, 0.0]
    report = analyze([tmp_path], tmp_path / "analysis", plots=False, n_bootstrap=10)
    assert report["adapter"]["complete_merge_ratio"] == 0.5
    import csv

    with (tmp_path / "analysis/adapter_reward_baselines.csv").open(encoding="utf-8-sig") as stream:
        baseline = next(csv.DictReader(stream))
    assert float(baseline["rollout_mean"]) == 0.5
    assert float(baseline["sample_mean"]) == pytest.approx(1 / 3)
