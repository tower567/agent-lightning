# Copyright (c) Microsoft. All rights reserved.

"""Standalone scientific figures; never convert unavailable observations to zero."""

from __future__ import annotations

import importlib
from pathlib import Path


def plot_all(output: Path, *, rows, bins, risk, grouped, pairs, synthetic=False):
    try:
        matplotlib = importlib.import_module("matplotlib")
    except ImportError as exc:
        raise RuntimeError("Install agentlightning[research] for figures, or use --no-plots") from exc
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    def save(fig, name):
        if synthetic:
            fig.suptitle("Synthetic fixtures: recording/statistics validation only", fontsize=10)
        fig.tight_layout()
        fig.savefig(output / (name + ".png"), dpi=160)
        fig.savefig(output / (name + ".svg"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    valid = [(i, b) for i, b in enumerate(bins) if b["n"]]
    ax.bar([i for i, _ in valid], [b["reward_missing_rate"] for _, b in valid], color="#497ba7")
    for i, b in valid:
        ax.text(i, b["reward_missing_rate"] + 0.03, f"n={b['n']}", ha="center")
    ax.set(
        xticks=range(len(bins)),
        xticklabels=[b["observed_llm_rounds"] for b in bins],
        ylim=(0, 1.15),
        xlabel="Observed LLM rounds (interruption censors final length)",
        ylabel="Reward missing / started",
        title="Observed length and missing reward: descriptive association",
    )
    save(fig, "length_reward_missing")

    for key, data in grouped.items():
        fig, axes = plt.subplots(1, 2, figsize=(max(9, min(16, len(data) * 0.5)), 4))
        labels = [str(item[key])[:40] for item in data]
        for i, item in enumerate(data):
            if item["confirmed_success_proportion"] is not None:
                axes[0].bar(i, item["confirmed_success_proportion"], color="#448a72")
                axes[0].text(
                    i, item["confirmed_success_proportion"] + 0.03, f"n={item['n_started']}", ha="center", fontsize=8
                )
            cost = item["cost_usd"]["mean"]
            if cost is not None:
                axes[1].bar(i, cost, color="#c58c42")
            else:
                axes[1].text(i, 0, "N/A", ha="center", fontsize=8)
        for ax in axes:
            ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
        axes[0].set(title=f"Confirmed success / started by {key}", ylim=(0, 1.15))
        axes[1].set(title="Mean provider-price cost estimate (USD)")
        save(fig, "by_" + key)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    faults = [r for r in rows if r["first_fault_round"] is not None]
    positions = sorted({r["first_fault_round"] for r in faults})
    axes[0].bar(positions, [sum(r["first_fault_round"] == t for r in faults) for t in positions], color="#b75e5c")
    axes[0].set(xlabel="First observed fault round", ylabel="Rollouts", title="First fault position (unknown excluded)")
    eligible = sum(r["recovery_eligible_episodes"] for r in rows)
    recovered = sum(r["recovered_retry_episodes"] for r in rows)
    if eligible:
        axes[1].bar(["Recovered", "Not yet recovered"], [recovered, eligible - recovered], color=["#448a72", "#b75e5c"])
    else:
        axes[1].text(0.5, 0.5, "No linked retry episodes", ha="center")
    axes[1].set(title=f"Recovery after initial failure (episodes n={eligible})", ylabel="Episodes")
    save(fig, "first_fault_and_retry_recovery")

    fig, ax = plt.subplots(figsize=(9, max(3, len({r["task_id"] for r in rows}) * 0.35)))
    tasks = sorted({str(r["task_id"]) for r in rows})
    for task_index, task in enumerate(tasks):
        attempts = [r for r in rows if str(r["task_id"]) == task]
        for i, row in enumerate(attempts):
            color = "#448a72" if row["success"] is True else "#b75e5c" if row["success"] is False else "#999999"
            ax.scatter(i + 1, task_index, color=color)
            ax.annotate(
                row["condition"][:12], (i + 1, task_index), xytext=(2, 4), textcoords="offset points", fontsize=6
            )
    ax.set(
        yticks=range(len(tasks)),
        yticklabels=tasks,
        xlabel="Recorded attempt (not a matched-pair ID)",
        title="Repeated attempts: green=confirmed success, red=failure, gray=unknown",
    )
    save(fig, "attempts_by_task")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["confirmed_success_proportion", "active_wall_seconds"], strict=True):
        selected = [p for p in pairs if p["metric"] == metric and p["mean_b_minus_a"] is not None]
        for i, p in enumerate(selected):
            value = p["mean_b_minus_a"]
            ax.barh(i, value, color="#497ba7")
            if p["ci95_low"] is not None:
                ax.plot([p["ci95_low"], p["ci95_high"]], [i, i], color="black")
        ax.set_yticks(range(len(selected)), [p["condition_b"] + " - " + p["condition_a"] for p in selected])
        ax.axvline(0, color="gray", linewidth=0.7)
        ax.set_title(metric + "\npaired task means; task-cluster CI")
        if not selected:
            ax.text(
                0.5,
                0.5,
                "Unavailable: no matched task/condition observations",
                transform=ax.transAxes,
                ha="center",
                wrap=True,
            )
    save(fig, "paired_conditions")

    fig, ax = plt.subplots(figsize=(8, 4))
    if risk:
        ax.plot([r["round"] for r in risk], [r["observed_interruption_risk"] for r in risk], "o-", color="#b75e5c")
        ax.set_xticks([r["round"] for r in risk])
        ax.set_xlim(min(r["round"] for r in risk) - 0.5, max(r["round"] for r in risk) + 0.5)
        for r in risk:
            ax.annotate(
                f"{r['interrupted_here']}/{r['at_risk_entered_round']}; censored={r['right_censored_here']}",
                (r["round"], r["observed_interruption_risk"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    else:
        ax.text(0.5, 0.5, "Unavailable: no explicit round-entry observations", transform=ax.transAxes, ha="center")
    ax.set(
        xlabel="Round actually entered",
        ylabel="Interruptions here / entered and at risk",
        ylim=(-0.03, 1.15),
        title="Observed interruption risk; ongoing trajectories are right-censored",
    )
    save(fig, "round_interruption_risk")
