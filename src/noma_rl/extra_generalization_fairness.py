#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Channel generalization and user fairness experiments.

The experiments reuse the project environment instead of defining a separate
channel or reward model, so the generated metrics are directly comparable with
the main, QoS, and SIC experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from noma_rl.baselines import (
    equal_action,
    grid_search_optimize,
    heuristic_action,
    pso_optimize,
    random_action,
)
from noma_rl.config import ExperimentConfig
from noma_rl.ddpg import DDPGAgent
from noma_rl.env import NomaSecurityEnv
from noma_rl.sac import SACAgent
from noma_rl.td3 import TD3Agent

try:
    import torch
except ImportError:  # pragma: no cover - torch is part of the project deps.
    torch = None


class RLAgent(Protocol):
    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray: ...


SCENARIOS: dict[str, dict[str, float | str]] = {
    "original_channel": {
        "desc": "Original Channel",
        "le_scale": 1.0,
        "noise_scale": 1.0,
    },
    "strong_eavesdropper": {
        "desc": "Strong Eavesdropper",
        "le_scale": 3.0,
        "noise_scale": 1.0,
    },
    "high_noise": {
        "desc": "High Noise",
        "le_scale": 1.0,
        "noise_scale": 5.0,
    },
}

BASELINE_ALGORITHMS = ["Random", "Equal", "Heuristic", "Grid", "PSO"]
RL_ALGORITHMS = ["DDPG", "TD3", "SAC"]


def load_base_cfg(results_dir: Path) -> ExperimentConfig:
    config_path = results_dir / "config_used.json"
    if not config_path.exists():
        return ExperimentConfig()
    with config_path.open("r", encoding="utf-8") as f:
        return ExperimentConfig(**json.load(f))


def scenario_cfg(base_cfg: ExperimentConfig, scenario_name: str, eval_steps: int) -> ExperimentConfig:
    scenario = SCENARIOS[scenario_name]
    return replace(
        base_cfg,
        le=base_cfg.le * float(scenario["le_scale"]),
        noise_power=base_cfg.noise_power * float(scenario["noise_scale"]),
        episode_steps=eval_steps,
        eval_episodes=1,
    )


def sample_channel_gains(cfg: ExperimentConfig, rng: np.random.Generator) -> dict[str, float]:
    return {
        "h1": float(rng.exponential(scale=cfg.l1)),
        "h2": float(rng.exponential(scale=cfg.l2)),
        "g": float(rng.exponential(scale=cfg.le)),
        "hj1": float(rng.exponential(scale=cfg.lj1)),
        "hj2": float(rng.exponential(scale=cfg.lj2)),
        "gj": float(rng.exponential(scale=cfg.lje)),
    }


def generate_fixed_channels(
    cfg: ExperimentConfig,
    eval_steps: int,
    seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    return [sample_channel_gains(cfg, rng) for _ in range(eval_steps)]


def jain_fairness(values: list[float] | tuple[float, ...]) -> float:
    x = np.asarray(values, dtype=np.float64)
    denom = len(x) * float(np.sum(x**2))
    if denom <= 1e-12:
        return 0.0
    return float((float(np.sum(x)) ** 2) / denom)


def load_agent_weights(
    agent: RLAgent,
    results_dir: Path,
    prefix: str,
    device: str,
) -> RLAgent | None:
    if torch is None:
        return None

    actor_path = results_dir / f"{prefix}_actor.pt"
    if not actor_path.exists():
        return None

    agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    agent.actor.eval()

    critic_path = results_dir / f"{prefix}_critic.pt"
    critic1_path = results_dir / f"{prefix}_critic1.pt"
    critic2_path = results_dir / f"{prefix}_critic2.pt"
    if hasattr(agent, "critic") and critic_path.exists():
        agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
        agent.critic.eval()
    if hasattr(agent, "critic1") and critic1_path.exists():
        agent.critic1.load_state_dict(torch.load(critic1_path, map_location=device))
        agent.critic1.eval()
    if hasattr(agent, "critic2") and critic2_path.exists():
        agent.critic2.load_state_dict(torch.load(critic2_path, map_location=device))
        agent.critic2.eval()

    return agent


def make_loaded_agents(
    cfg: ExperimentConfig,
    results_dir: Path,
    device: str,
) -> dict[str, RLAgent]:
    loaded: dict[str, RLAgent] = {}
    candidates: dict[str, tuple[RLAgent, str]] = {
        "DDPG": (DDPGAgent(cfg, device=device), "ddpg"),
        "TD3": (TD3Agent(cfg, device=device), "td3"),
        "SAC": (SACAgent(cfg, device=device), "sac"),
    }
    for name, (agent, prefix) in candidates.items():
        maybe_agent = load_agent_weights(agent, results_dir, prefix, device)
        if maybe_agent is not None:
            loaded[name] = maybe_agent
    return loaded


def select_action(
    name: str,
    env: NomaSecurityEnv,
    gains: dict[str, float],
    rng: np.random.Generator,
    agents: dict[str, RLAgent],
    grid_resolution: int,
    pso_particles: int,
    pso_iters: int,
) -> np.ndarray:
    if name in agents:
        return agents[name].select_action(env._build_state(), noise_std=0.0)
    if name == "Random":
        return random_action(rng)
    if name == "Equal":
        return equal_action()
    if name == "Heuristic":
        return heuristic_action()
    if name == "Grid":
        return grid_search_optimize(
            lambda action: env.evaluate_action(action, gains=gains)["reward"],
            resolution=grid_resolution,
        )
    if name == "PSO":
        return pso_optimize(
            lambda action: env.evaluate_action(action, gains=gains)["reward"],
            rng=rng,
            particles=pso_particles,
            iters=pso_iters,
        )
    raise ValueError(f"Unknown algorithm: {name}")


def evaluate_algorithm(
    cfg: ExperimentConfig,
    channels: list[dict[str, float]],
    scenario_name: str,
    algorithm: str,
    agents: dict[str, RLAgent],
    seed: int,
    grid_resolution: int,
    pso_particles: int,
    pso_iters: int,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    env = NomaSecurityEnv(cfg)
    rng = np.random.default_rng(seed)
    env.prev_action = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)

    rows = []
    for sample_idx, gains in enumerate(channels, start=1):
        env.current_gains = gains
        t0 = time.perf_counter()
        action = select_action(
            algorithm,
            env,
            gains,
            rng,
            agents,
            grid_resolution,
            pso_particles,
            pso_iters,
        )
        decision_time_ms = (time.perf_counter() - t0) * 1e3

        action = env._sanitize_action(action)
        info = env.evaluate_action(action, gains=gains)
        env.prev_action = action.copy()

        row = {
            "scenario": scenario_name,
            "scenario_desc": SCENARIOS[scenario_name]["desc"],
            "algorithm": algorithm,
            "sample_idx": sample_idx,
            "r1": info["r1"],
            "r2": info["r2"],
            "rs1": info["rs1"],
            "rs2": info["rs2"],
            "rs_sum": info["rs_sum"],
            "re1": info["re1"],
            "re2": info["re2"],
            "user1_qos_ok": float(info["r1"] >= cfg.r1_min),
            "user2_qos_ok": float(info["r2"] >= cfg.r2_min),
            "qos_ok": info["qos_ok"],
            "sic_ok": info["sic_ok"],
            "secrecy_outage": info["secrecy_outage"],
            "jain_rate": jain_fairness([info["r1"], info["r2"]]),
            "jain_secrecy": jain_fairness([info["rs1"], info["rs2"]]),
            "p1": info["p1"],
            "p2": info["p2"],
            "pj": info["pj"],
            "decision_time_ms": decision_time_ms,
        }
        rows.append(row)

    detail_df = pd.DataFrame(rows)
    summary = {
        "scenario": scenario_name,
        "scenario_desc": SCENARIOS[scenario_name]["desc"],
        "algorithm": algorithm,
        "avg_secrecy_sum": float(detail_df["rs_sum"].mean()),
        "avg_user1_rate": float(detail_df["r1"].mean()),
        "avg_user2_rate": float(detail_df["r2"].mean()),
        "avg_user1_secrecy": float(detail_df["rs1"].mean()),
        "avg_user2_secrecy": float(detail_df["rs2"].mean()),
        "user1_qos_rate": float(detail_df["user1_qos_ok"].mean()),
        "user2_qos_rate": float(detail_df["user2_qos_ok"].mean()),
        "qos_satisfaction_rate": float(detail_df["qos_ok"].mean()),
        "sic_feasible_rate": float(detail_df["sic_ok"].mean()),
        "secrecy_outage_prob": float(detail_df["secrecy_outage"].mean()),
        "jain_fairness_index": float(detail_df["jain_rate"].mean()),
        "jain_secrecy_index": float(detail_df["jain_secrecy"].mean()),
        "avg_p1": float(detail_df["p1"].mean()),
        "avg_p2": float(detail_df["p2"].mean()),
        "avg_pj": float(detail_df["pj"].mean()),
        "avg_decision_time_ms": float(detail_df["decision_time_ms"].mean()),
    }
    return summary, detail_df


def plot_grouped_bars(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    scenarios = list(SCENARIOS.keys())
    algorithms = list(df["algorithm"].drop_duplicates())
    x = np.arange(len(scenarios))
    width = min(0.12, 0.76 / max(1, len(algorithms)))

    plt.figure(figsize=(12, 5.8))
    for idx, algorithm in enumerate(algorithms):
        values = []
        for scenario_name in scenarios:
            sub = df[(df["scenario"] == scenario_name) & (df["algorithm"] == algorithm)]
            values.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
        offset = (idx - (len(algorithms) - 1) / 2) * width
        plt.bar(x + offset, values, width, label=algorithm)

    plt.xticks(x, [str(SCENARIOS[name]["desc"]) for name in scenarios])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_fairness_user_metrics(df: pd.DataFrame, out_path: Path) -> None:
    original = df[df["scenario"] == "original_channel"].copy()
    x = np.arange(len(original))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].bar(x - width / 2, original["avg_user1_rate"], width, label="User 1")
    axes[0].bar(x + width / 2, original["avg_user2_rate"], width, label="User 2")
    axes[0].set_ylabel("Average Legitimate Rate")
    axes[0].set_title("User Rate")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(original["algorithm"], rotation=30, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].bar(x - width / 2, original["user1_qos_rate"], width, label="User 1")
    axes[1].bar(x + width / 2, original["user2_qos_rate"], width, label="User 2")
    axes[1].set_ylabel("QoS Satisfaction Rate")
    axes[1].set_title("Per-User QoS Satisfaction")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(original["algorithm"], rotation=30, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend()

    fig.suptitle("Fairness Experiment on Original Channel")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_combined_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("avg_secrecy_sum", "Average Secrecy Sum Rate"),
        ("qos_satisfaction_rate", "QoS Satisfaction Rate"),
        ("secrecy_outage_prob", "Secrecy Outage Probability"),
        ("jain_fairness_index", "Jain Fairness Index"),
    ]
    scenarios = list(SCENARIOS.keys())
    algorithms = list(summary_df["algorithm"].drop_duplicates())
    x = np.arange(len(scenarios))
    width = min(0.12, 0.76 / max(1, len(algorithms)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for idx, algorithm in enumerate(algorithms):
            values = []
            for scenario_name in scenarios:
                sub = summary_df[
                    (summary_df["scenario"] == scenario_name) & (summary_df["algorithm"] == algorithm)
                ]
                values.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
            offset = (idx - (len(algorithms) - 1) / 2) * width
            ax.bar(x + offset, values, width, label=algorithm)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(SCENARIOS[name]["desc"]) for name in scenarios])
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(algorithms)), frameon=False)
    fig.suptitle("Channel Generalization and Fairness Summary", fontsize=15)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_outputs(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    generalization_cols = [
        "scenario",
        "scenario_desc",
        "algorithm",
        "avg_secrecy_sum",
        "qos_satisfaction_rate",
        "sic_feasible_rate",
        "secrecy_outage_prob",
        "avg_pj",
        "avg_decision_time_ms",
    ]
    fairness_cols = [
        "scenario",
        "scenario_desc",
        "algorithm",
        "avg_user1_rate",
        "avg_user2_rate",
        "user1_qos_rate",
        "user2_qos_rate",
        "jain_fairness_index",
    ]

    summary_df[generalization_cols].to_csv(out_dir / "results_generalization.csv", index=False)
    summary_df[fairness_cols].to_csv(out_dir / "results_fairness.csv", index=False)
    summary_df.to_csv(out_dir / "results_summary_all_metrics.csv", index=False)
    detail_df.to_csv(out_dir / "results_detail_all_samples.csv", index=False)

    plot_grouped_bars(
        summary_df,
        "avg_secrecy_sum",
        "Average Secrecy Sum Rate",
        "Channel Generalization: Secrecy Sum Rate",
        out_dir / "fig_generalization_secrecy.png",
    )
    plot_grouped_bars(
        summary_df,
        "qos_satisfaction_rate",
        "QoS Satisfaction Rate",
        "Channel Generalization: QoS Satisfaction",
        out_dir / "fig_generalization_qos.png",
    )
    plot_grouped_bars(
        summary_df,
        "secrecy_outage_prob",
        "Secrecy Outage Probability",
        "Channel Generalization: Secrecy Outage",
        out_dir / "fig_generalization_outage.png",
    )
    plot_grouped_bars(
        summary_df,
        "jain_fairness_index",
        "Jain Fairness Index",
        "Fairness Experiment: Jain Fairness",
        out_dir / "fig_fairness_jain.png",
    )
    plot_fairness_user_metrics(summary_df, out_dir / "fig_fairness_user_rate_qos.png")
    plot_combined_summary(summary_df, out_dir / "fig_generalization_fairness_combined.png")


def run_experiments(
    base_cfg: ExperimentConfig,
    out_dir: Path,
    results_dir: Path,
    eval_steps: int,
    seed: int,
    grid_resolution: int,
    pso_particles: int,
    pso_iters: int,
    device: str,
    include_rl: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    start = time.perf_counter()
    agents = make_loaded_agents(base_cfg, results_dir, device) if include_rl else {}
    algorithms = BASELINE_ALGORITHMS + [name for name in RL_ALGORITHMS if name in agents]

    if include_rl:
        missing = [name for name in RL_ALGORITHMS if name not in agents]
        if missing:
            print(f"Warning: skipped missing RL weights: {', '.join(missing)}")

    summaries = []
    details = []
    for scenario_idx, scenario_name in enumerate(SCENARIOS):
        cfg = scenario_cfg(base_cfg, scenario_name, eval_steps)
        channels = generate_fixed_channels(cfg, eval_steps, seed + 1000 * (scenario_idx + 1))
        print(f"\n===== {scenario_name}: {SCENARIOS[scenario_name]['desc']} =====")

        for alg_idx, algorithm in enumerate(algorithms):
            print(f"Evaluating {algorithm} ...")
            summary, detail_df = evaluate_algorithm(
                cfg=cfg,
                channels=channels,
                scenario_name=scenario_name,
                algorithm=algorithm,
                agents=agents,
                seed=seed + 10_000 + alg_idx,
                grid_resolution=grid_resolution,
                pso_particles=pso_particles,
                pso_iters=pso_iters,
            )
            summaries.append(summary)
            details.append(detail_df)
            print(
                f"{algorithm:10s} | secrecy={summary['avg_secrecy_sum']:.4f} | "
                f"QoS={summary['qos_satisfaction_rate']:.4f} | "
                f"Jain={summary['jain_fairness_index']:.4f} | "
                f"time={summary['avg_decision_time_ms']:.3f} ms"
            )

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(details, ignore_index=True)
    save_outputs(summary_df, detail_df, out_dir)
    elapsed = time.perf_counter() - start
    return summary_df, detail_df, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "generalization_fairness")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs" / "main")
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--grid-resolution", type=int, default=21)
    parser.add_argument("--pso-particles", type=int, default=20)
    parser.add_argument("--pso-iters", type=int, default=25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-rl", action="store_true", help="Only run Random/Equal/Heuristic/Grid/PSO.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_base_cfg(args.results_dir)
    summary_df, _, elapsed = run_experiments(
        base_cfg=base_cfg,
        out_dir=args.out_dir,
        results_dir=args.results_dir,
        eval_steps=args.eval_steps,
        seed=args.seed,
        grid_resolution=args.grid_resolution,
        pso_particles=args.pso_particles,
        pso_iters=args.pso_iters,
        device=args.device,
        include_rl=not args.no_rl,
    )

    print("\n===== Extra experiments complete =====")
    print(summary_df.to_string(index=False))
    print(f"\nElapsed: {elapsed:.1f} s")
    print(f"Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
