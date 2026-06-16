#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SIC feasibility study using the project environment."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from noma_rl.baselines import (
    equal_action,
    grid_search_optimize,
    heuristic_action,
    pso_optimize,
    random_action,
)
from noma_rl.config import ExperimentConfig
from noma_rl.env import NomaSecurityEnv


ALGORITHMS = [
    "Random",
    "Equal",
    "Heuristic",
    "Grid",
    "PSO",
    "Search_FullReward",
    "Search_NoSICPenalty",
    "Search_NoQoSPenalty",
    "Search_FixedJammer",
    "Search_NoJammer",
]


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
    steps: int,
    seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    return [sample_channel_gains(cfg, rng) for _ in range(steps)]


def reward_without_sic_penalty(env: NomaSecurityEnv, action: np.ndarray, gains: dict[str, float]) -> float:
    info = env.evaluate_action(action, gains=gains)
    cfg = env.cfg
    return float(info["rs_sum"] - cfg.lambda_qos * info["qos_pen"] - cfg.lambda_power * info["power_pen"])


def reward_without_qos_penalty(env: NomaSecurityEnv, action: np.ndarray, gains: dict[str, float]) -> float:
    info = env.evaluate_action(action, gains=gains)
    cfg = env.cfg
    return float(info["rs_sum"] - cfg.lambda_sic * info["sic_pen"] - cfg.lambda_power * info["power_pen"])


def grid_search_fixed_jammer(
    objective: Callable[[np.ndarray], float],
    jammer_share: float = 0.2,
    resolution: int = 21,
) -> np.ndarray:
    best = np.array([0.5 * (1 - jammer_share), 0.5 * (1 - jammer_share), jammer_share], dtype=np.float32)
    best_val = -1e18
    for a1_share in np.linspace(0.0, 1.0, resolution):
        a1 = (1.0 - jammer_share) * a1_share
        a2 = 1.0 - jammer_share - a1
        action = np.array([a1, a2, jammer_share], dtype=np.float32)
        val = objective(action)
        if val > best_val:
            best_val = val
            best = action
    return best


def grid_search_no_jammer(objective: Callable[[np.ndarray], float], resolution: int = 21) -> np.ndarray:
    best = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    best_val = -1e18
    for a1 in np.linspace(0.0, 1.0, resolution):
        action = np.array([a1, 1.0 - a1, 0.0], dtype=np.float32)
        val = objective(action)
        if val > best_val:
            best_val = val
            best = action
    return best


def select_action(
    name: str,
    env: NomaSecurityEnv,
    gains: dict[str, float],
    rng: np.random.Generator,
    cfg: ExperimentConfig,
) -> np.ndarray:
    reward_objective = lambda action: env.evaluate_action(action, gains=gains)["reward"]

    if name == "Random":
        return random_action(rng)
    if name == "Equal":
        return equal_action()
    if name == "Heuristic":
        return heuristic_action()
    if name == "Grid":
        return grid_search_optimize(reward_objective, resolution=cfg.grid_resolution)
    if name == "PSO":
        return pso_optimize(
            reward_objective,
            rng=rng,
            particles=cfg.pso_particles,
            iters=cfg.pso_iters,
        )
    if name == "Search_FullReward":
        return grid_search_optimize(reward_objective, resolution=cfg.grid_resolution)
    if name == "Search_NoSICPenalty":
        return grid_search_optimize(
            lambda action: reward_without_sic_penalty(env, action, gains),
            resolution=cfg.grid_resolution,
        )
    if name == "Search_NoQoSPenalty":
        return grid_search_optimize(
            lambda action: reward_without_qos_penalty(env, action, gains),
            resolution=cfg.grid_resolution,
        )
    if name == "Search_FixedJammer":
        return grid_search_fixed_jammer(reward_objective, resolution=cfg.grid_resolution)
    if name == "Search_NoJammer":
        return grid_search_no_jammer(reward_objective, resolution=cfg.grid_resolution)
    raise ValueError(f"Unknown algorithm: {name}")


def evaluate_algorithm(
    cfg: ExperimentConfig,
    channels: list[dict[str, float]],
    algorithm: str,
    seed: int,
) -> dict[str, float | str]:
    env = NomaSecurityEnv(cfg)
    rng = np.random.default_rng(seed)
    env.prev_action = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)

    rs_values, legit_values, eaves_values = [], [], []
    qos_hits, sic_hits, sic_margins, outage_hits, decision_times = [], [], [], [], []
    p1_values, p2_values, pj_values = [], [], []

    for gains in channels:
        env.current_gains = gains
        t0 = time.perf_counter()
        action = select_action(algorithm, env, gains, rng, cfg)
        decision_times.append((time.perf_counter() - t0) * 1e3)

        action = env._sanitize_action(action)
        info = env.evaluate_action(action, gains=gains)
        env.prev_action = action.copy()

        rs_values.append(info["rs_sum"])
        legit_values.append(info["r1"] + info["r2"])
        eaves_values.append(info["re1"] + info["re2"])
        qos_hits.append(info["qos_ok"])
        sic_hits.append(info["sic_ok"])
        sic_margins.append(info["gamma21"] - info["gamma1"])
        outage_hits.append(info["secrecy_outage"])
        p1_values.append(info["p1"])
        p2_values.append(info["p2"])
        pj_values.append(info["pj"])

    return {
        "algorithm": algorithm,
        "avg_secrecy_sum": float(np.mean(rs_values)),
        "avg_legit_rate_sum": float(np.mean(legit_values)),
        "avg_eaves_rate_sum": float(np.mean(eaves_values)),
        "qos_satisfaction_rate": float(np.mean(qos_hits)),
        "sic_feasible_rate": float(np.mean(sic_hits)),
        "avg_sic_margin": float(np.mean(sic_margins)),
        "secrecy_outage_prob": float(np.mean(outage_hits)),
        "avg_decision_time_ms": float(np.mean(decision_times)),
        "avg_p1": float(np.mean(p1_values)),
        "avg_p2": float(np.mean(p2_values)),
        "avg_pj": float(np.mean(pj_values)),
    }


def bar_plot(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 5.6))
    plt.bar(df["algorithm"], df[metric])
    plt.xlabel("Algorithm")
    plt.ylabel(ylabel)
    plt.title(f"Algorithm Comparison: {ylabel}")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def grouped_bar_plot(df: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(df["algorithm"]))
    width = 0.25
    metrics = ["avg_secrecy_sum", "qos_satisfaction_rate", "sic_feasible_rate"]
    labels = ["Secrecy Sum", "QoS Satisfaction", "SIC Feasible"]

    plt.figure(figsize=(11, 5.8))
    for i, metric in enumerate(metrics):
        plt.bar(x + (i - 1) * width, df[metric], width, label=labels[i])
    plt.xlabel("Algorithm")
    plt.ylabel("Value")
    plt.title("Security, QoS and SIC Feasibility Comparison")
    plt.xticks(x, df["algorithm"], rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_combined_metrics(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("sic_feasible_rate", "SIC Feasible Rate"),
        ("avg_sic_margin", "Average SIC Margin"),
        ("avg_secrecy_sum", "Average Secrecy Sum Rate"),
        ("qos_satisfaction_rate", "QoS Satisfaction Rate"),
        ("secrecy_outage_prob", "Secrecy Outage Probability"),
        ("avg_pj", "Average Jammer Power"),
    ]
    x = np.arange(len(df["algorithm"]))
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.2), sharex=True)

    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        ax.bar(x, df[metric])
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(df["algorithm"], rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("SIC Feasibility and Constraint Comparison", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_sic_feasible_rate(
    out_dir: Path,
    eval_steps: int,
    seed: int,
    r_min: float,
    grid_resolution: int,
    pso_particles: int,
    pso_iters: int,
) -> pd.DataFrame:
    cfg = ExperimentConfig(
        seed=seed,
        r1_min=r_min,
        r2_min=r_min,
        episode_steps=eval_steps,
        eval_episodes=1,
        grid_resolution=grid_resolution,
        pso_particles=pso_particles,
        pso_iters=pso_iters,
    )
    channels = generate_fixed_channels(cfg, eval_steps, seed=seed + 4040)
    rows = []
    progress = tqdm(ALGORITHMS, desc="SIC feasibility", unit="algorithm")
    for idx, algorithm in enumerate(progress):
        progress.set_postfix({"algorithm": algorithm})
        rows.append(evaluate_algorithm(cfg, channels, algorithm, seed=seed + idx + 20_000))
    df = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sic_feasible_results.csv", index=False)
    bar_plot(df, "sic_feasible_rate", "SIC Feasible Rate", out_dir / "sic_feasible_rate_comparison.png")
    bar_plot(df, "avg_sic_margin", "Average SIC Margin", out_dir / "sic_margin_comparison.png")
    bar_plot(df, "avg_secrecy_sum", "Average Secrecy Sum Rate", out_dir / "secrecy_sum_comparison.png")
    bar_plot(df, "qos_satisfaction_rate", "QoS Satisfaction Rate", out_dir / "qos_satisfaction_comparison.png")
    bar_plot(df, "secrecy_outage_prob", "Secrecy Outage Probability", out_dir / "secrecy_outage_comparison.png")
    grouped_bar_plot(df, out_dir / "summary_security_qos_sic.png")
    plot_combined_metrics(df, out_dir / "sic_feasible_rate_combined.png")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "sic_feasible_rate")
    parser.add_argument("--eval-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--r-min", type=float, default=1.0)
    parser.add_argument("--grid-resolution", type=int, default=21)
    parser.add_argument("--pso-particles", type=int, default=25)
    parser.add_argument("--pso-iters", type=int, default=35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_sic_feasible_rate(
        out_dir=args.out_dir,
        eval_steps=args.eval_steps,
        seed=args.seed,
        r_min=args.r_min,
        grid_resolution=args.grid_resolution,
        pso_particles=args.pso_particles,
        pso_iters=args.pso_iters,
    )
    print("\n===== SIC feasible rate experiment complete =====")
    print(df.to_string(index=False))
    print(f"\nResults saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
