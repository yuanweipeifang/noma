#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QoS threshold sensitivity study using the project environment."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

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


DEFAULT_QOS_THRESHOLDS = [0.5, 1.0, 1.5, 2.0]
ALGORITHMS = ["Random", "Equal", "Heuristic", "Grid", "PSO"]


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


def select_action(
    name: str,
    env: NomaSecurityEnv,
    gains: dict[str, float],
    rng: np.random.Generator,
    cfg: ExperimentConfig,
) -> np.ndarray:
    if name == "Random":
        return random_action(rng)
    if name == "Equal":
        return equal_action()
    if name == "Heuristic":
        return heuristic_action()
    if name == "Grid":
        return grid_search_optimize(
            lambda action: env.evaluate_action(action, gains=gains)["reward"],
            resolution=cfg.grid_resolution,
        )
    if name == "PSO":
        return pso_optimize(
            lambda action: env.evaluate_action(action, gains=gains)["reward"],
            rng=rng,
            particles=cfg.pso_particles,
            iters=cfg.pso_iters,
        )
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
    qos_hits, sic_hits, outage_hits, decision_times = [], [], [], []

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
        outage_hits.append(info["secrecy_outage"])

    return {
        "algorithm": algorithm,
        "avg_secrecy_sum": float(np.mean(rs_values)),
        "avg_legit_rate_sum": float(np.mean(legit_values)),
        "avg_eaves_rate_sum": float(np.mean(eaves_values)),
        "qos_satisfaction_rate": float(np.mean(qos_hits)),
        "sic_feasible_rate": float(np.mean(sic_hits)),
        "secrecy_outage_prob": float(np.mean(outage_hits)),
        "avg_decision_time_ms": float(np.mean(decision_times)),
    }


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(8.5, 5.2))
    for alg in ALGORITHMS:
        sub = df[df["algorithm"] == alg].sort_values("qos_threshold")
        if len(sub) == 0:
            continue
        plt.plot(sub["qos_threshold"], sub[metric], marker="o", linewidth=1.8, label=alg)
    plt.xlabel("QoS Threshold R_min (bit/s/Hz)")
    plt.ylabel(ylabel)
    plt.title(f"QoS Sensitivity: {ylabel}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_combined_metrics(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("avg_secrecy_sum", "Average Secrecy Sum Rate"),
        ("qos_satisfaction_rate", "QoS Satisfaction Rate"),
        ("sic_feasible_rate", "SIC Feasible Rate"),
        ("secrecy_outage_prob", "Secrecy Outage Probability"),
        ("avg_decision_time_ms", "Average Decision Time (ms)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8), sharex=True)
    axes = axes.ravel()

    for ax, (metric, ylabel) in zip(axes, metrics):
        for alg in ALGORITHMS:
            sub = df[df["algorithm"] == alg].sort_values("qos_threshold")
            if len(sub) == 0:
                continue
            ax.plot(sub["qos_threshold"], sub[metric], marker="o", linewidth=1.8, label=alg)
        ax.set_title(ylabel)
        ax.set_xlabel("QoS Threshold R_min (bit/s/Hz)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ALGORITHMS), frameon=False)
    fig.suptitle("QoS Threshold Sensitivity", fontsize=15)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_qos_sensitivity(
    out_dir: Path,
    thresholds: list[float],
    eval_steps: int,
    seed: int,
    grid_resolution: int,
    pso_particles: int,
    pso_iters: int,
) -> pd.DataFrame:
    rows = []
    progress = tqdm(
        total=len(thresholds) * len(ALGORITHMS),
        desc="QoS sensitivity",
        unit="case",
    )
    for r_min in thresholds:
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
        channels = generate_fixed_channels(cfg, eval_steps, seed=seed + 2026)
        for alg_idx, algorithm in enumerate(ALGORITHMS):
            progress.set_postfix({"R_min": r_min, "algorithm": algorithm})
            row = evaluate_algorithm(cfg, channels, algorithm, seed=seed + alg_idx + 10_000)
            rows.append({"qos_threshold": r_min, **row})
            progress.update(1)
    progress.close()

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "qos_sensitivity_results.csv", index=False)

    plot_metric(df, "avg_secrecy_sum", "Average Secrecy Sum Rate", out_dir / "qos_vs_secrecy_sum.png")
    plot_metric(df, "qos_satisfaction_rate", "QoS Satisfaction Rate", out_dir / "qos_vs_qos_satisfaction.png")
    plot_metric(df, "sic_feasible_rate", "SIC Feasible Rate", out_dir / "qos_vs_sic_feasible_rate.png")
    plot_metric(df, "secrecy_outage_prob", "Secrecy Outage Probability", out_dir / "qos_vs_secrecy_outage.png")
    plot_metric(df, "avg_decision_time_ms", "Average Decision Time (ms)", out_dir / "qos_vs_decision_time.png")
    plot_combined_metrics(df, out_dir / "qos_sensitivity_combined.png")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "qos_sensitivity")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_QOS_THRESHOLDS)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-resolution", type=int, default=21)
    parser.add_argument("--pso-particles", type=int, default=25)
    parser.add_argument("--pso-iters", type=int, default=35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_qos_sensitivity(
        out_dir=args.out_dir,
        thresholds=args.thresholds,
        eval_steps=args.eval_steps,
        seed=args.seed,
        grid_resolution=args.grid_resolution,
        pso_particles=args.pso_particles,
        pso_iters=args.pso_iters,
    )
    print("\n===== QoS sensitivity experiment complete =====")
    print(df.to_string(index=False))
    print(f"\nResults saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
