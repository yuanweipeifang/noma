#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Protocol

import numpy as np
import pandas as pd
import torch
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

from noma_rl.baselines import (  # noqa: E402
    ddpg_without_jammer,
    equal_action,
    grid_search_optimize,
    heuristic_action,
    pso_optimize,
    random_action,
)
from noma_rl.config import ExperimentConfig  # noqa: E402
from noma_rl.ddpg import DDPGAgent  # noqa: E402
from noma_rl.env import NomaSecurityEnv  # noqa: E402
from noma_rl.sac import SACAgent  # noqa: E402
from noma_rl.td3 import TD3Agent  # noqa: E402


class RLAgent(Protocol):
    replay: object

    def select_action(self, state: np.ndarray, noise_std: float = 0.05) -> np.ndarray: ...

    def train_step(self) -> tuple[float, float]: ...


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def moving_average(x: List[float], window: int = 50) -> np.ndarray:
    if len(x) < window:
        return np.array(x, dtype=np.float32)
    kernel = np.ones(window) / window
    arr = np.array(x, dtype=np.float32)
    return np.convolve(arr, kernel, mode="valid")


def train_agent(
    name: str, agent_cls: type[RLAgent], cfg: ExperimentConfig, device: str, out_dir: Path
):
    env = NomaSecurityEnv(cfg)
    agent = agent_cls(cfg, device=device)

    episode_rewards, episode_secrecy, episode_qos = [], [], []
    actor_losses, critic_losses = [], []

    for ep in trange(cfg.train_episodes, desc=f"Training {name}"):
        state = env.reset()
        ep_reward = 0.0
        ep_rs = 0.0
        ep_qos_hits = 0.0
        noise_std = max(0.01, 0.2 * (1.0 - ep / max(1, cfg.train_episodes)))

        for _ in range(cfg.episode_steps):
            action = agent.select_action(state, noise_std=noise_std)
            next_state, reward, done, info = env.step(action)
            agent.replay.add(
                state, action, np.array([reward], dtype=np.float32), next_state, float(done)
            )
            al, cl = agent.train_step()
            if al != 0.0 or cl != 0.0:
                actor_losses.append(al)
                critic_losses.append(cl)
            state = next_state
            ep_reward += reward
            ep_rs += info["rs_sum"]
            ep_qos_hits += info["qos_ok"]
            if done:
                break

        episode_rewards.append(ep_reward / cfg.episode_steps)
        episode_secrecy.append(ep_rs / cfg.episode_steps)
        episode_qos.append(ep_qos_hits / cfg.episode_steps)

    prefix = name.lower()
    if hasattr(agent, "actor"):
        torch.save(agent.actor.state_dict(), out_dir / f"{prefix}_actor.pt")
    if hasattr(agent, "critic"):
        torch.save(agent.critic.state_dict(), out_dir / f"{prefix}_critic.pt")
    if hasattr(agent, "critic1"):
        torch.save(agent.critic1.state_dict(), out_dir / f"{prefix}_critic1.pt")
    if hasattr(agent, "critic2"):
        torch.save(agent.critic2.state_dict(), out_dir / f"{prefix}_critic2.pt")

    train_df = pd.DataFrame(
        {
            "episode": np.arange(1, cfg.train_episodes + 1),
            "avg_reward": episode_rewards,
            "avg_secrecy_sum": episode_secrecy,
            "qos_satisfaction_rate": episode_qos,
        }
    )
    train_df.to_csv(out_dir / f"{prefix}_training_log.csv", index=False)

    return agent, train_df


def train_ddpg(cfg: ExperimentConfig, device: str, out_dir: Path):
    agent, train_df = train_agent("DDPG", DDPGAgent, cfg, device, out_dir)
    train_df.to_csv(out_dir / "training_log.csv", index=False)
    return agent, train_df


def evaluate_algorithm(name: str, cfg: ExperimentConfig, agents: Dict[str, RLAgent]):
    env = NomaSecurityEnv(cfg)
    rng = np.random.default_rng(cfg.seed + 123)

    rs_values, legit_values, eaves_values = [], [], []
    qos_hits, outage_hits, decision_times = [], [], []

    for _ in trange(cfg.eval_episodes, desc=f"Evaluating {name}"):
        _ = env.reset()
        for _ in range(cfg.episode_steps):
            t0 = time.perf_counter()

            if name in agents:
                action = agents[name].select_action(env._build_state(), noise_std=0.0)
            elif name == "Random":
                action = random_action(rng)
            elif name == "Equal":
                action = equal_action()
            elif name == "Heuristic":
                action = heuristic_action()
            elif name == "PSO":
                gains = dict(env.current_gains)

                def objective(a):
                    return env.evaluate_action(a, gains=gains)["reward"]

                action = pso_optimize(
                    objective,
                    rng=rng,
                    particles=cfg.pso_particles,
                    iters=cfg.pso_iters,
                )
            elif name == "Grid":
                gains = dict(env.current_gains)

                def objective(a):
                    return env.evaluate_action(a, gains=gains)["reward"]

                action = grid_search_optimize(objective, resolution=cfg.grid_resolution)
            elif name == "DDPG_NoJammer":
                raw = agents["DDPG"].select_action(env._build_state(), noise_std=0.0)
                action = ddpg_without_jammer(raw)
            else:
                raise ValueError(f"Unknown algorithm: {name}")

            decision_times.append((time.perf_counter() - t0) * 1e3)
            _, _, done, info = env.step(action)

            rs_values.append(info["rs_sum"])
            legit_values.append(info["r1"] + info["r2"])
            eaves_values.append(info["re1"] + info["re2"])
            qos_hits.append(info["qos_ok"])
            outage_hits.append(info["secrecy_outage"])

            if done:
                break

    return {
        "algorithm": name,
        "avg_secrecy_sum": float(np.mean(rs_values)),
        "avg_legit_rate_sum": float(np.mean(legit_values)),
        "avg_eaves_rate_sum": float(np.mean(eaves_values)),
        "qos_satisfaction_rate": float(np.mean(qos_hits)),
        "secrecy_outage_prob": float(np.mean(outage_hits)),
        "avg_decision_time_ms": float(np.mean(decision_times)),
    }


def plot_training_curve(
    train_df: pd.DataFrame,
    out_dir: Path,
    filename: str,
    title: str,
    metric: str = "avg_reward",
    ylabel: str = "Reward",
):
    plt.figure(figsize=(8, 4))
    if "algorithm" in train_df.columns:
        for name, group in train_df.groupby("algorithm", sort=False):
            plt.plot(group["episode"], group[metric], alpha=0.25, linewidth=0.8)
            ma = moving_average(group[metric].tolist(), window=50)
            if len(ma) > 0:
                x = (
                    np.arange(50, 50 + len(ma))
                    if len(group) >= 50
                    else np.arange(1, len(ma) + 1)
                )
                plt.plot(x, ma, linewidth=2.0, label=f"{name} MA(50)")
    else:
        plt.plot(train_df["episode"], train_df[metric], alpha=0.35, label="Episode value")
        ma = moving_average(train_df[metric].tolist(), window=50)
        if len(ma) > 0:
            x = (
                np.arange(50, 50 + len(ma))
                if len(train_df) >= 50
                else np.arange(1, len(ma) + 1)
            )
            plt.plot(x, ma, linewidth=2.0, label="MA(50)")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=160)
    plt.close()


def plot_metric_bar(
    metrics_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    out_dir: Path,
    ascending: bool = False,
):
    plt.figure(figsize=(9, 4.5))
    order = metrics_df.sort_values(metric, ascending=ascending)
    plt.bar(order["algorithm"], order[metric])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=160)
    plt.close()


def plot_results(train_df: pd.DataFrame, metrics_df: pd.DataFrame, out_dir: Path):
    plot_training_curve(
        train_df,
        out_dir,
        filename="fig_training_convergence.png",
        title="RL Training Reward Convergence",
        metric="avg_reward",
        ylabel="Reward",
    )
    plot_training_curve(
        train_df,
        out_dir,
        filename="fig_training_secrecy_convergence.png",
        title="RL Training Secrecy Sum Rate Convergence",
        metric="avg_secrecy_sum",
        ylabel="Average Secrecy Sum Rate (bit/s/Hz)",
    )
    plot_training_curve(
        train_df,
        out_dir,
        filename="fig_training_qos_convergence.png",
        title="RL Training QoS Satisfaction Convergence",
        metric="qos_satisfaction_rate",
        ylabel="QoS Satisfaction Rate",
    )

    plot_metric_bar(
        metrics_df,
        metric="avg_secrecy_sum",
        ylabel="Average Secrecy Sum Rate (bit/s/Hz)",
        title="Algorithm Comparison: Secrecy Sum Rate",
        filename="fig_algorithm_comparison.png",
        out_dir=out_dir,
    )
    plot_metric_bar(
        metrics_df,
        metric="qos_satisfaction_rate",
        ylabel="QoS Satisfaction Rate",
        title="Algorithm Comparison: QoS Satisfaction",
        filename="fig_qos_satisfaction_comparison.png",
        out_dir=out_dir,
    )
    plot_metric_bar(
        metrics_df,
        metric="secrecy_outage_prob",
        ylabel="Secrecy Outage Probability",
        title="Algorithm Comparison: Secrecy Outage",
        filename="fig_secrecy_outage_comparison.png",
        out_dir=out_dir,
        ascending=True,
    )
    plot_metric_bar(
        metrics_df,
        metric="avg_decision_time_ms",
        ylabel="Average Decision Time (ms)",
        title="Algorithm Comparison: Decision Time",
        filename="fig_decision_time_comparison.png",
        out_dir=out_dir,
        ascending=True,
    )

    rate_df = metrics_df.sort_values("avg_legit_rate_sum", ascending=False)
    x = np.arange(len(rate_df))
    width = 0.38
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width / 2, rate_df["avg_legit_rate_sum"], width, label="Legitimate users")
    plt.bar(x + width / 2, rate_df["avg_eaves_rate_sum"], width, label="Eavesdropper")
    plt.ylabel("Average Rate Sum (bit/s/Hz)")
    plt.title("Algorithm Comparison: Legitimate vs Eavesdropper Rate")
    plt.xticks(x, rate_df["algorithm"], rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_legit_eaves_rate_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    order = metrics_df.sort_values("avg_secrecy_sum", ascending=False)
    plt.plot(order["algorithm"], order["avg_secrecy_sum"], marker="o", label="Secrecy sum")
    plt.plot(order["algorithm"], order["qos_satisfaction_rate"], marker="s", label="QoS satisfaction")
    plt.plot(order["algorithm"], order["secrecy_outage_prob"], marker="^", label="Secrecy outage")
    plt.ylabel("Metric Value")
    plt.title("Algorithm Comparison: Security and Reliability Metrics")
    plt.xticks(rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_security_reliability_metrics.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--grid-resolution", type=int, default=21)
    parser.add_argument("--pso-particles", type=int, default=30)
    parser.add_argument("--pso-iters", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    cfg = ExperimentConfig(
        seed=args.seed,
        train_episodes=args.train_episodes,
        episode_steps=args.episode_steps,
        eval_episodes=args.eval_episodes,
        grid_resolution=args.grid_resolution,
        pso_particles=args.pso_particles,
        pso_iters=args.pso_iters,
    )

    with open(out_dir / "config_used.json", "w", encoding="utf-8") as f:
        import json

        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    ddpg_agent, train_df = train_ddpg(cfg, args.device, out_dir)
    td3_agent, td3_train_df = train_agent("TD3", TD3Agent, cfg, args.device, out_dir)
    sac_agent, sac_train_df = train_agent("SAC", SACAgent, cfg, args.device, out_dir)

    combined_train_df = pd.concat(
        [
            train_df.assign(algorithm="DDPG"),
            td3_train_df.assign(algorithm="TD3"),
            sac_train_df.assign(algorithm="SAC"),
        ],
        ignore_index=True,
    )
    combined_train_df.to_csv(out_dir / "rl_training_log.csv", index=False)

    agents: Dict[str, RLAgent] = {
        "DDPG": ddpg_agent,
        "TD3": td3_agent,
        "SAC": sac_agent,
    }
    algos = [
        "DDPG",
        "TD3",
        "SAC",
        "Random",
        "Equal",
        "Heuristic",
        "PSO",
        "Grid",
        "DDPG_NoJammer",
    ]
    rows = [evaluate_algorithm(name, cfg, agents) for name in algos]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics_table.csv", index=False)

    plot_results(combined_train_df, metrics_df, out_dir)
    print("\n===== 评估结果 =====")
    print(metrics_df.to_string(index=False))
    print(f"\n结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
