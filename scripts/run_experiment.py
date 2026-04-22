#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

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


def train_ddpg(cfg: ExperimentConfig, device: str, out_dir: Path):
    env = NomaSecurityEnv(cfg)
    agent = DDPGAgent(cfg, device=device)

    episode_rewards, episode_secrecy, episode_qos = [], [], []
    actor_losses, critic_losses = [], []

    for ep in trange(cfg.train_episodes, desc="Training DDPG"):
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

    torch.save(agent.actor.state_dict(), out_dir / "ddpg_actor.pt")
    torch.save(agent.critic.state_dict(), out_dir / "ddpg_critic.pt")

    train_df = pd.DataFrame(
        {
            "episode": np.arange(1, cfg.train_episodes + 1),
            "avg_reward": episode_rewards,
            "avg_secrecy_sum": episode_secrecy,
            "qos_satisfaction_rate": episode_qos,
        }
    )
    train_df.to_csv(out_dir / "training_log.csv", index=False)

    return agent, train_df


def evaluate_algorithm(name: str, cfg: ExperimentConfig, agent: DDPGAgent | None):
    env = NomaSecurityEnv(cfg)
    rng = np.random.default_rng(cfg.seed + 123)

    rs_values, legit_values, eaves_values = [], [], []
    qos_hits, outage_hits, decision_times = [], [], []

    for _ in trange(cfg.eval_episodes, desc=f"Evaluating {name}"):
        _ = env.reset()
        for _ in range(cfg.episode_steps):
            t0 = time.perf_counter()

            if name == "DDPG":
                assert agent is not None
                action = agent.select_action(env._build_state(), noise_std=0.0)
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
                assert agent is not None
                raw = agent.select_action(env._build_state(), noise_std=0.0)
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


def plot_results(train_df: pd.DataFrame, metrics_df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(8, 4))
    plt.plot(train_df["episode"], train_df["avg_reward"], alpha=0.35, label="Episode reward")
    ma = moving_average(train_df["avg_reward"].tolist(), window=50)
    if len(ma) > 0:
        x = np.arange(50, 50 + len(ma)) if len(train_df) >= 50 else np.arange(1, len(ma) + 1)
        plt.plot(x, ma, linewidth=2.0, label="MA(50)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DDPG Training Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_training_convergence.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    order = metrics_df.sort_values("avg_secrecy_sum", ascending=False)
    plt.bar(order["algorithm"], order["avg_secrecy_sum"])
    plt.ylabel("Average Secrecy Sum Rate (bit/s/Hz)")
    plt.title("Algorithm Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_algorithm_comparison.png", dpi=160)
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

    agent, train_df = train_ddpg(cfg, args.device, out_dir)

    algos = ["DDPG", "Random", "Equal", "Heuristic", "PSO", "Grid", "DDPG_NoJammer"]
    rows = [evaluate_algorithm(name, cfg, agent) for name in algos]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics_table.csv", index=False)

    plot_results(train_df, metrics_df, out_dir)
    print("\n===== 评估结果 =====")
    print(metrics_df.to_string(index=False))
    print(f"\n结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
