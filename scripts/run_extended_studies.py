#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Protocol

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from noma_rl.baselines import (  # noqa: E402
    ddpg_without_jammer,
    equal_action,
    grid_search_optimize,
    heuristic_action,
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


class ImitationAgent:
    def __init__(self, cfg: ExperimentConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = nn.Sequential(
            nn.Linear(cfg.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, cfg.action_dim),
        ).to(self.device)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        del noise_std
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        logits = self.model(s)
        action = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return action.astype(np.float32)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_base_experiment_cfg(results_dir: Path) -> ExperimentConfig:
    cfg_path = results_dir / "config_used.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return ExperimentConfig(**json.load(f))
    return ExperimentConfig()


def load_agent_weights(
    agent: RLAgent,
    results_dir: Path,
    prefix: str,
    device: str = "cpu",
) -> RLAgent:
    if hasattr(agent, "actor"):
        actor_path = results_dir / f"{prefix}_actor.pt"
        if actor_path.exists():
            agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    if hasattr(agent, "critic"):
        critic_path = results_dir / f"{prefix}_critic.pt"
        if critic_path.exists():
            agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
    if hasattr(agent, "critic1"):
        critic1_path = results_dir / f"{prefix}_critic1.pt"
        if critic1_path.exists():
            agent.critic1.load_state_dict(torch.load(critic1_path, map_location=device))
    if hasattr(agent, "critic2"):
        critic2_path = results_dir / f"{prefix}_critic2.pt"
        if critic2_path.exists():
            agent.critic2.load_state_dict(torch.load(critic2_path, map_location=device))
    return agent


def sample_channel_gains(cfg: ExperimentConfig, rng: np.random.Generator) -> dict[str, float]:
    return {
        "h1": float(rng.exponential(scale=cfg.l1)),
        "h2": float(rng.exponential(scale=cfg.l2)),
        "g": float(rng.exponential(scale=cfg.le)),
        "hj1": float(rng.exponential(scale=cfg.lj1)),
        "hj2": float(rng.exponential(scale=cfg.lj2)),
        "gj": float(rng.exponential(scale=cfg.lje)),
    }


def generate_fixed_channel_episodes(
    cfg: ExperimentConfig,
    eval_episodes: int,
    episode_steps: int,
    seed: int,
) -> list[list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    return [
        [sample_channel_gains(cfg, rng) for _ in range(episode_steps)]
        for _ in range(eval_episodes)
    ]


def no_jammer_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size == 2:
        return np.array([action[0], action[1], 0.0], dtype=np.float32)
    return ddpg_without_jammer(action)


def train_agent(
    agent_cls: type[RLAgent],
    env_cfg: ExperimentConfig,
    device: str = "cpu",
    agent_cfg: ExperimentConfig | None = None,
    action_adapter: Callable[[np.ndarray], np.ndarray] | None = None,
) -> RLAgent:
    env = NomaSecurityEnv(env_cfg)
    agent = agent_cls(agent_cfg or env_cfg, device=device)

    for ep in trange(env_cfg.train_episodes, desc=f"Training {agent_cls.__name__}", leave=False):
        state = env.reset()
        noise_std = max(0.01, 0.2 * (1.0 - ep / max(1, env_cfg.train_episodes)))
        for _ in range(env_cfg.episode_steps):
            raw_action = agent.select_action(state, noise_std=noise_std)
            env_action = action_adapter(raw_action) if action_adapter is not None else raw_action
            next_state, reward, done, _ = env.step(env_action)
            agent.replay.add(
                state,
                raw_action,
                np.array([reward], dtype=np.float32),
                next_state,
                float(done),
            )
            agent.train_step()
            state = next_state
            if done:
                break
    return agent


def train_imitation_agent(
    cfg: ExperimentConfig,
    device: str = "cpu",
    samples: int = 600,
    epochs: int = 40,
    label_grid_resolution: int = 11,
) -> tuple[ImitationAgent, pd.DataFrame]:
    env = NomaSecurityEnv(cfg)
    rng = np.random.default_rng(cfg.seed + 2024)
    states, actions = [], []

    for _ in trange(samples, desc="Labeling Imitation data", leave=False):
        env.current_gains = sample_channel_gains(cfg, rng)
        env.prev_action = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        state = env._build_state()
        label = grid_search_optimize(
            lambda a: env.evaluate_action(a, gains=env.current_gains)["reward"],
            resolution=label_grid_resolution,
        )
        states.append(state)
        actions.append(label)

    x = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(torch.device(device))
    y = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(torch.device(device))

    agent = ImitationAgent(cfg, device=device)
    optimizer = optim.Adam(agent.model.parameters(), lr=1e-3)
    losses = []
    batch_size = 64

    for epoch in trange(epochs, desc="Training Imitation", leave=False):
        indices = torch.randperm(x.size(0), device=x.device)
        epoch_loss = 0.0
        for start in range(0, x.size(0), batch_size):
            idx = indices[start : start + batch_size]
            logits = agent.model(x[idx])
            pred = torch.softmax(logits, dim=-1)
            loss = nn.functional.mse_loss(pred, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * idx.numel()
        losses.append({"epoch": epoch + 1, "train_loss": epoch_loss / x.size(0)})

    return agent, pd.DataFrame(losses)


def select_algorithm_action(
    name: str,
    state: np.ndarray,
    gains: dict[str, float],
    env: NomaSecurityEnv,
    agents: dict[str, RLAgent],
    rng: np.random.Generator,
    grid_resolution: int,
) -> np.ndarray:
    if name in agents:
        return agents[name].select_action(state, noise_std=0.0)
    if name == "DDPG_NoJammer":
        return ddpg_without_jammer(agents["DDPG"].select_action(state, noise_std=0.0))
    if name == "Random":
        return random_action(rng)
    if name == "Equal":
        return equal_action()
    if name == "Heuristic":
        return heuristic_action()
    if name == "Grid":
        return grid_search_optimize(
            lambda a: env.evaluate_action(a, gains=gains)["reward"],
            resolution=grid_resolution,
        )
    if name == "Grid_LegitRate":
        return grid_search_optimize(
            lambda a: env.evaluate_action(a, gains=gains)["r1"] + env.evaluate_action(a, gains=gains)["r2"],
            resolution=grid_resolution,
        )
    raise ValueError(f"Unknown algorithm: {name}")


def evaluate_on_fixed_channels(
    cfg: ExperimentConfig,
    channels: list[list[dict[str, float]]],
    agents: dict[str, RLAgent],
    algorithms: list[str],
    grid_resolution: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for name in algorithms:
        env = NomaSecurityEnv(cfg)
        rng = np.random.default_rng(seed)
        rs_values, legit_values, eaves_values = [], [], []
        qos_hits, outage_hits, decision_times = [], [], []

        for episode_gains in channels:
            env.prev_action = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
            for gains in episode_gains:
                env.current_gains = gains
                state = env._build_state()
                t0 = time.perf_counter()
                action = select_algorithm_action(
                    name,
                    state,
                    gains,
                    env,
                    agents,
                    rng,
                    grid_resolution,
                )

                decision_times.append((time.perf_counter() - t0) * 1e3)
                action = env._sanitize_action(action)
                info = env.evaluate_action(action, gains=gains)
                env.prev_action = action.copy()

                rs_values.append(info["rs_sum"])
                legit_values.append(info["r1"] + info["r2"])
                eaves_values.append(info["re1"] + info["re2"])
                qos_hits.append(info["qos_ok"])
                outage_hits.append(info["secrecy_outage"])

        rows.append(
            {
                "algorithm": name,
                "avg_secrecy_sum": float(np.mean(rs_values)),
                "avg_legit_rate_sum": float(np.mean(legit_values)),
                "avg_eaves_rate_sum": float(np.mean(eaves_values)),
                "qos_satisfaction_rate": float(np.mean(qos_hits)),
                "secrecy_outage_prob": float(np.mean(outage_hits)),
                "avg_decision_time_ms": float(np.mean(decision_times)),
            }
        )

    return pd.DataFrame(rows)


def collect_detailed_fixed_channel_records(
    cfg: ExperimentConfig,
    channels: list[list[dict[str, float]]],
    agents: dict[str, RLAgent],
    algorithms: list[str],
    grid_resolution: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for name in algorithms:
        env = NomaSecurityEnv(cfg)
        rng = np.random.default_rng(seed)
        for episode_idx, episode_gains in enumerate(channels, start=1):
            env.prev_action = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
            for step_idx, gains in enumerate(episode_gains, start=1):
                env.current_gains = gains
                state = env._build_state()
                t0 = time.perf_counter()
                action = select_algorithm_action(
                    name,
                    state,
                    gains,
                    env,
                    agents,
                    rng,
                    grid_resolution,
                )
                decision_time_ms = (time.perf_counter() - t0) * 1e3
                action = env._sanitize_action(action)
                info = env.evaluate_action(action, gains=gains)
                env.prev_action = action.copy()
                rows.append(
                    {
                        "algorithm": name,
                        "episode": episode_idx,
                        "step": step_idx,
                        "p1": info["p1"],
                        "p2": info["p2"],
                        "pj": info["pj"],
                        "rs1": info["rs1"],
                        "rs2": info["rs2"],
                        "rs_sum": info["rs_sum"],
                        "r1": info["r1"],
                        "r2": info["r2"],
                        "re1": info["re1"],
                        "re2": info["re2"],
                        "qos_ok": info["qos_ok"],
                        "sic_ok": info["sic_ok"],
                        "decision_time_ms": decision_time_ms,
                    }
                )
    return pd.DataFrame(rows)


def make_loaded_agents(cfg: ExperimentConfig, results_dir: Path, device: str) -> dict[str, RLAgent]:
    return {
        "DDPG": load_agent_weights(DDPGAgent(cfg, device=device), results_dir, "ddpg", device),
        "TD3": load_agent_weights(TD3Agent(cfg, device=device), results_dir, "td3", device),
        "SAC": load_agent_weights(SACAgent(cfg, device=device), results_dir, "sac", device),
    }


def run_fixed_testset_benchmark(
    out_dir: Path,
    cfg: ExperimentConfig,
    agents: dict[str, RLAgent],
    grid_resolution: int,
):
    channels = generate_fixed_channel_episodes(cfg, eval_episodes=16, episode_steps=40, seed=2026)
    df = evaluate_on_fixed_channels(
        cfg,
        channels,
        agents,
        algorithms=[
            "DDPG",
            "TD3",
            "SAC",
            "Imitation",
            "Random",
            "Equal",
            "Heuristic",
            "Grid",
            "Grid_LegitRate",
            "DDPG_NoJammer",
        ],
        grid_resolution=grid_resolution,
        seed=2026,
    ).sort_values("avg_secrecy_sum", ascending=False)
    df.to_csv(out_dir / "fixed_testset_benchmark.csv", index=False)
    return df


def run_metric_enrichment_study(
    out_dir: Path,
    cfg: ExperimentConfig,
    agents: dict[str, RLAgent],
    grid_resolution: int,
):
    channels = generate_fixed_channel_episodes(cfg, eval_episodes=12, episode_steps=30, seed=2500)
    detailed_df = collect_detailed_fixed_channel_records(
        cfg,
        channels,
        agents,
        algorithms=["DDPG", "TD3", "SAC", "Heuristic", "Grid", "DDPG_NoJammer"],
        grid_resolution=grid_resolution,
        seed=2500,
    )
    detailed_df.to_csv(out_dir / "fixed_testset_detailed_metrics.csv", index=False)
    summary_df = (
        detailed_df.groupby("algorithm", as_index=False)
        .agg(
            avg_rs1=("rs1", "mean"),
            avg_rs2=("rs2", "mean"),
            avg_rs_sum=("rs_sum", "mean"),
            avg_legit_gap=("r1", "mean"),
            avg_legit_gap_u2=("r2", "mean"),
            avg_eaves_u1=("re1", "mean"),
            avg_eaves_u2=("re2", "mean"),
            qos_violation_rate=("qos_ok", lambda x: 1.0 - float(np.mean(x))),
            sic_violation_rate=("sic_ok", lambda x: 1.0 - float(np.mean(x))),
        )
        .sort_values("avg_rs_sum", ascending=False)
    )
    summary_df.to_csv(out_dir / "fixed_testset_metric_enrichment.csv", index=False)
    return detailed_df, summary_df


def run_pmax_sweep_study(
    out_dir: Path,
    base_cfg: ExperimentConfig,
    agents: dict[str, RLAgent],
    grid_resolution: int,
    pmax_values: list[float],
):
    rows = []
    for pmax in pmax_values:
        cfg = ExperimentConfig(**{**asdict(base_cfg), "p_max": pmax})
        channels = generate_fixed_channel_episodes(cfg, eval_episodes=10, episode_steps=25, seed=int(3000 + pmax * 10))
        df = evaluate_on_fixed_channels(
            cfg,
            channels,
            agents,
            algorithms=["DDPG", "TD3", "SAC", "Imitation", "Heuristic", "Grid", "DDPG_NoJammer"],
            grid_resolution=grid_resolution,
            seed=int(3500 + pmax * 10),
        )
        df.insert(0, "p_max", pmax)
        rows.append(df)
    result_df = pd.concat(rows, ignore_index=True)
    result_df.to_csv(out_dir / "sweep_pmax.csv", index=False)
    return result_df


def run_latency_stability_study(
    out_dir: Path,
    cfg: ExperimentConfig,
    agents: dict[str, RLAgent],
    grid_resolution: int,
):
    channels = generate_fixed_channel_episodes(cfg, eval_episodes=6, episode_steps=20, seed=4040)
    detailed_df = collect_detailed_fixed_channel_records(
        cfg,
        channels,
        agents,
        algorithms=["DDPG", "TD3", "SAC", "Heuristic", "Grid"],
        grid_resolution=grid_resolution,
        seed=4040,
    )
    summary_df = (
        detailed_df.groupby("algorithm", as_index=False)
        .agg(
            avg_decision_time_ms=("decision_time_ms", "mean"),
            std_decision_time_ms=("decision_time_ms", "std"),
            p95_decision_time_ms=("decision_time_ms", lambda x: float(np.percentile(x, 95))),
            avg_rs_sum=("rs_sum", "mean"),
        )
        .sort_values("avg_decision_time_ms")
    )
    summary_df.to_csv(out_dir / "latency_stability.csv", index=False)
    return summary_df


def run_action_behavior_study(
    out_dir: Path,
    cfg: ExperimentConfig,
    agents: dict[str, RLAgent],
    grid_resolution: int,
):
    channels = generate_fixed_channel_episodes(cfg, eval_episodes=12, episode_steps=30, seed=5050)
    detailed_df = collect_detailed_fixed_channel_records(
        cfg,
        channels,
        agents,
        algorithms=["DDPG", "TD3", "SAC", "Heuristic", "Grid", "DDPG_NoJammer"],
        grid_resolution=grid_resolution,
        seed=5050,
    )
    summary_df = (
        detailed_df.groupby("algorithm", as_index=False)
        .agg(
            avg_p1=("p1", "mean"),
            avg_p2=("p2", "mean"),
            avg_pj=("pj", "mean"),
            avg_rs_sum=("rs_sum", "mean"),
        )
        .sort_values("avg_rs_sum", ascending=False)
    )
    total_power = summary_df["avg_p1"] + summary_df["avg_p2"] + summary_df["avg_pj"]
    summary_df["avg_p1_share"] = summary_df["avg_p1"] / np.maximum(total_power, 1e-8)
    summary_df["avg_p2_share"] = summary_df["avg_p2"] / np.maximum(total_power, 1e-8)
    summary_df["avg_pj_share"] = summary_df["avg_pj"] / np.maximum(total_power, 1e-8)
    summary_df.to_csv(out_dir / "action_behavior_summary.csv", index=False)
    return summary_df


def run_training_budget_study(
    out_dir: Path,
    base_cfg: ExperimentConfig,
    device: str,
    budgets: list[int],
    seeds: list[int],
):
    rows = []
    common = asdict(base_cfg)
    common.update({"episode_steps": 30, "eval_episodes": 12})
    eval_cfg = ExperimentConfig(**common)
    channels = generate_fixed_channel_episodes(eval_cfg, eval_cfg.eval_episodes, eval_cfg.episode_steps, 4096)

    for budget in budgets:
        for seed in seeds:
            cfg = ExperimentConfig(**{**common, "seed": seed, "train_episodes": budget})
            set_seed(seed)
            agents = {
                "DDPG": train_agent(DDPGAgent, cfg, device=device),
                "TD3": train_agent(TD3Agent, cfg, device=device),
                "SAC": train_agent(SACAgent, cfg, device=device),
            }
            df = evaluate_on_fixed_channels(
                cfg,
                channels,
                agents,
                algorithms=["DDPG", "TD3", "SAC", "Heuristic", "Grid"],
                grid_resolution=base_cfg.grid_resolution,
                seed=seed + budget,
            )
            df.insert(0, "seed", seed)
            df.insert(0, "train_episodes", budget)
            rows.append(df)

    detailed_df = pd.concat(rows, ignore_index=True)
    detailed_df.to_csv(out_dir / "training_budget_detailed.csv", index=False)

    summary_df = (
        detailed_df.groupby(["train_episodes", "algorithm"], as_index=False)
        .agg(
            avg_secrecy_sum_mean=("avg_secrecy_sum", "mean"),
            avg_secrecy_sum_std=("avg_secrecy_sum", "std"),
            qos_satisfaction_rate_mean=("qos_satisfaction_rate", "mean"),
            secrecy_outage_prob_mean=("secrecy_outage_prob", "mean"),
        )
        .sort_values(["train_episodes", "algorithm"])
    )
    summary_df.to_csv(out_dir / "training_budget_summary.csv", index=False)
    return summary_df


def run_reward_ablation_study(
    out_dir: Path,
    base_cfg: ExperimentConfig,
    device: str,
):
    variants = [
        ("FullReward", {}),
        ("NoQoSPenalty", {"lambda_qos": 0.0}),
        ("NoSICPenalty", {"lambda_sic": 0.0}),
        ("NoPowerPenalty", {"lambda_power": 0.0}),
    ]
    rows = []
    base = asdict(base_cfg)
    base.update({"seed": 42, "train_episodes": 220, "episode_steps": 30, "eval_episodes": 12})

    for name, updates in variants:
        cfg = ExperimentConfig(**{**base, **updates})
        set_seed(cfg.seed)
        agent = train_agent(DDPGAgent, cfg, device=device)
        channels = generate_fixed_channel_episodes(cfg, cfg.eval_episodes, cfg.episode_steps, 9000)
        df = evaluate_on_fixed_channels(
            cfg,
            channels,
            {"DDPG": agent},
            algorithms=["DDPG"],
            grid_resolution=cfg.grid_resolution,
            seed=77,
        )
        df["algorithm"] = name
        rows.append(df)

    result_df = pd.concat(rows, ignore_index=True)
    result_df.to_csv(out_dir / "reward_ablation.csv", index=False)
    return result_df


def run_retrained_no_jammer_study(
    out_dir: Path,
    base_cfg: ExperimentConfig,
    full_ddpg_agent: RLAgent,
    device: str,
):
    env_cfg = ExperimentConfig(**{**asdict(base_cfg), "seed": 42, "train_episodes": 220, "episode_steps": 30, "eval_episodes": 12})
    agent_cfg = ExperimentConfig(**{**asdict(env_cfg), "action_dim": 2})
    set_seed(env_cfg.seed)
    retrained_agent = train_agent(
        DDPGAgent,
        env_cfg,
        device=device,
        agent_cfg=agent_cfg,
        action_adapter=no_jammer_action,
    )
    channels = generate_fixed_channel_episodes(env_cfg, env_cfg.eval_episodes, env_cfg.episode_steps, 12345)
    df = evaluate_on_fixed_channels(
        env_cfg,
        channels,
        {
            "DDPG": full_ddpg_agent,
            "DDPG_NoJammer_Retrained": retrained_agent,
        },
        algorithms=["DDPG", "DDPG_NoJammer_Retrained", "DDPG_NoJammer", "Heuristic", "Grid"],
        grid_resolution=env_cfg.grid_resolution,
        seed=31415,
    ).sort_values("avg_secrecy_sum", ascending=False)
    df.to_csv(out_dir / "retrained_no_jammer_comparison.csv", index=False)
    return df


def plot_bar(df: pd.DataFrame, out_path: Path, title: str):
    order = df.sort_values("avg_secrecy_sum", ascending=False)
    plt.figure(figsize=(9, 4))
    plt.bar(order["algorithm"], order["avg_secrecy_sum"])
    plt.ylabel("Average Secrecy Sum Rate")
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_training_budget(summary_df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 4))
    for name in ["DDPG", "TD3", "SAC"]:
        sub = summary_df[summary_df["algorithm"] == name].sort_values("train_episodes")
        if len(sub) == 0:
            continue
        plt.errorbar(
            sub["train_episodes"],
            sub["avg_secrecy_sum_mean"],
            yerr=sub["avg_secrecy_sum_std"].fillna(0.0),
            marker="o",
            capsize=4,
            label=name,
        )
    for name in ["Heuristic", "Grid"]:
        sub = summary_df[summary_df["algorithm"] == name].sort_values("train_episodes")
        if len(sub) == 0:
            continue
        plt.plot(sub["train_episodes"], sub["avg_secrecy_sum_mean"], linestyle="--", marker="o", label=name)
    plt.xlabel("Training Episodes")
    plt.ylabel("Average Secrecy Sum Rate")
    plt.title("Training Budget Study")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_pmax_sweep(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 4))
    for name in ["DDPG", "TD3", "SAC", "Imitation", "Heuristic", "Grid", "DDPG_NoJammer"]:
        sub = df[df["algorithm"] == name].sort_values("p_max")
        if len(sub) == 0:
            continue
        plt.plot(sub["p_max"], sub["avg_secrecy_sum"], marker="o", label=name)
    plt.xlabel("p_max")
    plt.ylabel("Average Secrecy Sum Rate")
    plt.title("Pmax Sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_latency_tradeoff(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 4))
    plt.scatter(df["avg_decision_time_ms"], df["avg_rs_sum"])
    for _, row in df.iterrows():
        plt.annotate(row["algorithm"], (row["avg_decision_time_ms"], row["avg_rs_sum"]), fontsize=8)
    plt.xlabel("Average Decision Time (ms)")
    plt.ylabel("Average Secrecy Sum Rate")
    plt.title("Latency-Performance Tradeoff")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_action_behavior(df: pd.DataFrame, out_path: Path):
    order = df.sort_values("avg_rs_sum", ascending=False).reset_index(drop=True)
    x = np.arange(len(order))
    plt.figure(figsize=(9, 4))
    plt.bar(x, order["avg_p1_share"], label="P1 share")
    plt.bar(x, order["avg_p2_share"], bottom=order["avg_p1_share"], label="P2 share")
    plt.bar(
        x,
        order["avg_pj_share"],
        bottom=order["avg_p1_share"] + order["avg_p2_share"],
        label="PJ share",
    )
    plt.xticks(x, order["algorithm"], rotation=20)
    plt.ylabel("Average Power Share")
    plt.title("Action Allocation Behavior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "outputs" / "main"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "outputs" / "extended"))
    parser.add_argument("--budget-seeds", type=int, nargs="+", default=[7, 21, 42])
    parser.add_argument("--fast-only", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_base_experiment_cfg(results_dir)
    loaded_agents = make_loaded_agents(base_cfg, results_dir, args.device)
    imitation_agent, imitation_log = train_imitation_agent(
        base_cfg,
        device=args.device,
        samples=600,
        epochs=40,
        label_grid_resolution=11,
    )
    loaded_agents["Imitation"] = imitation_agent
    imitation_log.to_csv(out_dir / "imitation_training_log.csv", index=False)

    fixed_df = run_fixed_testset_benchmark(out_dir, base_cfg, loaded_agents, base_cfg.grid_resolution)
    plot_bar(fixed_df, out_dir / "fig_fixed_testset_benchmark.png", "Fixed Testset Benchmark")

    _, metric_enrichment_df = run_metric_enrichment_study(
        out_dir,
        base_cfg,
        loaded_agents,
        base_cfg.grid_resolution,
    )

    pmax_df = run_pmax_sweep_study(
        out_dir,
        base_cfg,
        loaded_agents,
        base_cfg.grid_resolution,
        pmax_values=[6.0, 10.0, 14.0],
    )
    plot_pmax_sweep(pmax_df, out_dir / "fig_sweep_pmax.png")

    latency_df = run_latency_stability_study(
        out_dir,
        base_cfg,
        loaded_agents,
        base_cfg.grid_resolution,
    )
    plot_latency_tradeoff(latency_df, out_dir / "fig_latency_tradeoff.png")

    action_behavior_df = run_action_behavior_study(
        out_dir,
        base_cfg,
        loaded_agents,
        base_cfg.grid_resolution,
    )
    plot_action_behavior(action_behavior_df, out_dir / "fig_action_behavior.png")

    summary = {
        "fixed_testset_benchmark": "fixed_testset_benchmark.csv",
        "fixed_testset_metric_enrichment": "fixed_testset_metric_enrichment.csv",
        "sweep_pmax": "sweep_pmax.csv",
        "latency_stability": "latency_stability.csv",
        "action_behavior_summary": "action_behavior_summary.csv",
        "imitation_training_log": "imitation_training_log.csv",
    }

    if not args.fast_only:
        budget_df = run_training_budget_study(
            out_dir,
            base_cfg,
            args.device,
            budgets=[80, 160, 320],
            seeds=args.budget_seeds,
        )
        plot_training_budget(budget_df, out_dir / "fig_training_budget_study.png")

        reward_df = run_reward_ablation_study(out_dir, base_cfg, args.device)
        plot_bar(reward_df, out_dir / "fig_reward_ablation.png", "Reward Ablation")

        no_jammer_df = run_retrained_no_jammer_study(out_dir, base_cfg, loaded_agents["DDPG"], args.device)
        plot_bar(
            no_jammer_df,
            out_dir / "fig_retrained_no_jammer_comparison.png",
            "Retrained No-Jammer Comparison",
        )

        summary.update(
            {
                "training_budget_summary": "training_budget_summary.csv",
                "reward_ablation": "reward_ablation.csv",
                "retrained_no_jammer_comparison": "retrained_no_jammer_comparison.csv",
            }
        )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== 扩展实验完成 =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\n结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
