from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import ExperimentConfig


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, s, a, r, s2, d):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s2
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.state[idx]),
            torch.from_numpy(self.action[idx]),
            torch.from_numpy(self.reward[idx]),
            torch.from_numpy(self.next_state[idx]),
            torch.from_numpy(self.done[idx]),
        )


def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, out_dim),
    )


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = mlp(state_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = mlp(state_dim + action_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class DDPGAgent:
    def __init__(self, cfg: ExperimentConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = Actor(cfg.state_dim, cfg.action_dim).to(self.device)
        self.actor_target = Actor(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic_target = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.replay = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.replay_size)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.actor(s).squeeze(0).cpu().numpy()
        if noise_std > 0.0:
            action = action + np.random.normal(0.0, noise_std, size=action.shape)
        action = np.maximum(action, 1e-8)
        total = action.sum()
        if total > 1.0:
            action = action / total
        return action.astype(np.float32)

    def train_step(self) -> Tuple[float, float]:
        cfg = self.cfg
        if self.replay.size < max(cfg.batch_size, cfg.warmup_steps):
            return 0.0, 0.0

        s, a, r, s2, d = self.replay.sample(cfg.batch_size)
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        with torch.no_grad():
            a2 = self.actor_target(s2)
            q2 = self.critic_target(s2, a2)
            y = r + cfg.gamma * (1.0 - d) * q2

        q = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.actor, self.actor_target, cfg.tau)
        self._soft_update(self.critic, self.critic_target, cfg.tau)
        return float(actor_loss.item()), float(critic_loss.item())

    @staticmethod
    def _soft_update(src: nn.Module, dst: nn.Module, tau: float):
        with torch.no_grad():
            for p_src, p_dst in zip(src.parameters(), dst.parameters()):
                p_dst.data.mul_(1.0 - tau)
                p_dst.data.add_(tau * p_src.data)
