from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import ExperimentConfig
from .ddpg import Actor, Critic, ReplayBuffer


class TD3Agent:
    def __init__(self, cfg: ExperimentConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = Actor(cfg.state_dim, cfg.action_dim).to(self.device)
        self.actor_target = Actor(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic1 = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic2 = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic1_target = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic2_target = Critic(cfg.state_dim, cfg.action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=cfg.critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=cfg.critic_lr)
        self.replay = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.replay_size)
        self.total_it = 0

    @torch.no_grad()
    def select_action(self, state: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.actor(s).squeeze(0).cpu().numpy()
        if noise_std > 0.0:
            action = action + np.random.normal(0.0, noise_std, size=action.shape)
        return self._normalize_action(action)

    def train_step(self) -> Tuple[float, float]:
        cfg = self.cfg
        if self.replay.size < max(cfg.batch_size, cfg.warmup_steps):
            return 0.0, 0.0

        self.total_it += 1
        s, a, r, s2, d = self.replay.sample(cfg.batch_size)
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        with torch.no_grad():
            a2 = self.actor_target(s2)
            noise = torch.randn_like(a2) * cfg.td3_policy_noise
            noise = noise.clamp(-cfg.td3_noise_clip, cfg.td3_noise_clip)
            a2 = torch.clamp(a2 + noise, min=1e-8)
            a2 = a2 / torch.clamp(a2.sum(dim=1, keepdim=True), min=1.0)
            q1_target = self.critic1_target(s2, a2)
            q2_target = self.critic2_target(s2, a2)
            y = r + cfg.gamma * (1.0 - d) * torch.min(q1_target, q2_target)

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        critic1_loss = nn.functional.mse_loss(q1, y)
        critic2_loss = nn.functional.mse_loss(q2, y)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.total_it % cfg.td3_policy_delay == 0:
            actor_loss = -self.critic1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.actor, self.actor_target, cfg.tau)
            self._soft_update(self.critic1, self.critic1_target, cfg.tau)
            self._soft_update(self.critic2, self.critic2_target, cfg.tau)

        critic_loss = critic1_loss + critic2_loss
        return float(actor_loss.item()), float(critic_loss.item())

    @staticmethod
    def _normalize_action(action: np.ndarray) -> np.ndarray:
        action = np.maximum(action, 1e-8)
        total = action.sum()
        if total > 1.0:
            action = action / total
        return action.astype(np.float32)

    @staticmethod
    def _soft_update(src: nn.Module, dst: nn.Module, tau: float):
        with torch.no_grad():
            for p_src, p_dst in zip(src.parameters(), dst.parameters()):
                p_dst.data.mul_(1.0 - tau)
                p_dst.data.add_(tau * p_src.data)
