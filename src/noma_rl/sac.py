from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import ExperimentConfig
from .ddpg import Critic, ReplayBuffer, mlp


class SACActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = mlp(state_dim, action_dim * 2)
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(state)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        logits = torch.tanh(z)
        action = torch.softmax(logits, dim=-1)
        log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(state)
        return torch.softmax(torch.tanh(mean), dim=-1)


class SACAgent:
    def __init__(self, cfg: ExperimentConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = SACActor(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic1 = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic2 = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic1_target = Critic(cfg.state_dim, cfg.action_dim).to(self.device)
        self.critic2_target = Critic(cfg.state_dim, cfg.action_dim).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=cfg.critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=cfg.critic_lr)
        self.replay = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.replay_size)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if noise_std > 0.0:
            action, _ = self.actor.sample(s)
        else:
            action = self.actor.deterministic(s)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

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
            a2, logp2 = self.actor.sample(s2)
            q1_target = self.critic1_target(s2, a2)
            q2_target = self.critic2_target(s2, a2)
            q_target = torch.min(q1_target, q2_target) - cfg.sac_alpha * logp2
            y = r + cfg.gamma * (1.0 - d) * q_target

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

        new_a, logp = self.actor.sample(s)
        q1_new = self.critic1(s, new_a)
        q2_new = self.critic2(s, new_a)
        actor_loss = (cfg.sac_alpha * logp - torch.min(q1_new, q2_new)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.critic1, self.critic1_target, cfg.tau)
        self._soft_update(self.critic2, self.critic2_target, cfg.tau)

        critic_loss = critic1_loss + critic2_loss
        return float(actor_loss.item()), float(critic_loss.item())

    @staticmethod
    def _soft_update(src: nn.Module, dst: nn.Module, tau: float):
        with torch.no_grad():
            for p_src, p_dst in zip(src.parameters(), dst.parameters()):
                p_dst.data.mul_(1.0 - tau)
                p_dst.data.add_(tau * p_src.data)
