from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np

from .config import ExperimentConfig


class NomaSecurityEnv:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.current_gains: Dict[str, float] = {}
        self.prev_action = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        self.step_idx = 0

    def reset(self) -> np.ndarray:
        self.step_idx = 0
        self.prev_action = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        self.current_gains = self._sample_channel_gains()
        return self._build_state()

    def _sample_channel_gains(self) -> Dict[str, float]:
        c = self.cfg
        return {
            "h1": self.rng.exponential(scale=c.l1),
            "h2": self.rng.exponential(scale=c.l2),
            "g": self.rng.exponential(scale=c.le),
            "hj1": self.rng.exponential(scale=c.lj1),
            "hj2": self.rng.exponential(scale=c.lj2),
            "gj": self.rng.exponential(scale=c.lje),
        }

    def _build_state(self) -> np.ndarray:
        g = self.current_gains
        c = self.cfg
        state = np.array(
            [
                g["h1"],
                g["h2"],
                g["g"],
                g["hj1"],
                g["hj2"],
                g["gj"],
                c.r1_min,
                c.r2_min,
                c.p_max,
                self.prev_action[0],
                self.prev_action[1],
                self.prev_action[2],
            ],
            dtype=np.float32,
        )
        return state

    def _sanitize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.maximum(action, 1e-8)
        total = float(action.sum())
        if total > 1.0:
            action = action / total
        return action

    def _action_to_power(self, action: np.ndarray) -> Tuple[float, float, float]:
        a = self._sanitize_action(action)
        p1, p2, pj = (a * self.cfg.p_max).tolist()
        return float(p1), float(p2), float(pj)

    def evaluate_action(
        self, action: np.ndarray, gains: Dict[str, float] | None = None
    ) -> Dict[str, float]:
        c = self.cfg
        g = gains if gains is not None else self.current_gains
        p1, p2, pj = self._action_to_power(action)
        sigma2 = c.noise_power

        gamma1 = g["h1"] * p1 / (g["h1"] * p2 + g["hj1"] * pj + sigma2)
        gamma21 = g["h2"] * p1 / (g["h2"] * p2 + g["hj2"] * pj + sigma2)
        gamma2 = g["h2"] * p2 / (g["hj2"] * pj + sigma2)

        gammae1 = g["g"] * p1 / (g["g"] * p2 + g["gj"] * pj + sigma2)
        gammae2 = g["g"] * p2 / (g["gj"] * pj + sigma2)

        r1 = np.log2(1.0 + gamma1)
        r2 = np.log2(1.0 + gamma2)
        re1 = np.log2(1.0 + gammae1)
        re2 = np.log2(1.0 + gammae2)

        rs1 = max(0.0, r1 - re1)
        rs2 = max(0.0, r2 - re2)
        rs_sum = rs1 + rs2

        qos_pen = max(0.0, c.r1_min - r1) + max(0.0, c.r2_min - r2)
        sic_pen = max(0.0, gamma1 - gamma21)
        power_pen = max(0.0, p1 + p2 + pj - c.p_max)

        reward = (
            rs_sum
            - c.lambda_qos * qos_pen
            - c.lambda_sic * sic_pen
            - c.lambda_power * power_pen
        )

        qos_ok = (r1 >= c.r1_min) and (r2 >= c.r2_min)
        sic_ok = gamma21 >= gamma1
        secrecy_outage = rs_sum < c.secrecy_outage_threshold

        return {
            "p1": p1,
            "p2": p2,
            "pj": pj,
            "gamma1": float(gamma1),
            "gamma21": float(gamma21),
            "gamma2": float(gamma2),
            "gammae1": float(gammae1),
            "gammae2": float(gammae2),
            "r1": float(r1),
            "r2": float(r2),
            "re1": float(re1),
            "re2": float(re2),
            "rs1": float(rs1),
            "rs2": float(rs2),
            "rs_sum": float(rs_sum),
            "qos_pen": float(qos_pen),
            "sic_pen": float(sic_pen),
            "power_pen": float(power_pen),
            "reward": float(reward),
            "qos_ok": float(qos_ok),
            "sic_ok": float(sic_ok),
            "secrecy_outage": float(secrecy_outage),
        }

    def step(self, action: np.ndarray):
        action = self._sanitize_action(action)
        self.prev_action = action.copy()
        metrics = self.evaluate_action(action, gains=self.current_gains)

        self.step_idx += 1
        done = self.step_idx >= self.cfg.episode_steps
        self.current_gains = self._sample_channel_gains()
        next_state = self._build_state()
        return next_state, metrics["reward"], done, metrics
