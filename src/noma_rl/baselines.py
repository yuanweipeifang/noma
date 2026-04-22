from __future__ import annotations

from typing import Callable

import numpy as np


def random_action(rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet(np.ones(3)).astype(np.float32)


def equal_action() -> np.ndarray:
    return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)


def heuristic_action() -> np.ndarray:
    base = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    return base / base.sum()


def ddpg_without_jammer(action: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).copy()
    a[2] = 0.0
    s = a[0] + a[1]
    if s <= 1e-8:
        return np.array([0.5, 0.5, 0.0], dtype=np.float32)
    a[0] /= s
    a[1] /= s
    return a


def pso_optimize(
    objective: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    particles: int = 30,
    iters: int = 50,
    w: float = 0.7,
    c1: float = 1.4,
    c2: float = 1.4,
) -> np.ndarray:
    pos = rng.dirichlet(np.ones(3), size=particles).astype(np.float32)
    vel = np.zeros_like(pos)

    pbest = pos.copy()
    pbest_val = np.array([objective(x) for x in pbest], dtype=np.float32)
    gbest_idx = int(np.argmax(pbest_val))
    gbest = pbest[gbest_idx].copy()
    gbest_val = float(pbest_val[gbest_idx])

    for _ in range(iters):
        r1 = rng.random((particles, 3), dtype=np.float32)
        r2 = rng.random((particles, 3), dtype=np.float32)
        vel = (
            w * vel
            + c1 * r1 * (pbest - pos)
            + c2 * r2 * (gbest.reshape(1, 3) - pos)
        )
        pos = np.maximum(pos + vel, 1e-8)
        pos = pos / np.maximum(pos.sum(axis=1, keepdims=True), 1.0)

        values = np.array([objective(x) for x in pos], dtype=np.float32)
        improved = values > pbest_val
        pbest[improved] = pos[improved]
        pbest_val[improved] = values[improved]

        idx = int(np.argmax(pbest_val))
        if float(pbest_val[idx]) > gbest_val:
            gbest_val = float(pbest_val[idx])
            gbest = pbest[idx].copy()

    return gbest.astype(np.float32)


def grid_search_optimize(
    objective: Callable[[np.ndarray], float], resolution: int = 21
) -> np.ndarray:
    best = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    best_val = -1e18
    grid = np.linspace(0.0, 1.0, resolution)
    for a1 in grid:
        for a2 in grid:
            if a1 + a2 > 1.0:
                continue
            aj = 1.0 - a1 - a2
            a = np.array([a1, a2, aj], dtype=np.float32)
            val = objective(a)
            if val > best_val:
                best_val = val
                best = a
    return best.astype(np.float32)
