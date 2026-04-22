from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    seed: int = 42
    p_max: float = 10.0
    noise_power: float = 1.0
    r1_min: float = 1.0
    r2_min: float = 1.0
    episode_steps: int = 100

    # Large-scale fading coefficients
    l1: float = 0.8
    l2: float = 1.4
    le: float = 1.0
    lj1: float = 0.5
    lj2: float = 0.6
    lje: float = 1.6

    # Reward penalties
    lambda_qos: float = 5.0
    lambda_sic: float = 5.0
    lambda_power: float = 10.0

    # DDPG hyperparameters
    state_dim: int = 12
    action_dim: int = 3
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 128
    replay_size: int = 100_000
    warmup_steps: int = 1_000
    train_episodes: int = 3000

    # Evaluation settings
    eval_episodes: int = 200
    secrecy_outage_threshold: float = 0.1
    grid_resolution: int = 21
    pso_particles: int = 30
    pso_iters: int = 50
