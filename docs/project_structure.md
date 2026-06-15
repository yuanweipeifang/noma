# Project Structure

The repository is organized around source code, runnable experiment entry points, documentation, and generated outputs.

```text
noma_RL/
├── README.md
├── requirements.txt
├── docs/
│   └── project_structure.md
├── scripts/
│   ├── run_experiment.py
│   └── run_extended_studies.py
├── src/
│   └── noma_rl/
│       ├── __init__.py
│       ├── baselines.py
│       ├── config.py
│       ├── ddpg.py
│       ├── env.py
│       ├── sac.py
│       └── td3.py
└── outputs/
    ├── main/
    ├── extended/
    └── legacy/
        └── results1/
```

- `src/noma_rl/`: reusable environment, configuration, baseline methods, and RL agents.
- `scripts/`: command-line experiment entry points.
- `outputs/main/`: generated files from the main benchmark.
- `outputs/extended/`: generated files from the extended studies.
- `outputs/legacy/`: preserved historical outputs that are not part of the current default workflow.

Generated outputs and Python caches are ignored by Git. Keep code and reusable documentation outside `outputs/`.
