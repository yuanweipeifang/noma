# 项目结构

本项目按照源码、实验入口、说明文档和实验输出进行组织。

```text
noma_RL/
├── README.md
├── requirements.txt
├── docs/
│   ├── project_structure.md
│   └── results_index.md
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
│       ├── qos_sensitivity_experiment.py
│       ├── sac.py
│       ├── sic_feasible_rate_experiment.py
│       └── td3.py
└── outputs/
    ├── additional_figures/
    ├── qos_sensitivity/
    ├── sic_feasible_rate/
    ├── main/
    ├── extended/
    └── legacy/
        └── results1/
```

- `src/noma_rl/`：可复用的环境、配置、基线方法和强化学习智能体。
- `scripts/`：命令行实验入口。
- `outputs/main/`：主实验生成的日志、模型、指标表和图。
- `outputs/extended/`：扩展实验生成的固定测试集、延迟、功率扫描、奖励消融和动作行为分析结果。
- `outputs/qos_sensitivity/`：`QoS` 门限敏感性实验的 `CSV` 和图。
- `outputs/sic_feasible_rate/`：`SIC` 可行率实验的 `CSV` 和图。
- `outputs/additional_figures/`：主输出目录之外生成的补充图。
- `outputs/legacy/`：历史实验输出归档，不属于当前默认运行流程。

生成的实验输出和 `Python` 缓存默认不纳入版本管理。可复用代码和说明文档应放在 `outputs/` 之外。
