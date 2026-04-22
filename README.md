# 基于深度强化学习的友好干扰辅助 NOMA 物理层安全传输

本项目实现了一个最简可运行的下行 NOMA 安全传输仿真框架，核心目标是通过联合分配 `P1/P2/PJ` 最大化系统保密容量之和，并与传统基线进行对比。

## 功能概览

- 系统模型：`2` 个合法用户 (`U1/U2`) + `1` 个友好干扰节点 (`J`) + `1` 个被动窃听者 (`E`)
- 信道模型：路径损耗 + 瑞利小尺度衰落
- 强化学习算法：DDPG（连续动作功率分配）
- 基线算法：
  - Random
  - Equal
  - Heuristic (`0.5:0.3:0.2`)
  - PSO
  - Grid Search
  - DDPG without Jammer（消融）
- 输出内容：
  - 训练日志
  - 算法对比指标表
  - 收敛曲线与对比柱状图

## 目录结构

```text
noma/
├── requirements.txt
├── README.md
├── src/
│   └── noma_rl/
│       ├── __init__.py
│       ├── config.py
│       ├── env.py
│       ├── ddpg.py
│       └── baselines.py
├── scripts/
│   └── run_experiment.py
└── results/
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行实验

快速验证（建议先跑这个）：

```bash
python scripts/run_experiment.py --train-episodes 80 --episode-steps 40 --eval-episodes 30
```

完整训练（按建议参数）：

```bash
python scripts/run_experiment.py --train-episodes 3000 --episode-steps 100 --eval-episodes 200
```

## 输出文件说明

运行后在 `results/` 目录下生成：

- `config_used.json`：实验参数记录
- `training_log.csv`：DDPG 训练日志（可用于收敛曲线）
- `metrics_table.csv`：各算法对比指标表
- `fig_training_convergence.png`：训练收敛图
- `fig_algorithm_comparison.png`：算法保密容量对比图

## 指标定义

- 主指标：平均保密容量之和
- 辅助指标：
  - 平均合法总速率
  - 平均窃听总速率
  - QoS 满足率
  - 保密中断概率
  - 单次决策时间

## 说明

- 奖励函数中包含 QoS、SIC 可行性、功率约束惩罚项。
- 动作使用三维连续分配系数并归一化到总功率约束内。
- 该代码优先保证“可复现 + 可扩展 + 可直接出图表”，便于课程实验报告撰写。
