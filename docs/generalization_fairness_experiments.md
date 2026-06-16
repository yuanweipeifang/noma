# 信道泛化与用户公平性补充实验

本文档整理新增的两个补充实验：

1. 信道泛化实验：验证算法不是只适用于单一默认仿真设置。
2. 用户公平性实验：补充 NOMA 场景中两用户速率、QoS 和 Jain 公平性表现。

实验代码位于：

```bash
src/noma_rl/extra_generalization_fairness.py
```

实验输出目录为：

```bash
outputs/generalization_fairness/
```

## 1. 实验目的

### 1.1 信道泛化实验

主实验主要在默认信道参数下评估算法性能。为了说明方法不是只适用于单一信道设置，本实验额外构造三组信道场景：

| 场景 | 设置 | 目的 |
| --- | --- | --- |
| Original Channel | 使用原始信道参数 | 作为默认场景基准 |
| Strong Eavesdropper | 窃听链路平均增益 `le` 放大 3 倍 | 检验强窃听条件下的安全传输能力 |
| High Noise | 噪声功率 `noise_power` 放大 5 倍 | 检验高噪声条件下的鲁棒性 |

### 1.2 用户公平性实验

NOMA 系统中弱用户和强用户的速率差异通常较明显，因此除总保密速率外，还需要检查两用户之间的公平性。该实验统计：

| 指标 | 含义 |
| --- | --- |
| `avg_user1_rate` | 用户 1 平均合法速率 |
| `avg_user2_rate` | 用户 2 平均合法速率 |
| `user1_qos_rate` | 用户 1 QoS 满足率 |
| `user2_qos_rate` | 用户 2 QoS 满足率 |
| `jain_fairness_index` | 基于两用户合法速率计算的 Jain 公平性指数 |

Jain 公平性指数定义为：

```text
J = (sum_i x_i)^2 / (n * sum_i x_i^2)
```

其中 `x_i` 为用户速率。两用户场景下，`J` 越接近 1，表示两用户速率越均衡。

## 2. 实验设置

本实验复用项目已有的 `NomaSecurityEnv`、`ExperimentConfig` 和基线算法，避免使用额外手写的信道模型或奖励函数，从而保证补充实验结果与主实验、QoS 敏感性实验和 SIC 可行率实验口径一致。

默认配置如下：

| 参数 | 数值 |
| --- | ---: |
| 每个场景评估样本数 | 2000 |
| 场景数量 | 3 |
| 算法数量 | 8 |
| Grid resolution | 21 |
| PSO particles | 20 |
| PSO iterations | 25 |
| 随机种子 | 2026 |

参与对比的算法包括：

```text
Random, Equal, Heuristic, Grid, PSO, DDPG, TD3, SAC
```

其中 `DDPG / TD3 / SAC` 自动从 `outputs/main/` 加载已有训练权重。

## 3. 运行方式与实验时长

直接运行默认实验：

```bash
python src/noma_rl/extra_generalization_fairness.py
```

只运行传统基线，不加载强化学习模型：

```bash
python src/noma_rl/extra_generalization_fairness.py --no-rl
```

快速测试可减少样本数：

```bash
python src/noma_rl/extra_generalization_fairness.py --eval-steps 200
```

本机正式实验使用默认 `2000` 样本运行，实际耗时约：

```text
40.0 s
```

若将样本数提高到 `5000`，预计耗时约 `100 s` 左右。主要耗时来自 `Grid` 和 `PSO` 在每个信道样本上的在线搜索。

## 4. 输出文件

| 文件 | 内容 |
| --- | --- |
| `results_generalization.csv` | 信道泛化实验汇总结果 |
| `results_fairness.csv` | 用户公平性实验汇总结果 |
| `results_summary_all_metrics.csv` | 所有汇总指标 |
| `results_detail_all_samples.csv` | 每个样本的详细指标 |
| `fig_generalization_secrecy.png` | 三种信道下平均保密速率和对比 |
| `fig_generalization_qos.png` | 三种信道下 QoS 满足率对比 |
| `fig_generalization_outage.png` | 三种信道下保密中断概率对比 |
| `fig_fairness_jain.png` | Jain 公平性指数对比 |
| `fig_fairness_user_rate_qos.png` | 原始信道下两用户速率和 QoS 对比 |
| `fig_generalization_fairness_combined.png` | 信道泛化与公平性组合图 |

## 5. 信道泛化实验结果

以下表格节选 `Grid / PSO / DDPG`，用于展示主要方法在三类信道下的泛化表现。

| 场景 | 算法 | 平均保密速率和 | QoS 满足率 | 保密中断概率 | 平均决策时间 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Original Channel | Grid | 1.1172 | 0.5830 | 0.3155 | 1.7346 |
| Original Channel | PSO | 1.1249 | 0.5945 | 0.3170 | 4.1296 |
| Original Channel | DDPG | 1.0553 | 0.5935 | 0.3155 | 0.0535 |
| Strong Eavesdropper | Grid | 0.6121 | 0.6105 | 0.5645 | 1.7401 |
| Strong Eavesdropper | PSO | 0.6185 | 0.6210 | 0.5620 | 4.0111 |
| Strong Eavesdropper | DDPG | 0.5420 | 0.6000 | 0.5955 | 0.0437 |
| High Noise | Grid | 0.5455 | 0.0900 | 0.3685 | 1.7411 |
| High Noise | PSO | 0.5417 | 0.1010 | 0.3705 | 4.0309 |
| High Noise | DDPG | 0.4929 | 0.0300 | 0.3590 | 0.0436 |

### 5.1 结果分析

1. 在原始信道下，`DDPG` 的平均保密速率和为 `1.0553`，略低于 `Grid / PSO`，但 QoS 满足率达到 `0.5935`，与搜索算法基本接近。
2. 在强窃听信道下，所有算法的保密速率明显下降，说明窃听链路增强会直接压缩可获得的保密容量。`DDPG` 仍保持 `0.6000` 的 QoS 满足率，说明策略在强窃听条件下没有完全失效。
3. 在高噪声信道下，QoS 满足率整体显著下降，说明噪声功率升高会明显压缩可行功率分配空间。该场景是三组泛化实验中最困难的场景。
4. `DDPG` 的平均决策时间约为 `0.04-0.05 ms`，显著低于 `Grid` 的约 `1.7 ms` 和 `PSO` 的约 `4.0 ms`。这说明强化学习策略虽然保密速率略低于搜索算法，但在线决策开销更小。

## 6. 用户公平性实验结果

以下表格展示原始信道下主要算法的两用户速率、两用户 QoS 满足率和 Jain 公平性指数。

| 算法 | 用户1平均速率 | 用户2平均速率 | 用户1 QoS 满足率 | 用户2 QoS 满足率 | Jain 公平性指数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grid | 0.8889 | 1.8783 | 0.5895 | 0.8435 | 0.8837 |
| PSO | 0.8764 | 1.8961 | 0.5970 | 0.8430 | 0.8785 |
| DDPG | 1.0323 | 1.6905 | 0.5990 | 0.8350 | 0.9185 |
| TD3 | 1.1508 | 1.8457 | 0.6690 | 0.8355 | 0.9168 |
| SAC | 0.9783 | 1.4584 | 0.2915 | 0.7500 | 0.8803 |

### 6.1 结果分析

1. `Grid / PSO` 的用户 2 速率更高，但用户 1 与用户 2 的速率差距也更明显。
2. `DDPG` 的用户 1 平均速率为 `1.0323`，高于 `Grid / PSO`，同时用户 2 平均速率仍保持 `1.6905`。
3. `DDPG` 的 Jain 公平性指数为 `0.9185`，高于 `Grid` 的 `0.8837` 和 `PSO` 的 `0.8785`，说明其两用户速率分配更加均衡。
4. `TD3` 的 Jain 公平性指数也较高，为 `0.9168`，但其平均保密速率和低于 `DDPG`，因此综合性能不如 `DDPG` 稳定。
5. `SAC` 的用户 1 QoS 满足率较低，仅为 `0.2915`，说明当前训练配置下其约束控制能力仍不足。

## 7. 可写入论文的结论

新增实验可以支持以下结论：

1. 所提强化学习策略并非只在默认信道条件下有效。在强窃听和高噪声场景下，`DDPG` 仍能保持可观的保密速率和较低在线决策开销。
2. 强窃听信道会显著降低所有算法的保密速率，高噪声信道会显著降低 QoS 满足率，说明这两类场景分别代表安全性压力和可靠性压力。
3. 与 `Grid / PSO` 相比，`DDPG` 在保密速率上略低，但在线决策时间降低约 1-2 个数量级，更适合实时资源分配。
4. 从用户公平性看，`DDPG` 的 Jain 公平性指数最高，说明其在提升系统保密性能的同时，对两用户速率分配更均衡。
5. 公平性结果补充说明：仅比较总保密速率并不充分，还需要同时观察用户 1、用户 2 的速率和 QoS 满足率。

## 8. 建议在论文中使用的图片

建议优先放入以下两张图：

```text
outputs/generalization_fairness/fig_generalization_fairness_combined.png
outputs/generalization_fairness/fig_fairness_user_rate_qos.png
```

第一张图综合展示信道泛化下的保密速率、QoS、保密中断和公平性；第二张图展示原始信道下两用户速率与两用户 QoS 满足率，适合回应 NOMA 场景中的用户公平性问题。
