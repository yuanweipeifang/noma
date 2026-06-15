# 实验结果索引

本文档记录各类实验输出的保存位置。

## 当前输出目录

- `outputs/main/`：主实验的训练日志、模型权重、指标表和基线对比图。
- `outputs/extended/`：扩展实验结果，包括固定测试集评估、延迟分析、`p_max` 扫描、奖励消融和动作行为分析。
- `outputs/qos_sensitivity/`：由 `qos_sensitivity_experiment.py` 生成的 `QoS` 门限敏感性实验结果。
- `outputs/sic_feasible_rate/`：由 `sic_feasible_rate_experiment.py` 生成的 `SIC` 可行率实验和搜索消融对比结果。
- `outputs/additional_figures/`：临时绘图或补充分析生成的图。
- `outputs/legacy/results1/`：保留的历史实验结果。

## 补充图归档

以下文件已从根目录下的 `results/` 移动到 `outputs/additional_figures/`，仅做归档整理，没有删除任何结果：

- `fig_decision_time_comparison.png`
- `fig_legit_eaves_rate_comparison.png`
- `fig_qos_satisfaction_comparison.png`
- `fig_secrecy_outage_comparison.png`
- `fig_security_reliability_metrics.png`
- `fig_training_qos_convergence.png`
- `fig_training_secrecy_convergence.png`
