# simforge

Dynamic system modeling & simulation toolkit. 动态系统建模与仿真工具库

## 当前功能：直流电机速度环系统辨识

从采样率、PWM 输出、编码器输出三样数据建立电机传递函数：

**G(s) = K · e^(−t_d·s) / (τ·s + 1)** — 得到 K、τ、t_d 三个参数（含 95% 置信区间）+ 五层验证报告。

### 快速开始

```bash
# 环境：conda env simforge (Python 3.11, numpy/scipy/matplotlib/plotly/pytest)
python examples/simulate_motor.py                 # 生成仿真数据（已知真值）
python examples/motor_speed_id.py                 # 辨识 -> output/motor_sim/report.html
python -m pytest tests -q                         # 12 项测试（端到端参数恢复回归）
```

真实数据：`python examples/motor_speed_id.py --input your_run.csv --holdout 50`

### 项目结构

```
simforge/id/          系统辨识核心
  io.py               CSV 加载/保存/数据体检
  excitation.py       激励序列设计（多阶跃 + 留出段 + PRBS）
  preprocess.py       实验分段、留出索引
  arx.py              ARX(1,1)+纯滞后 线性最小二乘（主方法，含置信区间）
  stepfit.py          速度域输出误差 PEM 非线性拟合（交叉验证）
  validate.py         VAF / Ljung-Box / 互相关 / 频响估计 / 10 项验收清单
  pipeline.py         端到端流程编排
  report.py           报告输出：8 张图 PNG + 交互式 HTML + model.json + report.md
  simulator.py        合成数据生成器（真值已知，用于算法自检）
examples/             仿真数据生成、端到端辨识脚本
tests/                pytest 回归测试
docs/motor_id_guide.md  采集规范 + 使用说明 + 验收标准
```

### 输出（每次辨识）

- `report.html` — 交互式报告（可缩放/悬停），含模型参数、10 项验收清单、全部图表
- `report.md` + 8 张 PNG — T1 数据总览 / T2 拟合 / T3 留出预测 / T4 残差 / T5 双方法对比 / F1-F2 伯德图 / F3 残差谱
- `model.json` — K、τ、t_d、离散形式、置信区间、验证指标、工况条件

采集规范与验收标准见 [docs/motor_id_guide.md](docs/motor_id_guide.md)。
