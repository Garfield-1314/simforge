# simforge

Dynamic system modeling & simulation toolkit. 动态系统建模与仿真工具库

## 当前功能：SISO 线性系统辨识（含电机速度环）

从一段 (输入 u, 输出 y) 数据建立传递函数 **G(z)（精确）/ G(s) / 特征方程 / 零极点**，
自动定阶（BIC + 输出误差一致准则 + 5% 平局取简），11 项可配置验收清单。

### 快速开始

```bash
# 环境：conda env simforge (Python 3.11, numpy/scipy/matplotlib/plotly/pytest)

# 通用对象：自动定阶辨识 -> output/<name>/report.html
python examples/identify.py --input data.csv --max-order 3

# 电机工程例子（自包含，真值已知可自检）
python examples/motor/simulate_motor.py       # 生成仿真数据
python examples/motor/motor_speed_id.py       # 辨识 -> 11 图 + 报告

python -m pytest tests -q                     # 回归测试
```

### 项目结构

```
simforge/id/            系统辨识核心
  io.py                 SystemData (u/y + 单位)、CSV 加载/保存/体检
  excitation.py         激励设计（多阶跃序列 + random_dwell/PRBS 宽带激励）
  preprocess.py         实验分段、留出索引
  tf.py                 传递函数：G(z)/G(s)/特征方程/零极点/频响
  arx.py                ARX(n,m,d) 网格 + BIC/AIC 定阶；LS/IV/auto 估计器
  stepfit.py            输出误差 PEM + 两级结构选择（一致准则 + 平局取简）
  validate.py           Thresholds 可配阈值 + VAF/Ljung-Box/互相关/11 项验收
  pipeline.py           端到端编排
  report.py             11 张图 PNG + 交互 HTML + model.json + report.md
  simulator.py          合成数据生成（电机 + 通用 LTI/二阶系统，真值已知）
examples/identify.py    通用 CLI
examples/motor/         电机工程例子（自包含：脚本 + 数据 + 报告）
tests/motor/            电机回归测试
tests/system/           通用系统测试（二阶恢复、阈值配置、列名兼容）
docs/system_id_guide.md 通用辨识指南（激励/方法/验收/边界）
docs/motor_id_guide.md  电机采集规范与验收标准
```

### 输出（每次辨识）

- `report.html` — 交互式报告（可缩放/悬停）：参数、特征方程、11 项验收、全部图表
- `report.md` + 11 张 PNG — T1-T5 时域 / F1-F3 频域 / M1 摘要卡 / M2 零极点 / M3 阶次扫描
- `model.json` — `transfer_function`、`characteristic_equation`、`poles`、`order_selection`、`validation`、`checks`

指南：[docs/system_id_guide.md](docs/system_id_guide.md)（通用）、
[docs/motor_id_guide.md](docs/motor_id_guide.md)（电机采集规范）。
