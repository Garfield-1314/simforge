# 电机工程例子：直流电机速度环辨识

完整演示从"采样率 + PWM + 编码器"三样数据建立电机传递函数：

**G(s) = K · e^(−t_d·s) / (τ·s + 1)** — 3 个参数 + 11 项验收清单 + 12 项回归测试。

本目录自包含：脚本、生成的 CSV、辨识报告都在这里（`data/` 与 `output/` 不入库，可随时重新生成）。

## 运行

```bash
# 环境: conda env simforge (Python 3.11)
python examples/motor/simulate_motor.py     # 1) 生成仿真数据（真值已知: K=50, τ=50ms, td=4ms）
python examples/motor/motor_speed_id.py     # 2) 辨识 -> output/motor_sim/report.html
python -m pytest tests/motor -q             # 3) 回归测试
```

真实数据：

```bash
python examples/motor/motor_speed_id.py --input your_run.csv --holdout 50
```

CSV 格式与采集规范见 [../../docs/motor_id_guide.md](../../docs/motor_id_guide.md)
（悬空装轮、500 Hz、4096 counts/rev、多阶跃激励 + 留出段）。

## 预期结果（真值对照）

| 指标 | 阈值 | 实测 |
|---|---|---|
| 参数恢复 K / τ / t_d | ≤ 2% / ≤ 2% / ≤ 1 采样 | 0.58% / 1.20% / 2 ms（延迟 ±1 采样内）|
| 训练段 VAF | ≥ 95% | 99.9% |
| 留出段 VAF / 稳态误差 | ≥ 90% / ≤ 3% | 99.6% / 0.22% |
| Ljung-Box 白噪声 p（滞后 ≥ n+1） | > 0.05 | 0.24 |
| 双方法 K/τ 差异 | ≤ 5% | 0.01% / 2.6% |

11 项验收清单：9 项自动判定通过（第 8 项需提供理论 `ref_gain`），第 10 项金标准复测需人工，第 11 项在 `--max-order 1` 下跳过。

> 说明：延迟在 ±1 个采样内不可辨识（阶跃激励的已知局限），因此 t_d 的 2 ms 偏差属正常，检查项 9 以"≤1 采样"为准。

## 产物（`output/motor_sim/`）

- `report.html` — 交互式报告（可缩放/悬停）
- `report.md` + 11 张 PNG — T1 数据总览 / T2 拟合 / T3 留出预测 / T4 残差 / T5 双方法对比 / F1-F2 伯德图 / F3 残差谱 / **M1 模型摘要卡 / M2 零极点 / M3 阶次扫描**
- `model.json` — 传递函数 G(s)/G(z)、特征方程、极点、离散 (a,b,d)、95% 置信区间、阶次选择、验证指标、工况条件
- `metrics.json` — 全部数值指标

## 文件说明

| 文件 | 作用 |
|---|---|
| `simulate_motor.py` | 仿真生成器：一阶真值 + 过程噪声 + 4096 cpr 量化，按采集规范的激励序列造数 |
| `motor_speed_id.py` | 端到端：加载 → 分段 → ARX+PEM 双方法 → 五层验证 → 报告 |
| `data/motor_sim.csv` | 生成的数据（gitignore，可重建） |
| `output/` | 辨识报告（gitignore，可重建） |
