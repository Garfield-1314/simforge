# 通用 SISO 系统辨识指南

对任意**单输入单输出、线性定常**对象，从一段 (输入 u, 输出 y, 采样率) 数据建立传递函数：

- **G(z)**：离散传递函数（精确，含纯滞后）
- **G(s)**：连续传递函数（双线性逆变换近似，纯滞后以 `e^{-td·s}` 因子单列）
- **特征方程**（z 域精确 / s 域极点映射 `s = ln z / Ts`）
- 极点/零点、DC 增益、稳定性、伯德图 + 全部验证指标

电机应用见 [motor_id_guide.md](motor_id_guide.md)；电机工程例子见 `examples/motor/`。

---

## 1. 数据要求

| 项 | 要求 | 原因 |
|---|---|---|
| CSV 格式 | `# fs: 500` 元数据 + 表头 `t_s,u,y` | 列别名自动识别（`input/output/voltage/...` 均可） |
| 采样率 | ≥ 10× 预期带宽 | 混叠与延迟分辨率 |
| 幅值 | 信号变化量 ≫ 量化/噪声 | 信噪比决定参数置信区间 |
| **激励带宽** | **n 阶系统需要 2n 阶持续激励** | 长时间"阶跃+保持"对二阶以上动态激励不足（矩条件病态、阶次误判）|
| 激励推荐 | 一阶：多电平阶跃即可；**二阶及以上：`excitation.random_dwell()` / PRBS** | 宽带激励使 AIC/BIC 与工具变量矩条件良态 |
| 时长 | 每段 ≥ 5× 时间常数，总长 ≥ 100× 最慢时间常数 | 稳态增益与低频动态的辨识精度 |

## 2. 使用方法

```bash
# 自动定阶（默认 max-order=3，BIC）
python examples/identify.py --input data.csv --max-order 3

# 留出某输入幅值做 out-of-sample 验证 + 覆盖单位标签
python examples/identify.py --input data.csv --holdout 10 \
    --input-unit V --output-unit degC --outdir output/run1
```

Python API：

```python
from simforge.id import load_system_csv, run_identification, write_report, Thresholds

data = load_system_csv("data.csv")
res = run_identification(data, holdout_duty=None, max_order=3,
                         thresholds=Thresholds(vaf_train=97.0))   # 阈值可配
print(res.arx.to_tf().g_s_str(), res.arx.to_tf().char_eq_s())
write_report(res, "output/run1")
```

## 3. 方法（两级辨识 + 一致准则）

1. **Stage-1：ARX(n,m,d) 网格搜索**（`arx.identify_arx_auto`）
   - 最小二乘拟合，BIC（默认，抗大数据过拟合）或 AIC 选阶
   - 估计器 `method="auto"`：同时用 **LS** 与 **IV**（工具变量，对数间隔输入滞后作工具）拟合，
     按两者系数差异自动判别噪声类型——差异小 = 过程噪声（LS 低方差），差异大 = 输出测量噪声（IV 无偏）
2. **Stage-2：结构选择（`stepfit.select_structure_pem`）**
   - 对每个候选 (n, m, d)（两种 Stage-1 各自的最优延迟 ±2）做**输出误差 PEM 精修**：
     在训练行的随机 **70% 上拟合、30% 验证残差上排序**（一致的 OE 准则——
     样本内噪声投影不外推，真动态外推得动，从而避免误差变量偏置与过拟合选阶）
   - **5% RSS 简单性平局规则**：阶次间验证 RSS 差 <5% 视为噪声级平局，取低阶
   - 胜出结构在全部训练行上重拟合 ARX（协方差/CI）与 PEM（最终系数）
3. **报告的参数**：ARX（含 95%CI、协方差）+ PEM（动力学主估计，CI 由 Jacobian 得）

## 4. 验收清单（11 项，阈值 `Thresholds` 可配置）

1. 训练段 VAF ≥ 95%
2. 留出段 VAF ≥ 90%（需 `--holdout`）
3. 留出段稳态误差 ≤ 3%
4. Ljung-Box 白噪声 p > 0.05（**从滞后 n+1 起检验**：测量噪声下一步预测误差是 MA(n)，从滞后 1 检验必然误报）
5. 残差×输入互相关在 95% 置信带内
6. 双方法（ARX/PEM）增益与时间常数差 ≤ 5%
7. 参数 95%CI 宽度 ≤ 20%（PEM 为主）
8. 增益与理论参考偏差 ≤ 15%（元数据 `ref_gain`，无则跳过）
9. 参数恢复误差 ≤ 2%（仅仿真数据，元数据 `true_k/true_tau_s`）
10. **金标准复测**（人工）：换一个未用过的输入幅值录 5 s，预测误差 ≤ 5%
11. 阶次选择置信：跨阶次准则差 ≥ 10（强证据），或在 5% 平局容差内取低阶（通过并标注）

## 5. 产物（`output/<name>/`）

| 文件 | 内容 |
|---|---|
| `report.html` | 交互式报告（11 图可缩放悬停 + 验收清单）|
| `report.md` + 11 张 PNG | T1 数据总览 / T2 拟合 / T3 留出 / T4 残差 / T5 双方法 / F1-F2 伯德 / F3 残差谱 / M1 摘要卡 / M2 零极点 / M3 阶次扫描 |
| `model.json` | `transfer_function`、`characteristic_equation`、`poles`、`order_selection`、`validation`、`checks`、工况 |
| `metrics.json` | 全部数值指标 |

## 6. 诚实边界

- 仅覆盖 **SISO、线性、时不变**；MIMO / 强非线性 / 时变不在范围
- 高阶 `G(s)` 是双线性近似（报告注明）；**`G(s)` 的极点映射与 `G(z)` 是精确的**——分析与设计优先用 G(z)、极点与伯德图
- 模型是特定工况的局部描述：换负载/激励幅值区间可能失效
- 欠激励数据下阶次与延迟不可辨识——检查清单会标注"存疑"，此时应回到第 1 节补采数据
