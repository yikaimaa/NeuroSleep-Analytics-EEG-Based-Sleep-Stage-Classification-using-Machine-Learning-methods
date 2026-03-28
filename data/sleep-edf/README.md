# Sleep-EDF 数据说明

## 文件命名规则

每个 subject 有 **2 个 EDF 文件**：

```
SC40{XX}E0-PSG.edf        ← 多导睡眠信号文件
SC40{XX}E{Y}-Hypnogram.edf ← 睡眠分期标注文件
```

其中 `{XX}` = subject 编号（01, 02, ...），`{Y}` 是标注者代号（C/H 等）。

---

## 1. PSG.edf — 多导睡眠信号

整夜录制的生理信号，采样率 **100 Hz**，共 **7 个通道**：

| 通道 | 类型 | 单位 | 说明 |
|---|---|---|---|
| `EEG Fpz-Cz` | 脑电 | µV | 前额-头顶导联，捕捉**入睡/浅睡**特征（vertex sharp waves, spindles） |
| `EEG Pz-Oz` | 脑电 | µV | 顶枕导联，捕捉**α节律**（清醒闭眼）和**δ波**（深睡） |
| `EOG horizontal` | 眼电 | µV | 水平眼动，用于识别 **REM 睡眠**（快速眼动） |
| `Resp oro-nasal` | 呼吸 | a.u. | 口鼻气流，监测呼吸模式（与睡眠呼吸暂停相关） |
| `EMG submental` | 肌电 | µV | 下颌肌肉活动，**REM 时肌张力消失**是关键判据 |
| `Temp rectal` | 体温 | °C | 直肠温度，反映昼夜节律 |
| `Event marker` | 标记 | - | 刺激/系统事件标记 |

### 关键元数据
- **录制时长**: ~10-22 小时（包含入睡前后的清醒时段）
- **受试者信息**: 包含性别、年龄（嵌入在 EDF 头信息中）
- **录制日期**: 1989 年
- **数据量**: 每个文件 ~46-51 MB

---

## 2. Hypnogram.edf — 睡眠分期标注

由专家按 **R&K 标准** 标注，每 **30 秒** 一个 epoch 的睡眠阶段标签。

| 标签 | 含义 | 对应脑电特征 |
|---|---|---|
| `Sleep stage W` | **清醒 (Wake)** | α 波（8-12Hz），高肌电 |
| `Sleep stage 1` | **NREM-1 浅睡** | θ 波（4-7Hz），vertex sharp waves |
| `Sleep stage 2` | **NREM-2 轻睡** | Sleep spindles（12-14Hz），K-complexes |
| `Sleep stage 3` | **NREM-3 深睡** | δ 波（0.5-2Hz）占 20-50% |
| `Sleep stage 4` | **NREM-4 极深睡** | δ 波占 >50%（现代标准合并 3+4 为 N3） |
| `Sleep stage R` | **REM 快速眼动** | 低幅混合频率，快速眼动，肌张力消失 |
| `Sleep stage ?` | **未评分** | 信号质量差或录制首尾无效段 |

### Subject 0 的标注分布示例

| 阶段 | epoch 数 | 时长 |
|---|---|---|
| Wake | 12 | 6 分钟 |
| Stage 1 | 24 | 12 分钟 |
| Stage 2 | 40 | 20 分钟 |
| Stage 3 | 48 | 24 分钟 |
| Stage 4 | 23 | 11.5 分钟 |
| REM | 6 | 3 分钟 |
| ? | 1 | 0.5 分钟 |

> **注意**: Hypnogram 中的 epoch 是变长标注（相同阶段的连续 30s 合并为一条记录），不是每 30s 一行。MNE 会自动展开。

---

## 数据来源

- **数据集**: PhysioNet Sleep-EDF Expanded
- **链接**: https://physionet.org/content/sleep-edfx/1.0.0/
- **参考文献**: Kemp B, et al. Analysis of a sleep-and-event-related database. IEEE-BME 47(9):1185-1193 (2000)
