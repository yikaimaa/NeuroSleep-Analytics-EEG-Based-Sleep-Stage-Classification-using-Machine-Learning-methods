"""
Single-Subject EDA: Subject 0 (SC4001E0)
========================================
This script performs Exploratory Data Analysis (EDA) on a single subject from the Sleep-EDF dataset.
It generates three key visualizations:
1. Sleep Stage Distribution (Pie/Bar chart)
2. Whole-night Multitaper Spectrogram aligned with Hypnogram
3. Power Spectral Density (PSD) by Sleep Stage (Welch's method)
"""

import os
import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置路径
DATA_DIR = '/Users/yimingshen20000719/NeuroSleep-Analytics-using-Machine-Learning-methods/data/sleep-edf'
PSG_FILE = os.path.join(DATA_DIR, 'SC4001E0-PSG.edf')
HYP_FILE = os.path.join(DATA_DIR, 'SC4001EC-Hypnogram.edf')
OUT_DIR = '/Users/yimingshen20000719/NeuroSleep-Analytics-using-Machine-Learning-methods/data visualization'

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("Loading data for Subject 0...")
    raw = mne.io.read_raw_edf(PSG_FILE, preload=True, verbose=False)
    annot = mne.read_annotations(HYP_FILE)
    
    # 过滤掉末尾的 'Sleep stage ?' 并且截断 raw 数据
    annot = mne.Annotations(
        onset=[o for o, d in zip(annot.onset, annot.description) if d != 'Sleep stage ?'],
        duration=[dur for dur, d in zip(annot.duration, annot.description) if d != 'Sleep stage ?'],
        description=[d for d in annot.description if d != 'Sleep stage ?']
    )
    raw.set_annotations(annot, emit_warning=False)
    
    # 裁剪 raw 数据长度匹配 annot 的真实结尾
    valid_duration = annot.onset[-1] + annot.duration[-1]
    raw.crop(tmax=min(valid_duration, raw.times[-1]))
    
    # 映射字典：将 R&K 阶段映射到现代 AASM 标准 (N3/N4 -> N3)
    annotation_desc_2_event_id = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4
    }
    event_id = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
    
    # 将 Annotations 切分成 30 秒的 epochs
    events, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id, chunk_duration=30., verbose=False)
    
    # 创建 Epochs 对象 (只提取核心通道 EEG Fpz-Cz 来做分析)
    tmax = 30. - 1. / raw.info['sfreq']  # 30秒
    epochs = mne.Epochs(raw=raw, events=events, event_id=event_id, 
                        tmin=0., tmax=tmax, baseline=None, preload=True, 
                        picks=['EEG Fpz-Cz'], verbose=False)
    
    print(f"Total 30s epochs created: {len(epochs)}")
    
    # 1. 睡眠阶段分布图 (Class Distribution)
    print("Plotting Class Distribution...")
    stage_counts = [len(epochs[stage]) for stage in event_id.keys()]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(event_id.keys(), stage_counts, color=['#ef4444', '#93c5fd', '#60a5fa', '#2563eb', '#f59e0b'])
    ax.set_title('Subject 0: Sleep Stage Distribution (30s Epochs)', fontsize=14)
    ax.set_ylabel('Number of Epochs')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}\\n({height/len(epochs)*100:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sub0_class_distribution.png'), dpi=150)
    plt.close()
    
    # 2. 计算并绘制各阶段的功率谱密度 (PSD by Stage)
    print("Plotting Power Spectral Density (PSD) by Stage...")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Wake': '#ef4444', 'N1': '#93c5fd', 'N2': '#60a5fa', 'N3': '#2563eb', 'REM': '#f59e0b'}
    
    for stage, color in colors.items():
        # 获取当前阶段的所有 epoch 数据并计算 PSD (Welch method)
        spectrum = epochs[stage].compute_psd(method='welch', fmin=0.5, fmax=30., n_fft=2048, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)
        # 求所有 epoch 的平均 PSD，并将单位转为 10*log10 (dB)
        psd_mean = 10 * np.log10(np.mean(psds, axis=0).squeeze())
        ax.plot(freqs, psd_mean, color=color, label=stage, linewidth=1.5)

    # 标出一些关键生理特征频段
    ax.axvspan(0.5, 4, color='gray', alpha=0.1, label='Delta (0.5-4 Hz)')
    ax.axvspan(12, 14, color='yellow', alpha=0.1, label='Sleep Spindles (12-14 Hz)')
    ax.axvspan(8, 12, color='green', alpha=0.1, label='Alpha (8-12 Hz)')
    
    ax.set_title('Subject 0: Power Spectral Density by Sleep Stage (EEG Fpz-Cz)', fontsize=14)
    ax.set_ylabel('Power (dB/Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim(0.5, 30)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sub0_psd_by_stage.png'), dpi=150)
    plt.close()
    
    print("EDA Complete. Plots saved to data visualization/ folder.")

if __name__ == '__main__':
    main()
