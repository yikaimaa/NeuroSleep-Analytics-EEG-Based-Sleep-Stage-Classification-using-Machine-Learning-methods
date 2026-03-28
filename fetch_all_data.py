"""
Sleep-EDF 数据下载脚本
======================
从 PhysioNet 下载 Sleep-EDF Expanded 数据集到 data/sleep-edf/

用法: conda activate mne && python fetch_all_data.py
"""

import os
import shutil
from pathlib import Path

import mne

# ─── 配置 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent / "data" / "sleep-edf"
N_SUBJECTS = int(os.getenv("N_SUBJECTS_FETCH", 20))  # Default 20, override for CI


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    print("=" * 60)
    print("Sleep-EDF Data Fetcher")
    print(f"Subjects: {N_SUBJECTS}")
    print(f"Output: {BASE_DIR}")
    print("=" * 60)

    for subj in range(N_SUBJECTS):
        print(f"\n  Subject {subj}...")
        try:
            # MNE 下载到 ~/mne_data/physionet-sleep-data/
            files = mne.datasets.sleep_physionet.age.fetch_data(
                subjects=[subj], recording=[1]
            )[0]

            # 复制到项目 data/ 目录（如果还没有）
            for src in files:
                dst = BASE_DIR / Path(src).name
                if not dst.exists():
                    shutil.copy2(src, dst)
                    print(f"    copied {Path(src).name}")
                else:
                    print(f"    [skip] {Path(src).name} already exists")

        except Exception as e:
            print(f"    [ERROR] {e}")

    # 统计
    edf_files = list(BASE_DIR.glob("*.edf"))
    total_size = sum(f.stat().st_size for f in edf_files)
    print(f"\n{'=' * 60}")
    print(f"Done! {len(edf_files)} files, {total_size / 1e6:.0f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
