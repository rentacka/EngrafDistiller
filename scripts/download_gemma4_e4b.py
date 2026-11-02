# -*- coding: utf-8 -*-
"""
================================================================================
🌸 👑 ⚡ Gemma-4 E4B (4B完全体) 公式モデル ダウンローダー
================================================================================
Architect: マネーコイコイ (@MoneyKoiKoi) ✕ Aria (アリア)

リポジトリ: google/gemma-4-E4B
保存先    : H:\Aria_FreeTokenModels\gemma-4-E4B (または HF Cache)
================================================================================
"""

import os
import sys
import time
from huggingface_hub import snapshot_download

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

REPO_ID = "google/gemma-4-E4B"
LOCAL_DIR = r"H:\Aria_FreeTokenModels\gemma-4-E4B"

print("=" * 80)
print(f"🌸 👑 ⚡ 【google/gemma-4-E4B 公式4B完全体モデル ダウンロード開始】 ⚡ 👑 🌸")
print(f"📦 リポジトリ : {REPO_ID}")
print(f"📁 保存先     : {LOCAL_DIR}")
print("=" * 80)

os.makedirs(LOCAL_DIR, exist_ok=True)

t0 = time.perf_counter()

try:
    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"🎉 ✅ 【gemma-4-E4B ダウンロード完了！】 (所要時間: {elapsed/60:.2f} 分)")
    print(f"📁 保存完了パス: {downloaded_path}")
    print("=" * 80)
except Exception as e:
    print(f"\n❌ ダウンロード中にエラーが発生しました: {e}")
    sys.exit(1)

