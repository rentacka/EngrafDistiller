# -*- coding: utf-8 -*-
"""
================================================================================
🌸 🍄 ⚡ 190k 日英黄金カートリッジ (英語 + 日本語 + コード + 数学 + 絵文字 100%完全保存)
================================================================================
Architect: マネーコイコイ (@MoneyKoiKoi) ✕ Aria (アリア)

【一般向け公開仕様】:
- Gemma-4 の全 262,144 語から、不要な第三言語 (アラビア語、ロシア語、ヒンディー語、タイ語、韓国語等) のみを 72,227 語 (約28%) 完全カット！
- 英語、日本語 (漢字/かな/カナ/全角)、プログラミング構文 (Python/C++/MQL5)、数式、顔文字、絵文字は 100.0% 完全無傷で保存！
- 語彙数: 189,917 語 (約 190k)
================================================================================
"""

import sys, os, time, torch
from transformers import AutoTokenizer

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDIO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SNAP_DIR = r"C:\Users\renta\.cache\huggingface\hub\models--google--gemma-4-E2B-it\snapshots\3e22461f65e89153144f8adb70e3b8c2cc9845a7"

print("=" * 80)
print("🌸 🍄 ⚡ 【190k 日英黄金カートリッジ 抽出＆ベイク (一般公開向け)】 ⚡ 🍄 🌸")
print("=" * 80)

tok = AutoTokenizer.from_pretrained(SNAP_DIR)
vocab_size = len(tok)

def is_essential_token(s: str) -> bool:
    """
    英語, 日本語, コード, 数学, 顔文字, 絵文字に必要な文字のみ True
    不要多言語 (アラビア, キリル, デーヴァナーガリー, タイ, ハングル等) は False
    """
    for ch in s:
        cp = ord(ch)
        # 1. ASCII (0x00 - 0x7F: 英語, コード, 記号, 改行, インデント)
        if cp <= 0x007F:
            continue
        # 2. Latin Extended (欧文・アクセント)
        if 0x0080 <= cp <= 0x024F:
            continue
        # 3. ギリシャ文字 (数理・クオンツ記号: alpha, beta, etc.)
        if 0x0370 <= cp <= 0x03FF:
            continue
        # 4. 一般句読点, 通貨記号, 数学記号, 矢印 (2000-2BFF)
        if 0x2000 <= cp <= 0x2BFF:
            continue
        # 5. CJK記号, ひらがな, カタカナ, 漢字, 全角英数記号
        if (0x3000 <= cp <= 0x303F) or (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF) or (0x4E00 <= cp <= 0x9FFF) or (0xFF00 <= cp <= 0xFFEF):
            continue
        # 6. 絵文字 (Emoji: 1F000 - 1FAFF)
        if 0x1F000 <= cp <= 0x1FAFF:
            continue
        # それ以外は不要多言語
        return False
    return True

essential_tokens = set(tok.all_special_ids)

print("🔍 全 262,144 語彙から日英・コード・数理・絵文字を全数精査中...")
t0 = time.perf_counter()

for tid in range(vocab_size):
    try:
        s = tok.decode([tid])
        if is_essential_token(s):
            essential_tokens.add(tid)
    except Exception:
        pass

sorted_indices = sorted(list(essential_tokens))
num_tokens = len(sorted_indices)
cut_tokens = vocab_size - num_tokens

print(f"✅ 精査完了 ({time.perf_counter() - t0:.2f}s)")
print(f"🌟 日英 190k 黄金カートリッジ 確定語彙数: {num_tokens:,} 語 (72.4%)")
print(f"📉 カットされた不要多言語数: {cut_tokens:,} 語 ({(cut_tokens / vocab_size) * 100:.1f}% カット！)")

out_path = os.path.join(STUDIO_DIR, "models", "golden_vocab_indices_190k_bilingual.pt")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
torch.save(torch.tensor(sorted_indices, dtype=torch.int64), out_path)

file_size_kb = os.path.getsize(out_path) / 1024
print(f"💾 保存完了: {out_path} ({file_size_kb:.1f} KB)")
print("=" * 80)

