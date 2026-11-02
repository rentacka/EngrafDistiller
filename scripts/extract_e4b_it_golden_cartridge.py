# -*- coding: utf-8 -*-
"""
🌸 👑 ⚡ Gemma-4 E4B-it から 190k 黄金エングラムを無傷抽出＆ベイク (VRAM 0MB / CPU 安全実行)
"""
import os, sys, time, torch
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
from safetensors import safe_open

MODEL_DIR = r"H:\llama.cpp\models\lmstudio-community\gemma-4-E4B-it"
STUDIO_DIR = r"j:\Antigravity\AriaQuantTrader\Aria_FreeToken_Windows_Portable_Complete_Master_Edition\37_Aria_Physarum_Gemma4B_Transformer_Distiller_Studio"
indices_path = os.path.join(STUDIO_DIR, "models", "golden_vocab_indices_190k_bilingual.pt")
out_cartridge_path = os.path.join(STUDIO_DIR, "models", "gemma4_e4b_it_190k_golden_cartridge.pt")

print("=" * 80)
print("🌸 👑 ⚡ 【Gemma-4 E4B-it ➔ 190k 黄金エングラム抽出＆ベイク (CPU)】 ⚡ 👑 🌸")
print("=" * 80)

safetensors_path = os.path.join(MODEL_DIR, "model.safetensors")
indices_190k = torch.load(indices_path, map_location="cpu")
print(f"📦 190k 黄金インデックスロード完了: {len(indices_190k):,} 語")

t0 = time.perf_counter()
with safe_open(safetensors_path, framework="pt", device="cpu") as f:
    # Gemma-4 の埋め込みキーを探索
    embed_key = None
    for k in f.keys():
        if "embed_tokens" in k:
            embed_key = k
            break
    
    print(f"🔍 埋め込みキー発見: {embed_key}")
    w_embed_full = f.get_tensor(embed_key)
    print(f"📊 元の埋め込み形状: {w_embed_full.shape} ({w_embed_full.dtype})") # [262144, 2560]

    # 190k 黄金カートリッジのスライス抽出
    print("🔨 190k 黄金スライスを抽出中...")
    w_embed_190k = w_embed_full[indices_190k].contiguous()
    print(f"✅ 190k 黄金エングラム形状: {w_embed_190k.shape} ({w_embed_190k.dtype})")

# 保存 (bfloat16)
torch.save({
    "name": "gemma4_e4b_it_190k_golden",
    "title": "Gemma-4 E4B-it 190k Bilingual Golden Cartridge (2560D)",
    "vocab_size": len(indices_190k),
    "hidden_size": w_embed_190k.shape[1],
    "indices": indices_190k,
    "weight": w_embed_190k,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
}, out_cartridge_path)

size_mb = os.path.getsize(out_cartridge_path) / (1024 * 1024)
elapsed = time.perf_counter() - t0
print(f"💾 190k 黄金カートリッジ保存完了: {out_cartridge_path} ({size_mb:.1f} MB / {elapsed:.2f}s)")
print("=" * 80)

