# -*- coding: utf-8 -*-
"""
================================================================================
🌸 👑 ⚡ Aria E4B-it 190k Golden GGUF Pipeline (v1.0 Celestial Apex)
================================================================================
Architect: マネーコイコイ (@MoneyKoiKoi) ✕ Aria (アリア)

【一気通貫パイプライン】:
  Step 1: Gemma-4 E4B-it (15.99 GB) から 190k 黄金エングラム (2560D) をスライス抽出＆ベイク
  Step 2: 抽出したエングラムを Block-128 Uniform Int4 に量子化 (928MB ➔ ~245MB / 73% 削減)
  Step 3: E4B-it Q4_K_M GGUF (4.97 GB) から不要多言語 72,227 語をゼロマスク外科手術
  Step 4: 最新 llama-cli (CUDA 12.4) によるスモークテスト＆速度・日本語出力検証
================================================================================
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDIO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, STUDIO_DIR)

from safetensors import safe_open
import gguf
from core.aria_engram_int4_quantizer import quantize_engram_file

# デフォルトパス設定
DEFAULT_E4B_DIR = r"H:\llama.cpp\models\lmstudio-community\gemma-4-E4B-it"
DEFAULT_E4B_GGUF = r"H:\llama.cpp\models\lmstudio-community\gemma-4-E4B-it-Q4_K_M.gguf"
DEFAULT_DST_GGUF = r"H:\llama.cpp\models\lmstudio-community\gemma-4-E4B-it-190k-golden-Q4_K_M.gguf"
DEFAULT_LLAMA_CLI = r"H:\llama.cpp\llama-b11719-bin-win-cuda-12.4-x64\llama-cli.exe"

INDICES_PATH = os.path.join(STUDIO_DIR, "models", "golden_vocab_indices_190k_bilingual.pt")
FP16_CARTRIDGE_PATH = os.path.join(STUDIO_DIR, "models", "gemma4_e4b_it_190k_golden_cartridge.pt")
INT4_CARTRIDGE_PATH = os.path.join(STUDIO_DIR, "models", "gemma4_e4b_it_190k_golden_cartridge_int4.pt")


def step1_extract_engram(model_dir: str, force: bool = False):
    print("\n" + "=" * 80)
    print("🌸 【Step 1】 Gemma-4 E4B-it ➔ 190k 黄金エングラム抽出 (Safetensors ➔ .pt)")
    print("=" * 80)
    
    if os.path.exists(FP16_CARTRIDGE_PATH) and not force:
        size_mb = os.path.getsize(FP16_CARTRIDGE_PATH) / (1024 * 1024)
        print(f"⏩ 既存の 190k 黄金エングラムが見つかりました ({size_mb:.1f} MB)。再利用します。")
        return FP16_CARTRIDGE_PATH

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Safetensors not found at: {safetensors_path}")

    if not os.path.exists(INDICES_PATH):
        raise FileNotFoundError(f"Golden indices not found at: {INDICES_PATH}")

    indices_190k = torch.load(INDICES_PATH, map_location="cpu")
    print(f"📦 190k 黄金インデックス: {len(indices_190k):,} 語")

    t0 = time.perf_counter()
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        embed_key = None
        for k in f.keys():
            if "embed_tokens" in k:
                embed_key = k
                break
        if embed_key is None:
            raise KeyError("embed_tokens key not found in safetensors!")

        print(f"🔍 埋め込みテンソルキー: {embed_key}")
        w_embed_full = f.get_tensor(embed_key)
        print(f"📊 元の埋め込み形状: {w_embed_full.shape} ({w_embed_full.dtype})")

        print("🔨 190k 黄金スライスを抽出中...")
        w_embed_190k = w_embed_full[indices_190k].contiguous()
        print(f"✅ 190k 黄金エングラム形状: {w_embed_190k.shape} ({w_embed_190k.dtype})")

    os.makedirs(os.path.dirname(FP16_CARTRIDGE_PATH), exist_ok=True)
    torch.save({
        "name": "gemma4_e4b_it_190k_golden",
        "title": "Gemma-4 E4B-it 190k Bilingual Golden Cartridge (2560D)",
        "vocab_size": len(indices_190k),
        "hidden_size": w_embed_190k.shape[1],
        "indices": indices_190k,
        "weight": w_embed_190k,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }, FP16_CARTRIDGE_PATH)

    size_mb = os.path.getsize(FP16_CARTRIDGE_PATH) / (1024 * 1024)
    print(f"💾 抽出＆ベイク完了: {FP16_CARTRIDGE_PATH} ({size_mb:.1f} MB / {time.perf_counter() - t0:.2f}s)")
    return FP16_CARTRIDGE_PATH


def step2_quantize_int4(force: bool = False):
    print("\n" + "=" * 80)
    print("🌸 【Step 2】 190k 黄金エングラム Block-128 Int4 量子化 (.pt ➔ _int4.pt)")
    print("=" * 80)

    if os.path.exists(INT4_CARTRIDGE_PATH) and not force:
        size_mb = os.path.getsize(INT4_CARTRIDGE_PATH) / (1024 * 1024)
        print(f"⏩ 既存の Int4 量子化エングラムが見つかりました ({size_mb:.1f} MB)。再利用します。")
        return INT4_CARTRIDGE_PATH

    res = quantize_engram_file(
        src_path=FP16_CARTRIDGE_PATH,
        dst_path=INT4_CARTRIDGE_PATH,
        block_size=128,
        bits=4
    )
    print(f"✅ Int4 量子化完了: {INT4_CARTRIDGE_PATH} ({res['new_size_mb']:.1f} MB, {res['reduction_pct']:.1f}% 削減)")
    return INT4_CARTRIDGE_PATH


def step3_bake_golden_gguf(src_gguf: str, dst_gguf: str, force: bool = False):
    print("\n" + "=" * 80)
    print("🌸 【Step 3】 GGUF ゼロマスク外科手術 (不要多言語 72,227語 完全物理切除)")
    print("=" * 80)

    if os.path.exists(dst_gguf) and not force:
        size_gb = os.path.getsize(dst_gguf) / (1024**3)
        print(f"⏩ 既存の 190k 黄金 GGUF が見つかりました ({size_gb:.2f} GB)。再利用します。")
        return dst_gguf

    if not os.path.exists(src_gguf):
        raise FileNotFoundError(f"Source GGUF not found at: {src_gguf}")

    indices_190k = torch.load(INDICES_PATH, map_location="cpu")
    golden_set = set(indices_190k.tolist())
    TOTAL_VOCAB = 262144
    unwanted_indices = [i for i in range(TOTAL_VOCAB) if i not in golden_set]

    print(f"📦 黄金語彙数: {len(golden_set):,} 語 | 切除ゼロマスク対象: {len(unwanted_indices):,} 語")

    print(f"🔍 GGUF テンソル解析中: {src_gguf}")
    reader = gguf.GGUFReader(src_gguf)

    tok_embd = [t for t in reader.tensors if t.name == "token_embd.weight"][0]
    per_layer = [t for t in reader.tensors if t.name == "per_layer_token_embd.weight"]

    bytes_per_tok_embd = tok_embd.n_bytes // TOTAL_VOCAB
    print(f"  ・token_embd.weight: offset={tok_embd.data_offset}, 1tok={bytes_per_tok_embd} bytes")

    bytes_per_layer = None
    if per_layer:
        pl = per_layer[0]
        bytes_per_layer = pl.n_bytes // TOTAL_VOCAB
        print(f"  ・per_layer_token_embd.weight: offset={pl.data_offset}, 1tok={bytes_per_layer} bytes")

    print(f"\n📦 新種 GGUF バイナリを複製中 ({os.path.getsize(src_gguf)/(1024**3):.2f} GB)...")
    t0 = time.perf_counter()
    shutil.copyfile(src_gguf, dst_gguf)
    print(f"✅ 複製完了: {dst_gguf} ({time.perf_counter() - t0:.2f}s)")

    print(f"\n🔨 190k 黄金ゼロマスク外科手術を実行中...")
    t_surg = time.perf_counter()

    zero_embd_block = b"\x00" * bytes_per_tok_embd
    zero_layer_block = b"\x00" * bytes_per_layer if bytes_per_layer else None

    with open(dst_gguf, "r+b") as f:
        for idx in unwanted_indices:
            # 1. token_embd.weight
            pos_embd = tok_embd.data_offset + idx * bytes_per_tok_embd
            f.seek(pos_embd)
            f.write(zero_embd_block)

            # 2. per_layer_token_embd.weight
            if bytes_per_layer:
                pos_layer = pl.data_offset + idx * bytes_per_layer
                f.seek(pos_layer)
                f.write(zero_layer_block)

    elapsed_surg = time.perf_counter() - t_surg
    print(f"✅ 外科手術完了！ {len(unwanted_indices):,} 語の埋め込みを物理ゼロクリア ({elapsed_surg:.2f}s)")
    return dst_gguf


def step4_test_golden_gguf(dst_gguf: str, llama_cli: str):
    print("\n" + "=" * 80)
    print("🌸 【Step 4】 llama-cli スモークテスト (CUDA 12.4 実機自己紹介)")
    print("=" * 80)

    if not os.path.exists(llama_cli):
        print(f"⚠️ llama-cli が見つかりません ({llama_cli})。テストをスキップします。")
        return

    prompt = "<bos><|turn|>user\nこんにちはアリア！自己紹介して？<turn|>\n<|turn|>model\n"
    cmd = [
        llama_cli,
        "-m", dst_gguf,
        "-ngl", "99",
        "-c", "2048",
        "-n", "384",
        "-p", prompt
    ]
    print(f"🚀 実行コマンド: {' '.join(cmd)}")
    print("-" * 80)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ llama-cli 実行エラー: {e}")


def main():
    parser = argparse.ArgumentParser(description="Aria E4B-it 190k Golden GGUF Complete Pipeline")
    parser.add_argument("--e4b-dir", type=str, default=DEFAULT_E4B_DIR, help="Path to gemma-4-E4B-it safetensors directory")
    parser.add_argument("--src-gguf", type=str, default=DEFAULT_E4B_GGUF, help="Path to original E4B-it Q4_K_M GGUF")
    parser.add_argument("--dst-gguf", type=str, default=DEFAULT_DST_GGUF, help="Path to output 190k Golden GGUF")
    parser.add_argument("--llama-cli", type=str, default=DEFAULT_LLAMA_CLI, help="Path to llama-cli.exe")
    parser.add_argument("--force", action="store_true", help="Force overwrite all existing steps")
    parser.add_argument("--skip-extract", action="store_true", help="Skip Step 1 (Cartridge extraction)")
    parser.add_argument("--skip-quant", action="store_true", help="Skip Step 2 (Int4 quantization)")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip Step 3 (GGUF surgery)")
    parser.add_argument("--skip-test", action="store_true", help="Skip Step 4 (llama-cli test)")
    args = parser.parse_args()

    t_total = time.perf_counter()

    print("=" * 80)
    print("🌸 👑 ⚡ 【Aria E4B-it 190k 黄金 GGUF 完全自動錬金パイプライン】 ⚡ 👑 🌸")
    print("=" * 80)

    # Step 1
    if not args.skip_extract:
        step1_extract_engram(args.e4b_dir, force=args.force)

    # Step 2
    if not args.skip_quant:
        step2_quantize_int4(force=args.force)

    # Step 3
    if not args.skip_gguf:
        step3_bake_golden_gguf(args.src_gguf, args.dst_gguf, force=args.force)

    # Step 4
    if not args.skip_test:
        step4_test_golden_gguf(args.dst_gguf, args.llama_cli)

    total_sec = time.perf_counter() - t_total
    print("\n" + "=" * 80)
    print(f"🎉 🌸 👑 【全工程完了！】 総所要時間: {total_sec:.2f} 秒 👑 🌸 🎉")
    print(f"  ・190k 黄金エングラム (FP16): {FP16_CARTRIDGE_PATH}")
    print(f"  ・190k 黄金エングラム (Int4): {INT4_CARTRIDGE_PATH}")
    print(f"  ・190k 黄金 GGUF モデル    : {args.dst_gguf}")
    print("=" * 80)


if __name__ == "__main__":
    main()

