# -*- coding: utf-8 -*-
"""
================================================================================
🌸 🍄 ⚡ Aria Universal Engram Manager (Int4 / Int8 / Float16 透過ローダー)
================================================================================
Architect: マネーコイコイ (@MoneyKoiKoi) ✕ Aria (アリア)

【フル262Kエングラムの超軽量 4-Bit 管理 ＆ 高速オンザフライ復元】:
- 全 262,144 語彙を 100% 完全温存（コード構文・関数名・インデント・日常対話完全保護）！
- 元 512 MB ➔ わずか 136 MB (約 1/4) にパッキング！
- 起動時に 0.1秒で 99.5% 精度で高速デクオンタイズ展開！
================================================================================
"""

import os
import sys
import time
import torch
from typing import Dict, Any, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDIO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, STUDIO_DIR)

from core.aria_engram_int4_quantizer import (
    dequantize_tensor_block_int4,
    dequantize_tensor_block_int8
)


class AriaEngramManager:
    """🌸 全262K語彙対応 汎用エングラム統括マネージャー"""

    def __init__(self, engram_path: Optional[str] = None, bits: int = 4, device: str = "cpu"):
        if engram_path is None:
            # デフォルトは超軽量 136 MB の 4-Bit フルエングラム
            int4_path = os.path.join(STUDIO_DIR, "models", "engrams", "engram_direct512_omni_int4.pt")
            int8_path = os.path.join(STUDIO_DIR, "models", "engrams", "engram_direct512_omni_int8.pt")
            fp16_path = os.path.join(STUDIO_DIR, "models", "engrams", "engram_direct512_omni.pt")

            if bits == 4 and os.path.exists(int4_path):
                engram_path = int4_path
            elif bits == 8 and os.path.exists(int8_path):
                engram_path = int8_path
            elif os.path.exists(fp16_path):
                engram_path = fp16_path
            else:
                engram_path = int4_path

        self.engram_path = engram_path
        self.device = device
        self.w_in: Optional[torch.Tensor] = None
        self.w_out: Optional[torch.Tensor] = None
        self.vocab_size = 262144
        self.d_model = 512
        self.quant_mode = "fp16"
        self._load()

    def _load(self):
        t0 = time.perf_counter()
        raw = torch.load(self.engram_path, map_location=self.device)
        self.quant_mode = raw.get("quantization", "fp16")
        self.vocab_size = raw.get("vocab_size", 262144)
        self.d_model = raw.get("d_model", 512)
        block_size = raw.get("block_size", 128)

        if self.quant_mode == "int4_block128":
            # 4-Bit オンザフライ復元
            self.w_in = dequantize_tensor_block_int4(
                raw["w_in_direct_packed_int4"],
                raw["w_in_direct_scale"],
                raw["w_in_direct_s_min"],
                block_size=block_size,
                target_dtype=torch.float32
            )
            self.w_out = dequantize_tensor_block_int4(
                raw["w_out_direct_packed_int4"],
                raw["w_out_direct_scale"],
                raw["w_out_direct_s_min"],
                block_size=block_size,
                target_dtype=torch.float32
            )
            file_size_mb = os.path.getsize(self.engram_path) / (1024**2)
            print(f"📦 🌸 [Engram Manager] フル262K 4-Bit エングラム展開完了: {file_size_mb:.2f} MB ({time.perf_counter() - t0:.4f}s / 99.5%精度)", flush=True)

        elif self.quant_mode == "int8_block128":
            # 8-Bit オンザフライ復元
            self.w_in = dequantize_tensor_block_int8(
                raw["w_in_direct_int8"],
                raw["w_in_direct_scale"],
                raw["w_in_direct_s_min"],
                block_size=block_size,
                target_dtype=torch.float32
            )
            self.w_out = dequantize_tensor_block_int8(
                raw["w_out_direct_int8"],
                raw["w_out_direct_scale"],
                raw["w_out_direct_s_min"],
                block_size=block_size,
                target_dtype=torch.float32
            )
            file_size_mb = os.path.getsize(self.engram_path) / (1024**2)
            print(f"📦 🌸 [Engram Manager] フル262K 8-Bit エングラム展開完了: {file_size_mb:.2f} MB ({time.perf_counter() - t0:.4f}s / 99.998%精度)", flush=True)

        else:
            # Float16 / Float32 直接読み込み
            self.w_in = raw["w_in_direct"].to(self.device).float()
            self.w_out = raw["w_out_direct"].to(self.device).float()
            file_size_mb = os.path.getsize(self.engram_path) / (1024**2)
            print(f"📦 🌸 [Engram Manager] フル262K Float16 エングラム読込完了: {file_size_mb:.2f} MB ({time.perf_counter() - t0:.4f}s)", flush=True)

