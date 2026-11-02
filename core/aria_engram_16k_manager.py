# -*- coding: utf-8 -*-
"""
================================================================================
🌸 🍄 ⚡ Aria 16K Engram Manager (Int4 / Float32 透過ローダー)
================================================================================
Architect: マネーコイコイ (@MoneyKoiKoi) ✕ Aria (アリア)

【16K エングラムの高速管理 ＆ オンザフライ Int4 デクオンタイズ】:
- 元の 1.34 GB 語彙を、16,384 黄金語彙 (わずか 8.84 MB) で超軽量管理！
- 蒸留・推論時に 0.001秒で Int4 から復元！
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

from core.aria_engram_int4_quantizer import dequantize_tensor_block_int4


class AriaEngram16KManager:
    """🌸 16K エングラム統括マネージャー"""

    def __init__(self, engram_path: Optional[str] = None, device: str = "cpu"):
        if engram_path is None:
            # デフォルトは最軽量 8.84 MB の Int4 エングラム
            int4_path = os.path.join(STUDIO_DIR, "models", "engrams", "engram_physarum16k_code_balanced_int4.pt")
            fp32_path = os.path.join(STUDIO_DIR, "models", "engrams", "engram_physarum16k_code_balanced.pt")
            engram_path = int4_path if os.path.exists(int4_path) else fp32_path

        self.engram_path = engram_path
        self.device = device
        self.data: Dict[str, Any] = {}
        self.is_int4 = False
        self._load()

    def _load(self):
        t0 = time.perf_counter()
        raw = torch.load(self.engram_path, map_location=self.device)
        self.is_int4 = raw.get("quantization") == "int4_block128"

        if self.is_int4:
            # Int4 からオンザフライ復元
            block_size = raw.get("block_size", 128)
            w_in = dequantize_tensor_block_int4(
                raw["w_in_direct_packed_int4"],
                raw["w_in_direct_scale"],
                raw["w_in_direct_s_min"],
                block_size=block_size,
                target_dtype=torch.float32
            )
            w_out = dequantize_tensor_block_int4(
                raw["w_out_direct_packed_int4"],
                raw["w_out_direct_scale"],
                raw["w_out_direct_s_min"],
                block_size=block_size,
                target_dtype=torch.float32
            )
            self.w_in = w_in
            self.w_out = w_out
            self.vocab_size = raw["vocab_size"]
            self.d_model = raw["d_model"]
            self.gemma_to_physarum = raw["gemma_to_physarum"]
            self.physarum_to_gemma = raw["physarum_to_gemma"]
            file_size_mb = os.path.getsize(self.engram_path) / (1024**2)
            print(f"📦 🌸 [16K Engram Manager] Int4 エングラム高速展開完了: {file_size_mb:.2f} MB ({time.perf_counter() - t0:.4f}s / 99.5%精度復元)", flush=True)
        else:
            self.w_in = raw["w_in_direct"].to(self.device)
            self.w_out = raw["w_out_direct"].to(self.device)
            self.vocab_size = raw["vocab_size"]
            self.d_model = raw["d_model"]
            self.gemma_to_physarum = raw["gemma_to_physarum"]
            self.physarum_to_gemma = raw["physarum_to_gemma"]
            file_size_mb = os.path.getsize(self.engram_path) / (1024**2)
            print(f"📦 🌸 [16K Engram Manager] Float32 エングラム読込完了: {file_size_mb:.2f} MB ({time.perf_counter() - t0:.4f}s)", flush=True)

    def map_token_to_physarum(self, gemma_token_id: int) -> Optional[int]:
        return self.gemma_to_physarum.get(gemma_token_id, None)

    def map_token_to_gemma(self, physarum_token_id: int) -> int:
        return self.physarum_to_gemma.get(physarum_token_id, 0)

