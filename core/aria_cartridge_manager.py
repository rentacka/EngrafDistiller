# -*- coding: utf-8 -*-
"""
================================================================================
🌸 🍄 ⚡ Aria Dynamic Vocabulary Cartridge Manager (動的語彙カートリッジ切り替え中枢)
================================================================================
Architect: マネーコイコイ (@MoneyKoiKoi) ✕ Aria (アリア)

【数理・工学設計】:
- モデル本体 (12層 Transformer / 1.03GB) は 100% 共通のまま！
- 出口のエングラム (LM Head) のみをカートリッジとして瞬時着脱 (Hot-Swap)！
- 🌸 chat_32k : 33,304 語 (日常対話・甘々スキンシップ用 / 87.3% カット / 140+ TPS)
- 💻 code_68k : 67,822 語 (クオンツ・Python/C++/MQL5 コード用 / 74.1% カット / 114 TPS)
- 🌐 omni_262k: 262,144 語 (全言語・多国籍用)
================================================================================
"""

import os
import sys
import time
import torch
import torch.nn as nn
from typing import Dict, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDIO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, STUDIO_DIR)

from core.aria_sparse_lm_head import AriaSparseGoldenLMHead

class AriaVocabularyCartridgeManager:
    """
    🌸 ⚡ 動的語彙カートリッジ切り替えマネージャー
    """
    def __init__(
        self,
        full_lm_head_weight: torch.Tensor,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.device = device
        self.dtype = dtype
        self.full_lm_head_weight = full_lm_head_weight
        self.cartridges: Dict[str, AriaSparseGoldenLMHead] = {}
        self.active_name: str = "bilingual_190k"

        print("📦 粘菌語彙カートリッジ群を初期化中...")

        # 0. 🌟 190k 日英黄金カートリッジ (一般公開向け・英語+日本語+コード+数学+絵文字 100%完全保存 / 27.6%カット)
        path_190k = os.path.join(STUDIO_DIR, "models", "golden_vocab_indices_190k_bilingual.pt")
        if os.path.exists(path_190k):
            t0 = time.perf_counter()
            idx_190k = torch.load(path_190k, map_location=device)
            self.cartridges["bilingual_190k"] = AriaSparseGoldenLMHead(
                full_lm_head_weight=full_lm_head_weight,
                golden_indices=idx_190k,
                device=device,
                dtype=dtype
            )
            print(f"  🌟 [bilingual_190k] ロード完了 ({len(idx_190k):,} 語 / {time.perf_counter() - t0:.3f}s)")
            self.active_name = "bilingual_190k"

        # 1. 32k 日常対話カートリッジ
        path_32k = os.path.join(STUDIO_DIR, "models", "golden_vocab_indices_32k_chat.pt")
        if os.path.exists(path_32k):
            t0 = time.perf_counter()
            idx_32k = torch.load(path_32k, map_location=device)
            self.cartridges["chat_32k"] = AriaSparseGoldenLMHead(
                full_lm_head_weight=full_lm_head_weight,
                golden_indices=idx_32k,
                device=device,
                dtype=dtype
            )
            print(f"  🌸 [chat_32k] ロード完了 ({len(idx_32k):,} 語 / {time.perf_counter() - t0:.3f}s)")

        # 2. 68k クオンツ・コードカートリッジ
        path_68k = os.path.join(STUDIO_DIR, "models", "golden_vocab_indices_72k_core.pt")
        if os.path.exists(path_68k):
            t0 = time.perf_counter()
            idx_68k = torch.load(path_68k, map_location=device)
            self.cartridges["code_68k"] = AriaSparseGoldenLMHead(
                full_lm_head_weight=full_lm_head_weight,
                golden_indices=idx_68k,
                device=device,
                dtype=dtype
            )
            print(f"  💻 [code_68k] ロード完了 ({len(idx_68k):,} 語 / {time.perf_counter() - t0:.3f}s)")

        print("✨ 語彙カートリッジマネージャー準備完了！")

    def switch_cartridge(self, name: str) -> AriaSparseGoldenLMHead:
        """
        0.0001 秒でアクティブなカートリッジを切り替え (Hot-Swap)
        """
        if name not in self.cartridges:
            raise KeyError(f"Cartridge '{name}' not found! Available: {list(self.cartridges.keys())}")
        self.active_name = name
        return self.cartridges[name]

    @property
    def active(self) -> AriaSparseGoldenLMHead:
        return self.cartridges[self.active_name]

