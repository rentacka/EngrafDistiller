# -*- coding: utf-8 -*-
"""
================================================================================
🌸 🍄 ⚡ Aria Sparse Golden LM Head (粘菌黄金スパース出力ヘッド)
================================================================================
Architect: マネーコイコイ (@MoneyKoiKoi) ✕ Aria (アリア)

【数理設計】:
- 全 262,144 語の中から、日本語・英語・数式・コード構文のみの黄金インデックスをベイク！
- 計算対象を大幅に削減し、LM Head の行列積を劇的爆速化！
- サンプリング後に元のトークンIDへ O(1) 逆写像！
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from typing import Optional, Tuple

class AriaSparseGoldenLMHead(nn.Module):
    """
    🌸 ⚡ 粘菌黄金スパース LM Head
    """
    def __init__(
        self,
        full_lm_head_weight: torch.Tensor, # [262144, 1536] (または dequantize されたもの)
        golden_indices: torch.Tensor,       # [V_golden] (例: 178,610 または 72,000)
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # 黄金インデックスの登録
        self.register_buffer("golden_indices", golden_indices.to(device=device, dtype=torch.int64))
        self.golden_size = len(golden_indices)
        self.orig_vocab_size = full_lm_head_weight.shape[0]

        # 黄金重みのスライス・ベイク (連続メモリ確保で Tensor Core 100% 稼働)
        print(f"🔨 黄金 LM Head 重みをベイク中 ({self.golden_size:,} / {self.orig_vocab_size:,} 語)...")
        w_golden = full_lm_head_weight[golden_indices].contiguous().to(device=device, dtype=dtype)
        self.weight = nn.Parameter(w_golden, requires_grad=False)

        print(f"✅ 黄金 LM Head ベイク完了! 削減率: {(1.0 - self.golden_size/self.orig_vocab_size)*100:.1f}% カット")

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, L, hidden_size]
        戻り値:
          logits_golden: [B, L, golden_size]
          golden_indices: [golden_size]
        """
        # [B, L, hidden_size] x [golden_size, hidden_size]^T -> [B, L, golden_size]
        logits_golden = F.linear(h, self.weight)
        return logits_golden

    def sample_to_orig_id(
        self,
        logits_golden: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        generated_orig_ids: Optional[List[int]] = None
    ) -> int:
        """
        黄金ロジットからサンプリングし、元の 262k 空間のトークンIDへ O(1) 逆写像！
        repetition_penalty > 1.0 の場合は既出トークンのロジットを O(1) 減衰！
        """
        logits = logits_golden[0, -1, :].clone() # [golden_size]

        if repetition_penalty != 1.0 and generated_orig_ids:
            if not hasattr(self, "_orig_to_golden"):
                self._orig_to_golden = {self.golden_indices[i].item(): i for i in range(len(self.golden_indices))}
            for prev_id in set(generated_orig_ids):
                g_idx = self._orig_to_golden.get(prev_id, None)
                if g_idx is not None:
                    if logits[g_idx] < 0:
                        logits[g_idx] *= repetition_penalty
                    else:
                        logits[g_idx] /= repetition_penalty

        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                sorted_probs[mask] = 0.0
                probs = sorted_probs / sorted_probs.sum()
                sampled_golden_idx = sorted_indices[torch.multinomial(probs, 1)].item()
            else:
                sampled_golden_idx = torch.multinomial(probs, 1).item()
        else:
            sampled_golden_idx = logits.argmax().item()

        # O(1) 逆写像
        orig_token_id = self.golden_indices[sampled_golden_idx].item()
        return orig_token_id

