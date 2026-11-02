# -*- coding: utf-8 -*-
"""
事前確保 静的テンソル KV キャッシュ (Zero Allocation RingBuffer)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any

class StaticGemma4KVCache:
    """
    🌸 ⚡ 事前確保 静的テンソル KV キャッシュ (Zero-Alloc KV-Cache)
    - 毎ステップの torch.cat を完全根絶！
    - 最大長 (例: 512) を事前確保し、スライス代入のみで O(1) 爆速更新！
    """
    def __init__(
        self,
        batch_size: int,
        layer_configs: List[Dict[str, int]],
        max_seq_len: int = 512,
        head_dim: int = 256,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.cur_len = 0

        self.k_caches = []
        self.v_caches = []

        for conf in layer_configs:
            num_kv = conf.get("num_kv_heads", 1)
            cur_hd = conf.get("head_dim", head_dim)
            k = torch.empty((batch_size, num_kv, max_seq_len, cur_hd), device=device, dtype=dtype)
            v = torch.empty((batch_size, num_kv, max_seq_len, cur_hd), device=device, dtype=dtype)
            self.k_caches.append(k)
            self.v_caches.append(v)

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        k_new, v_new: [B, num_kv, L_new, head_dim]
        """
        L_new = k_new.shape[2]
        start = self.cur_len
        end = start + L_new

        self.k_caches[layer_idx][:, :, start:end, :].copy_(k_new)
        self.v_caches[layer_idx][:, :, start:end, :].copy_(v_new)

        # 現在の有効なキャッシュスライスを返す
        k_valid = self.k_caches[layer_idx][:, :, :end, :]
        v_valid = self.v_caches[layer_idx][:, :, :end, :]
        return k_valid, v_valid

    def get_valid(self, layer_idx: int, cur_step_len: Optional[int] = None):
        """
        共有レイヤー向け: アンカー層 (Layer 13 / 14) の最新有効キャッシュを取得
        """
        end = cur_step_len if cur_step_len is not None else self.cur_len
        return self.k_caches[layer_idx][:, :, :end, :], self.v_caches[layer_idx][:, :, :end, :]

    def step(self, L_new: int):
        self.cur_len += L_new

