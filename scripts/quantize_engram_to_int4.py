# -*- coding: utf-8 -*-
"""
================================================================================
🌸 🍄 ⚡ scripts/quantize_engram_to_int4.py (Int4 & Int8 対応)
================================================================================
"""

import os
import sys
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDIO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, STUDIO_DIR)

from core.aria_engram_int4_quantizer import quantize_engram_file

SRC_DEFAULT = os.path.join(STUDIO_DIR, "models", "engrams", "engram_direct512_omni.pt")
if not os.path.exists(SRC_DEFAULT):
    SRC_DEFAULT = r"J:\Antigravity\AriaQuantTrader\Aria_FreeToken_Windows_Portable_Complete_Master_Edition\35_Aria_Physarum_Computer_Completion_Studio\models\engrams\engram_direct512_omni.pt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aria Engram Quantizer (4-bit & 8-bit)")
    parser.add_argument("--src", type=str, default=SRC_DEFAULT, help="Input engram path")
    parser.add_argument("--dst", type=str, default=None, help="Output engram path")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Target bits (4 or 8)")
    parser.add_argument("--block-size", type=int, default=128, help="Block size for quantization")
    args = parser.parse_args()

    if args.dst is None:
        base, ext = os.path.splitext(os.path.basename(args.src))
        args.dst = os.path.join(STUDIO_DIR, "models", "engrams", f"{base}_int{args.bits}{ext}")

    quantize_engram_file(args.src, args.dst, block_size=args.block_size, bits=args.bits)
