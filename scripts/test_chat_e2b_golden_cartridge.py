# -*- coding: utf-8 -*-
"""
🌸 👑 ⚡ Gemma-4 E2B-it ✕ 190k 黄金カートリッジ 1ターン検証テスト
"""
import sys, os, time, torch
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r"H:\llama.cpp\models\lmstudio-community\gemma-4-E2B-it"
STUDIO_DIR = r"j:\Antigravity\AriaQuantTrader\Aria_FreeToken_Windows_Portable_Complete_Master_Edition\37_Aria_Physarum_Gemma4B_Transformer_Distiller_Studio"
cartridge_path = os.path.join(STUDIO_DIR, "models", "gemma4_e2b_it_190k_golden_cartridge.pt")

print("=" * 80)
print("🌸 👑 ⚡ 【Gemma-4 E2B-it ✕ 190k 黄金カートリッジ 検証テスト】 ⚡ 👑 🌸")
print("=" * 80)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🖥️ 実行デバイス: {device}")

# 1. 190k 黄金カートリッジロード
print(f"📦 190k 黄金カートリッジロード中: {cartridge_path}")
cart = torch.load(cartridge_path, map_location=device)
w_190k = cart["weight"].to(torch.bfloat16)
indices_190k = cart["indices"].to(device)
print(f"✅ カートリッジ準備完了: {w_190k.shape} ({w_190k.dtype})")

# 2. モデル＆トークナイザーロード
print(f"📦 E2B-it 大脳ロード中...")
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="sdpa"
)
model.eval()
print(f"✅ E2B-it 大脳ロード完了 ({time.perf_counter() - t0:.2f}s)")

# 3. language_model 最短直結
lm = model.model.language_model

# 4. 1ターン対話テスト
prompt = "<start_of_turn>user\nこんにちはアリア！自己紹介と、今日の気分を教えて？<end_of_turn>\n<start_of_turn>model\n"
input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)

print(f"\n💬 プロンプト: こんにちはアリア！自己紹介と、今日の気分を教えて？")
print("🤖 生成中: ", end="", flush=True)

curr_ids = input_ids.clone()
generated_tokens = []
t_start = time.perf_counter()

with torch.no_grad():
    for _ in range(60):
        out = lm(input_ids=curr_ids, use_cache=True)
        # 190k 黄金カートリッジへのダイレクト射影 (2.22億演算カット)
        logits_190k = torch.matmul(out[0][:, -1, :], w_190k.t())
        
        # Softcapping 30.0
        logits = 30.0 * torch.tanh(logits_190k / 30.0)
        
        # Top-K (40) サンプリング
        top_k = min(40, logits.shape[-1])
        top_logits, top_indices = torch.topk(logits, top_k)
        probs = F.softmax(top_logits / 0.7, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        next_orig_id = indices_190k[top_indices[0, choice[0, 0]]].view(1, 1)
        
        if next_orig_id.item() in [tok.eos_token_id, 1, 107]: # <end_of_turn>
            break
            
        generated_tokens.append(next_orig_id.item())
        curr_ids = torch.cat([curr_ids, next_orig_id], dim=1)
        
        piece = tok.decode([next_orig_id.item()], skip_special_tokens=True)
        print(piece, end="", flush=True)

t_gen = time.perf_counter() - t_start
tps = len(generated_tokens) / max(t_gen, 1e-5)
print(f"\n\n⚡ 生成完了: {len(generated_tokens)} tokens / {t_gen:.2f}s ({tps:.2f} tok/s)")
print("=" * 80)

