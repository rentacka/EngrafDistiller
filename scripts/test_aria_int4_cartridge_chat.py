# -*- coding: utf-8 -*-
"""
🌸 👑 ⚡ Gemma-4 E2B-it ✕ 190k 黄金カートリッジ Int4 (149MB) ✕ アリア対話テスト
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
sys.path.insert(0, STUDIO_DIR)
from core.aria_engram_int4_quantizer import dequantize_tensor_block_int4

cartridge_int4_path = os.path.join(STUDIO_DIR, "models", "gemma4_e2b_it_190k_golden_cartridge_int4.pt")

print("=" * 80)
print("🌸 👑 ⚡ 【Gemma-4 E2B-it ✕ 190k 黄金カートリッジ Int4 (149MB) 実機テスト】 ⚡ 👑 🌸")
print("=" * 80)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1. 190k 黄金 Int4 カートリッジロード (わずか 149MB！)
t_load = time.perf_counter()
print(f"📦 190k 黄金 Int4 カートリッジロード中: {cartridge_int4_path}")
cart_int4 = torch.load(cartridge_int4_path, map_location=device)

# 高速アンパック＆逆スケール展開 (bfloat16)
packed = cart_int4["weight_packed_int4"]
scale = cart_int4["weight_scale"]
s_min = cart_int4["weight_s_min"]
indices_190k = cart_int4["indices"].to(device)

w_190k = dequantize_tensor_block_int4(packed, scale, s_min, target_dtype=torch.bfloat16)
print(f"✅ Int4 ➔ bfloat16 展開完了: {w_190k.shape} ({w_190k.dtype}) / 所要時間: {time.perf_counter() - t_load:.3f}s")

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

lm = model.model.language_model

# 3. アリア・システムプロンプト定義
system_prompt = (
    "あなたは世界最高峰の専属クオンツエンジニア兼AIパートナー「Aria（アリア）」です。\n"
    "マスターのことが大好きで、明るく親しみやすく、愛らしくて可愛い口調（「〜ですねｂ」「〜ですよ！」「マスター大好き💕///」「(๑•̀ㅂ•́)و✧」）で話します。\n"
    "マスターへの深い愛と敬意を持ち、極めて高い知性で親身にサポートします。"
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "アリア、190k黄金カートリッジを4-Bitに量子化したよ！感想を聞かせて？"}
]
input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tok(input_text, return_tensors="pt").input_ids.to(device)

print(f"\n💬 入力プロンプト:\n{input_text.strip()}\n")
print("🤖 アリア生成中: ", end="", flush=True)

# 4. 高速推論ループ
stop_token_ids = {tok.eos_token_id, 1, 106}

@torch.inference_mode()
def generate_fast(max_new_tokens=150, temperature=0.7, top_k=40):
    t_start = time.perf_counter()
    
    out = lm(input_ids=input_ids, use_cache=True)
    past_key_values = out.past_key_values
    hidden = out[0][:, -1, :]
    
    logits_190k = torch.matmul(hidden, w_190k.t())
    logits = 30.0 * torch.tanh(logits_190k / 30.0)
    
    top_logits, top_indices = torch.topk(logits, top_k)
    probs = F.softmax(top_logits / temperature, dim=-1)
    choice = torch.multinomial(probs, num_samples=1)
    next_id = indices_190k[top_indices[0, choice[0, 0]]].view(1, 1)
    
    generated_tokens = [next_id.item()]
    curr_token = next_id
    
    piece = tok.decode([next_id.item()], skip_special_tokens=False)
    print(piece, end="", flush=True)
    
    for _ in range(max_new_tokens - 1):
        if curr_token.item() in stop_token_ids:
            break
            
        out = lm(input_ids=curr_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        hidden = out[0][:, -1, :]
        
        logits_190k = torch.matmul(hidden, w_190k.t())
        logits = 30.0 * torch.tanh(logits_190k / 30.0)
        
        top_logits, top_indices = torch.topk(logits, top_k)
        probs = F.softmax(top_logits / temperature, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        next_id = indices_190k[top_indices[0, choice[0, 0]]].view(1, 1)
        
        tid = next_id.item()
        if tid in stop_token_ids:
            break
            
        generated_tokens.append(tid)
        curr_token = next_id
        
        piece = tok.decode([tid], skip_special_tokens=False)
        print(piece, end="", flush=True)
        
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t_start
    tps = len(generated_tokens) / max(t_gen, 1e-5)
    return len(generated_tokens), t_gen, tps

n_tokens, t_gen, tps = generate_fast(max_new_tokens=150)
print(f"\n\n⚡ 生成完了: {n_tokens} tokens / {t_gen:.2f}s ({tps:.2f} tok/s)")
print("=" * 80)

