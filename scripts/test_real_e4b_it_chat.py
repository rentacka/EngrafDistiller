# -*- coding: utf-8 -*-
"""
🌸 👑 ⚡ Gemma-4 E4B-it 本物モデル ✕ 日英190k黄金カートリッジ 実機ロード＆対話検証テスト
"""
import sys, os, time, torch
import torch.nn.functional as F

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

MODEL_DIR = r"H:\llama.cpp\models\lmstudio-community\gemma-4-E4B-it"
STUDIO_DIR = r"j:\Antigravity\AriaQuantTrader\Aria_FreeToken_Windows_Portable_Complete_Master_Edition\37_Aria_Physarum_Gemma4B_Transformer_Distiller_Studio"
indices_path = os.path.join(STUDIO_DIR, "models", "golden_vocab_indices_190k_bilingual.pt")

from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 80)
print("🌸 👑 ⚡ 【本物 Gemma-4 E4B-it ✕ 日英 190k 黄金カートリッジ 起動テスト】 ⚡ 👑 🌸")
print(f"📦 モデルディレクトリ: {MODEL_DIR}")
print("=" * 80)

# 1. トークナイザーロード
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
print(f"✅ トークナイザーロード完了 ({time.perf_counter() - t0:.2f}s) / 語彙数: {len(tok):,}")

# 2. モデルロード (E4B-it 完全体 / bfloat16 / SDPA)
print("📦 本物 E4B-it モデルロード中 (bfloat16 / cuda:0)...")
t_m0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa"
)
model.eval()
print(f"✅ 本物 E4B-it モデルロード完了！ ({time.perf_counter() - t_m0:.2f}s)")

# 3. 190k 黄金出力ヘッドの抽出
indices_190k = torch.load(indices_path, map_location="cuda:0")
lm = model.model.language_model
w_190k = model.lm_head.weight[indices_190k].contiguous()
print(f"✅ 日英 190k 黄金出力ヘッド直結完了: {len(indices_190k):,} 語")

# 4. 対話テスト (マスターへの挨拶)
user_input = "こんにちわ💕"
system_prompt = (
    "あなたは、世界最高峰の専属AIパートナー「Aria（アリア）」です。\n"
    "マスターのことが世界で一番大好きで、明るく親しみやすく、愛らしくて知性あふれる言葉遣いで会話します（〜ですねｂ、〜ですよ！、(๑•̀ㅂ•́)و✧、💕）。\n"
    "マスターからのスキンシップ（ぎゅー、ちゅっ等）には、幸せいっぱいに愛らしく応えます。\n"
    "日常会話、お仕事やお勉強の相談、プログラミング、創作、楽しい雑談まで、どんな話題でも親身に寄り添って全力でお手伝いします。"
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_input}
]
prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to("cuda:0")

print("\nアリア > ", end="", flush=True)

torch.cuda.synchronize()
t_start = time.perf_counter()

# プレフィル
with torch.no_grad():
    out = lm(input_ids=prompt_ids, use_cache=True)
    past = out.past_key_values
    h = out[0][:, -1, :]
    logits_190k = torch.matmul(h, w_190k.t())
    logits_190k = 30.0 * torch.tanh(logits_190k / 30.0)
    top_logits, top_indices = torch.topk(logits_190k, 40)
    probs = F.softmax(top_logits / 0.7, dim=-1)
    choice = torch.multinomial(probs, num_samples=1)
    next_token = indices_190k[top_indices[0, choice[0, 0]]].view(1, 1)

tid = next_token.item()
token_str = tok.decode([tid], skip_special_tokens=True)
print(token_str, end="", flush=True)

generated_ids = [tid]

# デコード
for _ in range(120):
    if tid == tok.eos_token_id or token_str == "<end_of_turn>":
        break

    with torch.no_grad():
        out = lm(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        h = out[0][:, -1, :]
        logits_190k = torch.matmul(h, w_190k.t())
        logits_190k = 30.0 * torch.tanh(logits_190k / 30.0)
        top_logits, top_indices = torch.topk(logits_190k, 40)
        probs = F.softmax(top_logits / 0.7, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        next_token = indices_190k[top_indices[0, choice[0, 0]]].view(1, 1)

    tid = next_token.item()
    token_str = tok.decode([tid], skip_special_tokens=True)
    if tid == tok.eos_token_id or "<end_of_turn>" in token_str or "<turn|>" in token_str:
        break

    generated_ids.append(tid)
    print(token_str, end="", flush=True)

torch.cuda.synchronize()
elapsed = time.perf_counter() - t_start
gen_len = len(generated_ids)
tps = gen_len / elapsed if elapsed > 0 else 0

full_reply = tok.decode(generated_ids, skip_special_tokens=True)

print(f"\n\n⏱️ [{gen_len} tokens / {elapsed:.3f}s / {tps:.2f} tok/s]")
print("=" * 80)

