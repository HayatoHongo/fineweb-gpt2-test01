import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import tiktoken

# ----------------------------
# モデル定義（train.py から必要な部分のみ抜粋）
# ----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # PyTorch の組み込み関数 scaled_dot_product_attention を使用（causal=True）
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# シンプルなデータクラスとして設定情報を保持
class GPTConfig:
    def __init__(self, *, block_size, vocab_size, n_layer, n_head, n_embd):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer    = n_layer
        self.n_head     = n_head
        self.n_embd     = n_embd

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 重み共有
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"入力シーケンス長 {T} が block_size {self.config.block_size} を超えています"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        tok_emb = self.wte(idx)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ----------------------------
# モデルのロードとテキスト生成関数
# ----------------------------

# デバイス設定（GPUがあればGPUを使用）
device = "cuda" if torch.cuda.is_available() else "cpu"

# 保存済みチェックポイントのパス
MODEL_PATH = "model_00999.pt"

# チェックポイントをロード（config と state_dict を取得）
checkpoint = torch.load(MODEL_PATH, map_location=device)
config = checkpoint['config']
# config が dict の場合は GPTConfig のインスタンスに変換
if isinstance(config, dict):
    config = GPTConfig(**config)
    
# モデル生成
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# tiktoken のエンコーダを取得（GPT-2 用）
enc = tiktoken.get_encoding("gpt2")

def generate_text(prompt, max_length=100, top_k=50):
    # プロンプトをトークン化
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # シーケンスを生成
    model.eval()
    with torch.no_grad():
        while x.size(1) < max_length:
            logits, _ = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # top-k サンプリング
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            # 確率に基づき次のトークンを選択
            next_token = torch.multinomial(topk_probs, num_samples=1)
            next_token = torch.gather(topk_indices, -1, next_token)
            x = torch.cat((x, next_token), dim=1)
    output = enc.decode(x[0].tolist())
    return output

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("LLM テキスト生成アプリ")
st.write("プロンプトを入力してテキスト生成を実行します。")

prompt = st.text_area("プロンプトを入力してください：", value="Hello, I'm a language model,", height=100)
max_length = st.number_input("生成するテキストの最大長（トークン数）", min_value=10, max_value=500, value=100, step=10)
top_k = st.number_input("Top-k サンプリングの k 値", min_value=10, max_value=100, value=50, step=1)

if st.button("生成開始"):
    st.write("生成中...")
    output_text = generate_text(prompt, max_length=int(max_length), top_k=int(top_k))
    st.subheader("生成結果")
    st.text_area("", value=output_text, height=300)
