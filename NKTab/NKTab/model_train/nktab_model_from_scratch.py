from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
TRAIN_FILE = DATA_DIR / "train.txt"
MODEL_FILE = CHECKPOINT_DIR / "nktab_model.pt"
META_FILE = CHECKPOINT_DIR / "nktab_meta.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    batch_size: int = 32
    block_size: int = 128
    max_iters: int = 2500
    eval_interval: int = 250
    eval_batches: int = 50
    learning_rate: float = 3e-4
    n_embd: int = 192
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.1
    seed: int = 42


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        if not chars:
            raise ValueError("Training text is empty. Fill data/train.txt first.")
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)

    def to_dict(self) -> dict:
        return {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        obj = cls.__new__(cls)
        obj.stoi = {str(k): int(v) for k, v in data["stoi"].items()}
        obj.itos = {int(k): str(v) for k, v in data["itos"].items()}
        obj.vocab_size = int(data["vocab_size"])
        return obj


def causal_mask(size: int, device: str) -> torch.Tensor:
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        attn_input = self.ln1(x)
        mask = causal_mask(t, x.device)
        y, _ = self.attn(attn_input, attn_input, attn_input, attn_mask=mask, need_weights=False)
        x = x + y
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(cfg.n_embd, cfg.n_head, cfg.dropout) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.cfg.block_size}")

        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(t, device=idx.device))
        x = tok + pos.unsqueeze(0)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            b, t_out, c = logits.shape
            loss = F.cross_entropy(logits.reshape(b * t_out, c), targets.reshape(b * t_out))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def read_text() -> str:
    if not TRAIN_FILE.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        TRAIN_FILE.write_text(
            "NKTab — это мой AI помощник.\n"
            "Он помогает писать код, исправлять ошибки и создавать проекты.\n"
            "Python полезен для ботов, сайтов и автоматизации.\n"
            "Telegram-бот может принимать команды и отвечать пользователю.\n"
            "Ошибки в коде нужно анализировать по traceback и строке сбоя.\n"
            "Хороший проект имеет понятную структуру файлов и инструкцию запуска.\n",
            encoding="utf-8",
        )
        print(f"Created sample training file: {TRAIN_FILE}")
    return TRAIN_FILE.read_text(encoding="utf-8")


def build_dataset(tokenizer: CharTokenizer, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer.encode(text)
    if len(encoded) < 2:
        raise ValueError(f"Training text in {TRAIN_FILE} is too small.")
    data = torch.tensor(encoded, dtype=torch.long)
    split_index = int(0.9 * len(data))
    split_index = min(max(split_index, 1), len(data) - 1)
    train_data = data[:split_index]
    val_data = data[split_index:]
    return train_data, val_data


def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    if len(data) <= cfg.block_size + 1:
        raise ValueError(
            f"Not enough text for training. Need more than {cfg.block_size + 1} tokens, got {len(data)}. "
            f"Add more text to {TRAIN_FILE}."
        )
    max_start = len(data) - cfg.block_size - 1
    ix = torch.randint(0, max_start + 1, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model: TinyGPT, train_data: torch.Tensor, val_data: torch.Tensor, cfg: Config) -> Dict[str, float]:
    out: Dict[str, float] = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_batches)
        for k in range(cfg.eval_batches):
            xb, yb = get_batch(split, train_data, val_data, cfg)
            _, loss = model(xb, yb)
            if loss is None:
                raise RuntimeError("Loss was not computed during evaluation.")
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_model(model: TinyGPT, tokenizer: CharTokenizer, cfg: Config) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_FILE)
    meta = {
        "tokenizer": tokenizer.to_dict(),
        "config": asdict(cfg),
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved metadata to: {META_FILE}")


def load_model() -> Tuple[TinyGPT, CharTokenizer, Config]:
    if not MODEL_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("Model not found. Train first: python nktab_model_from_scratch.py train")

    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    tokenizer = CharTokenizer.from_dict(meta["tokenizer"])
    cfg = Config(**meta["config"])
    model = TinyGPT(tokenizer.vocab_size, cfg).to(DEVICE)
    state_dict = torch.load(MODEL_FILE, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer, cfg


def generate_text(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 1.0,
) -> str:
    encoded = tokenizer.encode(prompt)
    if not encoded:
        encoded = tokenizer.encode(" ")
    if not encoded:
        encoded = [0]
    idx = torch.tensor([encoded], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(out[0].tolist())


def train() -> None:
    set_seed(CFG.seed)
    text = read_text()
    tokenizer = CharTokenizer(text)
    train_data, val_data = build_dataset(tokenizer, text)

    print(f"Device: {DEVICE}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_data)} | Val tokens: {len(val_data)}")

    model = TinyGPT(tokenizer.vocab_size, CFG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate)

    for step in range(CFG.max_iters):
        if step % CFG.eval_interval == 0 or step == CFG.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, CFG)
            print(f"step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", train_data, val_data, CFG)
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("Loss was not computed during training.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_model(model, tokenizer, CFG)

    prompt = "NKTab"
    generated = generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.9)
    print("\n=== Sample generation ===\n")
    print(generated)


app = FastAPI(title="NKTab Local Model")
MODEL_CACHE: TinyGPT | None = None
TOKENIZER_CACHE: CharTokenizer | None = None
CFG_CACHE: Config | None = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(120, ge=1, le=500)
    temperature: float = Field(0.9, ge=0.1, le=2.0)


@app.get("/")
def root() -> dict:
    return {
        "name": "NKTab Local Model",
        "status": "ok",
        "device": DEVICE,
        "message": "Use POST /generate",
    }


@app.post("/generate")
def api_generate(req: GenerateRequest) -> dict:
    global MODEL_CACHE, TOKENIZER_CACHE, CFG_CACHE
    if MODEL_CACHE is None or TOKENIZER_CACHE is None or CFG_CACHE is None:
        MODEL_CACHE, TOKENIZER_CACHE, CFG_CACHE = load_model()

    text = generate_text(
        MODEL_CACHE,
        TOKENIZER_CACHE,
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    return {
        "prompt": req.prompt,
        "response": text,
    }


def serve() -> None:
    print("Starting NKTab local model server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)


def print_help() -> None:
    print(
        """
Usage:
  python nktab_model_from_scratch.py train
  python nktab_model_from_scratch.py generate \"привет\"
  python nktab_model_from_scratch.py serve

Tips:
  1) Put your real training text into data/train.txt
  2) The more good text you add, the better your model becomes
  3) For coding style, add examples of code + explanations + prompts + answers
        """
    )


def main() -> None:
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()

    if command == "train":
        train()
        return

    if command == "generate":
        if len(sys.argv) < 3:
            print("Pass a prompt. Example: python nktab_model_from_scratch.py generate \"привет\"")
            return
        prompt = sys.argv[2]
        model, tokenizer, _ = load_model()
        print(generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.9))
        return

    if command == "serve":
        serve()
        return

    print_help()


if __name__ == "__main__":
    main()
