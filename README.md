# LLM from Scratch 🧠

A hands-on Python project that implements the foundational building blocks of a Large Language Model (LLM) from scratch — starting with tokenization. The project walks through different tokenization strategies used in modern LLMs, from simple character/word-level tokenizers to Byte Pair Encoding (BPE) used by GPT-2.

---

## 📁 Project Structure

```
LLM_Scratch/
├── main.py                         # Entry point: demonstrates all tokenizers end-to-end
├── Tokenizers/
│   ├── SimpleTokenizerV1.py        # Basic vocabulary-based tokenizer
│   ├── SimpleTokenizerV2.py        # Extended tokenizer with <|unk|> and <|endoftext|> support
│   ├── BPE_Tokenizer.py            # Byte Pair Encoding tokenizer via tiktoken (GPT-2)
│   └── tokenization.ipynb          # Jupyter notebook walkthrough of tokenization concepts
└── Data/
    ├── verdict.txt                  # Sample text corpus used for training the vocabulary
    ├── Attention_all_you_need.pdf   # Reference: "Attention Is All You Need" paper
    └── Byte_Pair_Encoding.pdf       # Reference: BPE algorithm paper
```

---

## 🔍 What's Implemented

### 1. `SimpleTokenizerV1`
A basic tokenizer that builds an integer vocabulary from a given text corpus. It supports:
- `encode(text)` — splits text using regex and maps tokens to integer IDs
- `decode(ids)` — converts integer IDs back to human-readable text

> ⚠️ Raises a `KeyError` for tokens not seen during vocabulary construction.

### 2. `SimpleTokenizerV2`
An improved version of V1 that handles out-of-vocabulary (OOV) words gracefully. It introduces two special tokens:
- `<|unk|>` — replaces any unknown token at encode time
- `<|endoftext|>` — used as a separator between unrelated text passages

### 3. `BPE_Tokenizer`
A wrapper around OpenAI's [tiktoken](https://github.com/openai/tiktoken) library using the `gpt2` encoding (50,257-token vocabulary). This is the same tokenizer used in GPT-2 and GPT-3.
- `encoder(text)` — encodes text to BPE token IDs
- `decoder(tokens)` — decodes token IDs back to text

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Install dependencies:

```bash
pip install tiktoken
```

### Run the Demo

```bash
python main.py
```

This will:
1. Load `Data/verdict.txt` and build a vocabulary
2. Encode and decode a sample sentence using **SimpleTokenizerV1**
3. Add special tokens and repeat using **SimpleTokenizerV2**
4. Encode and decode a sample text using the **BPE Tokenizer**

---

## 📖 Example Output

```
# SimpleTokenizerV1
[3, 85, 43, ...]
I had always thought Jack Gisburn rather a cheap genius -- though a good fellow enough ...

# SimpleTokenizerV2 (with special tokens)
[55, 1001, 75, ...]
Hello , <|unk|> you like tea ? <|endoftext|> In the sunlit <|unk|> of the place

# BPE Tokenizer (GPT-2)
BPE Encoding
[17250, 428, 318, 257, ...]
Hi this is a random text. <|endoftext|> this is a random word dksaimanikantakjdafk.
```

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) — Sennrich et al.
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's fast BPE tokenizer

---

## 🛣️ Roadmap

- [x] Simple vocabulary tokenizer (V1)
- [x] Tokenizer with special tokens (V2)
- [x] Byte Pair Encoding via tiktoken (GPT-2)
- [ ] Token embeddings & positional encodings
- [ ] Self-attention mechanism
- [ ] Transformer block
- [ ] GPT-style model training loop

---

## 🙌 Author

**Sai Manikanta** — building LLMs from the ground up, one component at a time.

> ⚡ **Not a single line of vibe coding here.** Every line in this repository was written by me, from scratch, with full understanding of what it does and why.

---


