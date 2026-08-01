
# 🧪 GenAI & LLMs Q&A Bot

A conversational chatbot focused on Generative AI and LLM concepts, with a built-in, **dataset-free hallucination detector**. Instead of relying on a fixed knowledge base, the bot checks its own reliability in real time by asking itself the same question multiple times and measuring how consistent its answers are — a lightweight version of the *SelfCheckGPT* technique used in hallucination research.

Built as a hands-on project applying the core concepts from a Generative AI course: transformer-based LLMs, tokenization, custom PyTorch data pipelines, and hallucination detection.

---

## ✨ Features

- **Real chat interface** — multi-turn conversation, like ChatGPT/Claude
- **Domain-scoped** — only answers questions about GenAI/LLM topics (GANs, transformers, tokenization, fine-tuning, RAG, etc.); politely redirects anything off-topic instead of guessing
- **Self-consistency hallucination detection** — every answer is checked by generating multiple independent samples and measuring their semantic agreement. High agreement → 🟢 Reliable. Low agreement → 🔴 Likely Hallucinated. No fixed knowledge base required — works without ever needing hand-written ground-truth answers.
- **Tokenizer Inspector tab** — compares how NLTK, spaCy, BertTokenizer, and XLNetTokenizer each split the same input text, including subword tokens, special tokens, token IDs, and attention masks
- **Custom PyTorch data pipeline** (`src/dataset.py`) — a standalone `Dataset` + `DataLoader` with a custom collate function demonstrating batching, padding, and attention masking for LLM training data

---

## 🖼️ Demo

*(Add a screenshot or GIF of the app here before sharing — this is the first thing recruiters see.)*

---

## 🧠 How answers are generated (and how hallucinations are caught)

This isn't a single fixed-answer bot, and it isn't a raw "ask an LLM anything" bot either — it's a layered pipeline that tries the most reliable source first:

1. **Curated knowledge base (102 entries)** — the question is embedded and compared against every KB entry using sentence-transformer embeddings. A strong match (≥0.75 similarity) returns the curated answer directly. No LLM involved, no hallucination risk.
2. **Domain check** — if nothing in the KB is even reasonably close (<0.35 similarity), the question is treated as out of scope, and the bot says so instead of guessing.
3. **LLM fallback, as a last resort** — only for in-domain questions with no strong KB match. The LLM (Claude Haiku via API if configured, otherwise a local `flan-t5-large`) generates an answer, hinted with the closest KB context found.
4. **Self-consistency gate** — the fallback answer is never shown as-is. The model is asked the same question multiple times; if its answers agree with each other, that's a signal of real knowledge. If they disagree, or the answer is degenerate (a single word, "I don't know," etc.), it's suppressed entirely and replaced with a suggestion to ask about a related, better-covered topic instead.

```
User question
     │
     ▼
Embed question, compare to all 102 KB entries (sentence-transformers)
     │
     ├─ similarity < 0.35 ──────────────────────────▶ "outside my scope" message
     │
     ├─ similarity ≥ 0.75 ──────────────────────────▶ curated answer, shown directly (✅ Verified)
     │
     └─ 0.35 ≤ similarity < 0.75 (weak/no exact match)
              │
              ▼
        LLM generates primary answer + 3 sampled alternates
              │
              ▼
        Compare all answers pairwise (semantic similarity)
              │
              ├─ low agreement / degenerate answer ──▶ suppressed, suggest related topics instead
              │
              └─ good agreement ─────────────────────▶ shown, with reliability badge
```

This means the LLM is a genuine last resort, not the default path — and nothing it generates reaches the user without passing a consistency check first.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Answer generation | `google/flan-t5-small` via 🤗 Transformers |
| Consistency scoring | `sentence-transformers` (`all-MiniLM-L6-v2`) embeddings |
| Tokenizer comparison | NLTK, spaCy, `BertTokenizer`, `XLNetTokenizer` |
| Data pipeline | Custom PyTorch `Dataset` + `DataLoader` with collate function |
| UI | Gradio (custom dark purple/pink theme) |

---

## 📂 Project Structure

```
├── app/
│   └── app.py                     # Main Gradio chat app
├── src/
│   ├── self_consistency_detector.py   # Dataset-free hallucination detection
│   ├── hallucination_detector.py      # Similarity backends (lexical + semantic)
│   ├── tokenizer_inspector.py         # NLTK/spaCy/BERT/XLNet comparison
│   └── dataset.py                     # Custom PyTorch Dataset/DataLoader demo
├── data/
│   └── knowledge_base.json        # Sample data used by dataset.py's pipeline demo
├── requirements.txt
└── README.md
```

---

## 🚀 Running Locally

```bash
git clone <your-repo-url>
cd genai-llms-qa-bot

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm

python app/app.py
```

Open the local URL printed in the terminal (usually `http://127.0.0.1:7860`).

> First run downloads a sentence-embedding model, and either `flan-t5-large` (~3GB, if no API key is set) or nothing extra (if using the hosted fallback below) — this needs internet access once; models are cached afterward.

### Optional: better fallback answers via Claude Haiku

The knowledge base directly and reliably answers the ~100 core GenAI/LLM topics it covers. For anything else in-domain but not in the KB, the app falls back to generation. By default this uses a small local model (`flan-t5-large`), which is CPU-friendly but noticeably less fluent. For much better fallback quality, set an Anthropic API key before running the app:

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "your-key-here"

# macOS/Linux
export ANTHROPIC_API_KEY="your-key-here"
```

With the key set, the app automatically uses Claude Haiku for fallback answers instead of the local model — no code changes needed.

---

## ⚠️ Known Limitations

- `flan-t5-small` is a small, CPU-friendly model chosen for fast local demos — its answers are less accurate than larger LLMs. The hallucination-detection *methodology* is the focus of this project, not raw answer quality.
- Self-consistency scoring is a heuristic, not a guarantee — a model can be consistently wrong. It's a useful *signal*, not a certainty.
- The domain guard uses keyword matching, so it's possible to phrase an in-domain question in a way it doesn't recognize, or vice versa.

---

## 📚 What This Project Demonstrates

Built to apply course concepts on generative AI and LLMs:
- Transformer-based LLM architectures (GPT/BERT-style models) and how they're applied to build NLP applications
- Tokenization: comparing word-level (NLTK, spaCy) vs. subword (BERT, XLNet) tokenizers
- Custom PyTorch data pipelines: `Dataset`, `DataLoader`, and collate functions for batching and padding
- AI hallucination detection: evaluating model outputs for factual reliability

---

## 📄 License

MIT — feel free to fork, learn from, or build on this.

# genai-llms-qa-bot
AI-powered Q&amp;A assistant built with Python and Gradio. Uses semantic search with all-MiniLM-L6-v2 to retrieve answers from a knowledge base and falls back to Claude Haiku for low-confidence queries. Includes hallucination detection for more reliable responses.

