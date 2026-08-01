"""
app.py

GenAI & LLMs Q&A Bot -- a real multi-turn chat interface (like ChatGPT/Claude),
restricted to GenAI/LLM topics, with built-in dataset-free hallucination
detection via self-consistency checking, plus a tokenizer inspector tab.

Run with:
    python app/app.py

First run downloads flan-t5-small (~300MB) and the sentence-transformers
embedding model -- needs internet, only happens once (cached after that).
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gradio as gr

from hallucination_detector import SemanticSimilarity, HallucinationDetector
from self_consistency_detector import SelfConsistencyDetector
from tokenizer_inspector import TokenizerInspector

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")
CONFIDENT_MATCH_THRESHOLD = 0.68  # tuned down from 0.75 after regression testing showed near-exact
                                    # phrasing variants (e.g. "What is DPO?" vs the full KB question)
                                    # were narrowly missing the higher threshold
DOMAIN_THRESHOLD = 0.35           # below this -> treat as off-topic; between this and confident -> LLM fallback with hint


APP_TITLE = "GenAI & LLMs Q&A Bot"
APP_SUBTITLE = "Ask about generative AI or LLMs. Every answer is checked for self-consistency to flag possible hallucinations."

NUM_SAMPLES = 3

EXAMPLE_QUESTIONS = [
    "What is a GAN?",
    "What is LoRA?",
    "What is the capital of France?",  # deliberately off-topic, demonstrates the domain guard
]

# --------------------------------------------------------------------------
# Domain guard -- now driven entirely by semantic similarity against the KB,
# instead of keyword matching. A question is "in domain" if it's semantically
# close to ANY topic in the knowledge base, however it's phrased.
# --------------------------------------------------------------------------

OFF_TOPIC_MESSAGE = (
    "🤖 I'm specialized in **Generative AI & LLM topics only** — things like GANs, transformers, "
    "tokenization, fine-tuning, RAG, or hallucination detection.\n\n"
    "That question looks outside my scope. Try asking something like *\"What is a GAN?\"* or "
    "*\"What is tokenization?\"* instead."
)

# --------------------------------------------------------------------------
# Small talk / greetings -- handled separately from the domain guard so basic
# conversational messages feel natural instead of being rejected outright.
# --------------------------------------------------------------------------

GREETING_EXACT = {
    "hi", "hii", "hiii", "hello", "hey", "hey there", "yo", "sup", "hola",
    "hola!", "greetings",
}
FAREWELL_EXACT = {"bye", "goodbye", "see you", "see ya", "tata", "cya"}
THANKS_EXACT = {"thanks", "thank you", "thankyou", "ty", "thanks a lot", "thank u"}
CONTAINS_RESPONSES = {
    "how are you": "I'm doing great, thanks for asking! 😊 I'm your GenAI & LLMs assistant — ask me about GANs, transformers, tokenization, hallucination detection, and more.",
    "how r u": "I'm doing great, thanks for asking! 😊 I'm your GenAI & LLMs assistant — ask me anything about generative AI or LLMs.",
    "how are u": "I'm doing great, thanks for asking! 😊 I'm your GenAI & LLMs assistant — ask me anything about generative AI or LLMs.",
    "how r you": "I'm doing great, thanks for asking! 😊 I'm your GenAI & LLMs assistant — ask me anything about generative AI or LLMs.",
    "whats up": "Not much, just here to talk Generative AI and LLMs! 🤖 What would you like to know?",
    "what's up": "Not much, just here to talk Generative AI and LLMs! 🤖 What would you like to know?",
    "good morning": "Good morning! ☀️ Ready to talk GenAI and LLMs whenever you are.",
    "good evening": "Good evening! 🌙 What GenAI or LLM topic can I help with?",
    "good afternoon": "Good afternoon! What GenAI or LLM topic can I help with today?",
    "who are you": "I'm a GenAI & LLMs Q&A bot 🧪 — ask me about GANs, transformers, tokenization, fine-tuning, hallucination detection, and similar topics.",
    "what can you do": "I can answer questions about Generative AI and LLM concepts — like GANs, transformers, tokenization, RAG, fine-tuning, and hallucination detection. Try asking me something!",
}


def match_smalltalk(message: str):
    """Returns a canned friendly reply for greetings/farewells/thanks, or None
    if the message isn't small talk and should go through the normal pipeline."""
    normalized = message.lower().strip().strip("!.,? ")
    if normalized in GREETING_EXACT:
        return "Hey there! 👋 I'm your GenAI & LLMs assistant. Ask me anything about GANs, transformers, tokenization, hallucination detection, and more!"
    if normalized in FAREWELL_EXACT:
        return "Goodbye! 👋 Come back anytime you have GenAI or LLM questions."
    if normalized in THANKS_EXACT:
        return "You're welcome! 😊 Feel free to ask if you have more GenAI/LLM questions."
    for phrase, response in CONTAINS_RESPONSES.items():
        if phrase in normalized:
            return response
    return None

print("Loading models... this may take a minute the first time.")

# --------------------------------------------------------------------------
# Generator fallback chain -- only used as a LAST RESORT, when the question
# is in-domain but doesn't confidently match the curated knowledge base.
#
# Priority:
#   1. Claude Haiku via the Anthropic API, if ANTHROPIC_API_KEY is set --
#      much more fluent and accurate than a small local model.
#   2. Local flan-t5-large (CPU-friendly, but noticeably weaker) if
#      ENABLE_LOCAL_FALLBACK is true (the default).
#   3. No fallback generator at all -- only the curated KB answers questions.
#      Set ENABLE_LOCAL_FALLBACK=false to use this mode; it's the right
#      choice for free/low-RAM hosting (e.g. Render's free tier, 512MB RAM),
#      since flan-t5-large plus its dependencies won't fit in that budget.
# --------------------------------------------------------------------------

FALLBACK_SYSTEM_PROMPT = (
    "You are a focused assistant that only answers questions about Generative AI and Large "
    "Language Models (e.g. GANs, transformers, tokenization, fine-tuning, RAG, RLHF, hallucination "
    "detection). Answer in one or two clear, accurate sentences. If you are genuinely not confident "
    "in the answer, say so plainly instead of guessing."
)

try:
    import anthropic
    _anthropic_import_ok = True
except ImportError:
    _anthropic_import_ok = False

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
USE_HOSTED_FALLBACK = _anthropic_import_ok and bool(ANTHROPIC_API_KEY)
ENABLE_LOCAL_FALLBACK = os.environ.get("ENABLE_LOCAL_FALLBACK", "true").lower() == "true"

FALLBACK_AVAILABLE = True  # overridden to False in the no-fallback branch below

if USE_HOSTED_FALLBACK:
    print("ANTHROPIC_API_KEY found -- using Claude Haiku as the fallback generator.")
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def generate_text(prompt: str, max_new_tokens: int = 150) -> str:
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_new_tokens,
            temperature=0.0,
            system=FALLBACK_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def generate_samples(prompt: str, n: int = NUM_SAMPLES, max_new_tokens: int = 150):
        samples = []
        for _ in range(n):
            response = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_new_tokens,
                temperature=1.0,
                system=FALLBACK_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            samples.append(response.content[0].text.strip())
        return samples

elif ENABLE_LOCAL_FALLBACK:
    print("No ANTHROPIC_API_KEY found -- using local flan-t5-large as the fallback generator.")
    print("(Tip: set the ANTHROPIC_API_KEY environment variable to use Claude Haiku instead "
          "for noticeably better fallback answers.)")

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    GEN_MODEL_NAME = "google/flan-t5-large"
    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)

    def generate_text(prompt: str, max_new_tokens: int = 80) -> str:
        inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True)
        output_ids = gen_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_samples(prompt: str, n: int = NUM_SAMPLES, max_new_tokens: int = 80):
        inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True)
        output_ids = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            num_return_sequences=n,
        )
        return [gen_tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

else:
    print("No ANTHROPIC_API_KEY and ENABLE_LOCAL_FALLBACK=false -- running in KB-only mode. "
          "Questions without a confident knowledge base match will get a 'no verified answer' "
          "reply instead of an LLM-generated one.")
    FALLBACK_AVAILABLE = False

    def generate_text(prompt: str, max_new_tokens: int = 80) -> str:
        raise RuntimeError("No fallback generator is configured (KB-only mode).")

    def generate_samples(prompt: str, n: int = NUM_SAMPLES, max_new_tokens: int = 80):
        raise RuntimeError("No fallback generator is configured (KB-only mode).")


semantic_similarity = SemanticSimilarity()
consistency_detector = SelfConsistencyDetector(similarity_fn=semantic_similarity.similarity)
kb_detector = HallucinationDetector(KB_PATH, backend=semantic_similarity)
tokenizer_inspector = TokenizerInspector()

print("Models loaded. Launching app...")


# --------------------------------------------------------------------------
# Quality gate for fallback-generated answers -- point 5/7: never show a
# generated answer unless it passes a consistency check AND a basic sanity
# check (not empty, not too short, not a known glitch/blocklisted phrase).
# If the first attempt fails the sanity check, one regeneration is tried
# before giving up and showing a suggestion message instead.
# --------------------------------------------------------------------------

MIN_ANSWER_CHARS = 18   # anything shorter than this is almost never a real answer
MIN_ANSWER_WORDS = 4

DEGENERATE_ANSWERS = {
    "no", "yes", "unanswerable", "unknown", "n/a", "unclear", "not sure",
    "i don't know", "i do not know", "i dont know", "unsure", "none",
    "maybe", "possibly", "not applicable", "no answer", "no idea",
    "i'm not sure", "cannot answer", "can't answer", "no information",
    "not available", "undefined", "null", "error", "true", "false",
}
NO_ANSWER_CONSISTENCY_THRESHOLD = 0.35  # aligns with SelfConsistencyDetector's "Likely Hallucinated" cutoff


def is_degenerate(text: str) -> bool:
    t = text.strip().lower().rstrip(".!?")
    if not t:
        return True
    if t in DEGENERATE_ANSWERS:
        return True
    if len(t) < MIN_ANSWER_CHARS:
        return True
    if len(t.split()) < MIN_ANSWER_WORDS:
        return True
    return False


def get_valid_answer(prompt: str, max_attempts: int = 2):
    """Tries generating up to max_attempts times, returning the first
    non-degenerate answer found, or None if every attempt fails the gate."""
    for _ in range(max_attempts):
        answer = generate_text(prompt)
        if not is_degenerate(answer):
            return answer
    return None


def build_no_answer_message(user_message: str) -> str:
    suggestions = kb_detector.retrieve_top_n(user_message, n=3)
    topic_list = ", ".join(f'"{s.topic}"' for s in suggestions)
    return (
        "🤔 I don't have a verified answer for that yet.\n\n"
        f"Try asking about one of these related topics instead: {topic_list}."
    )


# --------------------------------------------------------------------------
# Theme -- use Gradio's real theming system (not manual CSS var overrides)
# so every component, including chat bubbles, gets correct contrast in both
# light and dark mode automatically.
# --------------------------------------------------------------------------

theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0d0812",
    body_background_fill_dark="#0d0812",
    background_fill_primary="#150d1f",
    background_fill_primary_dark="#150d1f",
    background_fill_secondary="#1a1226",
    background_fill_secondary_dark="#1a1226",
    block_background_fill="#170f22",
    block_background_fill_dark="#170f22",
    block_border_color="rgba(236,72,153,0.18)",
    block_border_color_dark="rgba(236,72,153,0.18)",
    border_color_primary="rgba(236,72,153,0.18)",
    border_color_primary_dark="rgba(236,72,153,0.18)",
    body_text_color="#f1e9fb",
    body_text_color_dark="#f1e9fb",
    body_text_color_subdued="#b9a8d1",
    body_text_color_subdued_dark="#b9a8d1",
    button_primary_background_fill="linear-gradient(90deg, #ec4899, #a855f7)",
    button_primary_background_fill_dark="linear-gradient(90deg, #ec4899, #a855f7)",
    button_primary_background_fill_hover="linear-gradient(90deg, #f472b6, #c084fc)",
    button_primary_background_fill_hover_dark="linear-gradient(90deg, #f472b6, #c084fc)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#241833",
    button_secondary_background_fill_dark="#241833",
    button_secondary_text_color="#f1e9fb",
    button_secondary_text_color_dark="#f1e9fb",
    button_secondary_border_color="rgba(236,72,153,0.25)",
    button_secondary_border_color_dark="rgba(236,72,153,0.25)",
)

# Force the page into dark mode on load, so Tailwind's dark: utility classes
# (used internally by the Chatbot bubbles) apply -- without this, bubbles can
# fall back to light-mode colors and become unreadable against our dark bg.
FORCE_DARK_JS = """
() => {
    document.documentElement.classList.add('dark');
}
"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&display=swap');

html, body { height: 100vh !important; margin: 0 !important; overflow: hidden !important; }
.gradio-container {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow-y: auto !important;
    padding-bottom: 8px !important;
}

.hg-title-banner { text-align: center; padding: 14px 0 2px 0; }
.hg-title-banner h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    background: linear-gradient(90deg, #f472b6, #c084fc, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.9em !important;
    margin-bottom: 2px !important;
}
.hg-subtitle {
    text-align: center;
    opacity: 0.8;
    color: #c9b6e4;
    margin: 0 0 14px 0 !important;
    font-size: 0.95em;
}

#chatbox { border-radius: 16px !important; }

.hg-examples-row {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    justify-content: flex-start !important;
    margin: 6px 0 4px 0 !important;
}
.hg-examples-row button {
    width: auto !important;
    flex: none !important;
    padding: 4px 14px !important;
    font-size: 0.85em !important;
    min-width: unset !important;
}

details.hg-reliability {
    margin-top: 8px;
    font-size: 0.88em;
    border-top: 1px dashed rgba(236,72,153,0.25);
    padding-top: 6px;
    opacity: 0.9;
}
details.hg-reliability summary { cursor: pointer; font-weight: 600; }
"""


def _consistency_badge(label: str, pct: int) -> str:
    if "🟢" in label:
        color = "#34d399"
    elif "🟡" in label:
        color = "#fbbf24"
    else:
        color = "#f87171"
    return f'<span style="color:{color}; font-weight:600;">{label} ({pct}%)</span>'


def _format_bot_message(answer: str, result, samples) -> str:
    pct = round(result.consistency_score * 100)
    badge = _consistency_badge(result.confidence_label, pct)
    samples_list = "\n".join(f"- {s}" for s in samples)
    return (
        f"{answer}\n\n"
        f"<details class='hg-reliability'><summary>🛡️ Reliability: {badge}</summary>\n\n"
        f"{result.explanation}\n\n"
        f"**Alternate generations checked:**\n{samples_list}\n\n"
        f"</details>"
    )


# --------------------------------------------------------------------------
# Core chat logic
# --------------------------------------------------------------------------

def respond(user_message, history):
    if not user_message or not user_message.strip():
        return history, ""

    history = history + [{"role": "user", "content": user_message}]

    smalltalk_reply = match_smalltalk(user_message)
    if smalltalk_reply is not None:
        history = history + [{"role": "assistant", "content": smalltalk_reply}]
        yield history, ""
        return

def respond(user_message, history, last_topic):
    if not user_message or not user_message.strip():
        return history, "", last_topic

    history = history + [{"role": "user", "content": user_message}]

    smalltalk_reply = match_smalltalk(user_message)
    if smalltalk_reply is not None:
        history = history + [{"role": "assistant", "content": smalltalk_reply}]
        yield history, "", last_topic
        return

    history = history + [{"role": "assistant", "content": "🤔 Thinking..."}]
    yield history, "", last_topic

    # --- Context-aware retrieval: fold in the previous topic so short
    # follow-ups like "what is the generator and discriminator?" right after
    # a GAN explanation can still be resolved correctly. ---
    if last_topic:
        retrieval_query = f"{last_topic}. {user_message}"
    else:
        retrieval_query = user_message

    retrieved = kb_detector.retrieve_context(retrieval_query)

    # --- Domain guard: is this even close to anything in the KB? ---
    if retrieved.score < DOMAIN_THRESHOLD:
        history[-1] = {"role": "assistant", "content": OFF_TOPIC_MESSAGE}
        yield history, "", ""  # reset context on an off-topic break
        return

    # --- Strong match: answer directly from the curated KB ---
    if retrieved.score >= CONFIDENT_MATCH_THRESHOLD:
        verified_message = (
            f"{retrieved.answer}\n\n"
            f"<details class='hg-reliability'><summary>🛡️ Reliability: "
            f"<span style=\"color:#34d399; font-weight:600;\">✅ Verified</span></summary>\n\n"
            f"This answer comes directly from a curated knowledge base entry on **{retrieved.topic}**, "
            f"not from LLM generation, so there's no hallucination risk here.\n\n"
            f"</details>"
        )
        history[-1] = {"role": "assistant", "content": verified_message}
        yield history, "", retrieved.topic
        return

    # --- Weak-but-in-domain match: LAST RESORT -> generate, then gate the
    # result before ever showing it. Raw model output is never displayed
    # unless it passes both the consistency check and the sanity check. ---
    if not FALLBACK_AVAILABLE:
        # KB-only mode (e.g. free/low-RAM hosting) -- no generator configured at all.
        history[-1] = {"role": "assistant", "content": build_no_answer_message(user_message)}
        yield history, "", retrieved.topic
        return

    history[-1] = {
        "role": "assistant",
        "content": "🧪 No exact match in the knowledge base — generating and verifying a careful answer...",
    }
    yield history, "", retrieved.topic

    prompt = (
        f"Context: {retrieved.answer}\n\n"
        f"Question: {user_message}\nAnswer in one or two clear sentences:"
    )
    primary_answer = get_valid_answer(prompt)

    if primary_answer is None:
        history[-1] = {"role": "assistant", "content": build_no_answer_message(user_message)}
        yield history, "", retrieved.topic
        return

    samples = generate_samples(prompt)
    result = consistency_detector.evaluate(primary_answer, samples)

    if result.consistency_score < NO_ANSWER_CONSISTENCY_THRESHOLD or is_degenerate(primary_answer):
        history[-1] = {"role": "assistant", "content": build_no_answer_message(user_message)}
        yield history, "", retrieved.topic
        return

    final_message = _format_bot_message(primary_answer, result, samples)
    history[-1] = {"role": "assistant", "content": final_message}
    yield history, "", retrieved.topic


def inspect_tokenizers(text: str):
    if not text or not text.strip():
        return "Please enter some text to tokenize."
    results = tokenizer_inspector.compare_all(text)
    blocks = []
    for r in results.values():
        block = [f"### {r.method}", f"**Token count:** {r.num_tokens}"]
        if r.special_tokens:
            block.append(f"**Special tokens:** `{r.special_tokens}`")
        block.append(f"**Tokens:** `{r.tokens}`")
        if r.token_ids:
            block.append(f"**Token IDs:** `{r.token_ids}`")
        if r.attention_mask:
            block.append(f"**Attention mask:** `{r.attention_mask}`")
        blocks.append("\n\n".join(block))
    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------
# UI layout
# --------------------------------------------------------------------------

with gr.Blocks(title=APP_TITLE, fill_height=True) as demo:
    gr.Markdown(f"<div class='hg-title-banner'>\n\n# {APP_TITLE}\n\n</div>")
    gr.Markdown(f"<div class='hg-subtitle'>{APP_SUBTITLE}</div>")

    with gr.Tab("💬 Chat"):
        chatbot = gr.Chatbot(height=380, elem_id="chatbox", show_label=False)
        topic_state = gr.State("")  # tracks the last matched KB topic, for follow-up question context

        with gr.Row(elem_classes=["hg-examples-row"]):
            example_btns = [gr.Button(ex, size="sm") for ex in EXAMPLE_QUESTIONS]

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Ask about GANs, transformers, tokenization, hallucinations...",
                show_label=False,
                scale=8,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        clear_btn = gr.Button("🗑️ Clear Chat", size="sm")

        chat_inputs = [msg_box, chatbot, topic_state]
        chat_outputs = [chatbot, msg_box, topic_state]

        send_btn.click(fn=respond, inputs=chat_inputs, outputs=chat_outputs)
        msg_box.submit(fn=respond, inputs=chat_inputs, outputs=chat_outputs)

        for ex, btn in zip(EXAMPLE_QUESTIONS, example_btns):
            btn.click(fn=lambda e=ex: e, outputs=msg_box).then(
                fn=respond, inputs=chat_inputs, outputs=chat_outputs
            )

        clear_btn.click(fn=lambda: ([], "", ""), outputs=[chatbot, msg_box, topic_state])

    with gr.Tab("🔤 Tokenizer Inspector"):
        gr.Markdown(
            "See how different tokenizers (NLTK, spaCy, BERT, XLNet) split the same text — "
            "including special tokens, token IDs, and attention masks."
        )
        token_input = gr.Textbox(
            label="Text to tokenize",
            value="Generative AI models like GPT and BERT are transforming NLP.",
        )
        token_btn = gr.Button("Compare Tokenizers")
        token_output = gr.Markdown()
        token_btn.click(fn=inspect_tokenizers, inputs=token_input, outputs=token_output)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=CUSTOM_CSS,
        theme=theme,
        js=FORCE_DARK_JS,
    )
