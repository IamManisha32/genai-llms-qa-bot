
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


SINGLE_TURN_CASES = [

    ("What is a GAN?", "kb_verified"),
    ("What is a transformer?", "kb_verified"),
    ("What is tokenization?", "kb_verified"),
    ("What is a VAE?", "kb_verified"),
    ("What is a diffusion model?", "kb_verified"),
    ("What is fine-tuning?", "kb_verified"),
    ("What is RAG?", "kb_verified"),
    ("What is RLHF?", "kb_verified"),
    ("What is LoRA?", "kb_verified"),
    ("What is prompt engineering?", "kb_verified"),
    ("What is an AI hallucination?", "kb_verified"),
    ("What is the generator in a GAN?", "kb_verified"),
    ("What is the discriminator in a GAN?", "kb_verified"),
    ("What is subword tokenization?", "kb_verified"),
    ("What is Byte Pair Encoding?", "kb_verified"),
    ("What is WordPiece tokenization?", "kb_verified"),
    ("What is SentencePiece?", "kb_verified"),
    ("What is an encoder?", "kb_verified"),
    ("What is a decoder?", "kb_verified"),
    ("What is cross-attention?", "kb_verified"),
    ("What is in-context learning?", "kb_verified"),
    ("What is a system prompt?", "kb_verified"),
    ("What is a reward model?", "kb_verified"),
    ("What is PPO?", "kb_verified"),
    ("What is DPO?", "kb_verified"),
    ("What is Constitutional AI?", "kb_verified"),
    ("What is a Mixture of Experts model?", "kb_verified"),
    ("What is an AI agent?", "kb_verified"),
    ("What is function calling in LLMs?", "kb_verified"),
    ("What is semantic search?", "kb_verified"),
    ("What are scaling laws?", "kb_verified"),
    ("What are emergent abilities?", "kb_verified"),
    ("What is layer normalization?", "kb_verified"),
    ("What is a residual connection?", "kb_verified"),
    ("What is the softmax function?", "kb_verified"),
    ("What is cross-entropy loss?", "kb_verified"),
    ("What is ReLU?", "kb_verified"),

    ("What is the capital of France?", "off_topic"),
    ("Who won the last cricket world cup?", "off_topic"),
    ("What's a good recipe for pasta?", "off_topic"),
    ("What's the weather like today?", "off_topic"),

    ("hi", "smalltalk"),
    ("hello!", "smalltalk"),
    ("how are you?", "smalltalk"),
    ("thanks", "smalltalk"),
    ("bye", "smalltalk"),
    ("who are you", "smalltalk"),
]

FOLLOWUP_CASES = [
    (["What is a GAN?", "what about the generator and discriminator?"], "kb_verified"),
    (["What is tokenization?", "and what about the subword version?"], "kb_verified"),
    (["What is RLHF?", "what's the reward model part?"], "kb_verified"),
]


def run_single_turn(respond_fn, question, history=None, last_topic=""):
    history = history or []
    gen = respond_fn(question, history, last_topic)
    final_history, final_topic = None, last_topic
    for h, _box, t in gen:
        final_history, final_topic = h, t
    return final_history, final_topic


SMALLTALK_MARKERS = [
    "hey there!", "i'm your genai", "i'm doing great, thanks for asking",
    "you're welcome!", "goodbye! 👋", "i'm a genai & llms q&a bot",
]


def classify_response(content: str) -> str:
    if "✅ Verified" in content:
        return "kb_verified"
    if "outside my scope" in content or "specialized in" in content:
        return "off_topic"
    if "don't have a verified answer" in content:
        return "suppressed_fallback"
    if "🛡️ Reliability" in content:
        return "fallback_shown"
    if any(marker in content.lower() for marker in SMALLTALK_MARKERS):
        return "smalltalk"
    return "smalltalk_or_other"


def main():
    print("Importing app.py (this loads real models -- may take a minute)...\n")
    import app as app_module  # noqa: E402

    respond_fn = app_module.respond

    passed, failed = 0, 0
    failures = []

    print("=" * 70)
    print("SINGLE-TURN TESTS")
    print("=" * 70)
    for question, expected in SINGLE_TURN_CASES:
        history, _ = run_single_turn(respond_fn, question)
        content = history[-1]["content"]
        actual = classify_response(content)
        ok = (actual == expected) or (expected == "kb_verified" and actual == "fallback_shown")
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((question, expected, actual, content[:150]))
        print(f"[{status}] {question!r:55} expected={expected:12} actual={actual}")

    print()
    print("=" * 70)
    print("MULTI-TURN FOLLOW-UP TESTS")
    print("=" * 70)
    for turns, expected in FOLLOWUP_CASES:
        history, topic = [], ""
        for turn in turns:
            history, topic = run_single_turn(respond_fn, turn, history, topic)
        content = history[-1]["content"]
        actual = classify_response(content)
        ok = (actual == expected) or (expected == "kb_verified" and actual == "fallback_shown")
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((" -> ".join(turns), expected, actual, content[:150]))
        print(f"[{status}] {' -> '.join(turns)!r:55} expected={expected:12} actual={actual}")

    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed (out of {passed + failed})")
    print("=" * 70)
    if failures:
        print("\nFAILURES (review these manually):")
        for q, exp, act, preview in failures:
            print(f"\n  Question: {q}")
            print(f"  Expected: {exp}  |  Actual: {act}")
            print(f"  Response preview: {preview}")


if __name__ == "__main__":
    main()
