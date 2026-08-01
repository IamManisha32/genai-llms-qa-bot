"""
hallucination_detector.py

Core logic for the project's hallmark feature: flagging when a chatbot's
answer is NOT well-grounded in the known knowledge base (a lightweight
proxy for detecting hallucinations).
"""

import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Protocol

# Similarity backends

class SimilarityBackend(Protocol):
    def similarity(self, text_a: str, text_b: str) -> float:
        ...


class LexicalSimilarity:


    def _normalize(self, text: str) -> set:
        words = re.findall(r"[a-z0-9]+", text.lower())
        stopwords = {"a", "an", "the", "is", "are", "of", "in", "to", "and", "what", "how"}
        return {w for w in words if w not in stopwords}

    def similarity(self, text_a: str, text_b: str) -> float:
        set_a, set_b = self._normalize(text_a), self._normalize(text_b)
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0


class SemanticSimilarity:


    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer, util
        self.model = SentenceTransformer(model_name)
        self._util = util

    def similarity(self, text_a: str, text_b: str) -> float:
        emb_a = self.model.encode(text_a, convert_to_tensor=True)
        emb_b = self.model.encode(text_b, convert_to_tensor=True)
        score = self._util.cos_sim(emb_a, emb_b).item()
        return max(0.0, min(1.0, score))  # clamp to [0, 1]

# Retrieval + scoring


@dataclass
class RetrievalResult:
    kb_id: str
    topic: str
    question: str
    answer: str
    score: float


@dataclass
class GroundednessResult:
    generated_answer: str
    retrieved: RetrievalResult
    groundedness_score: float
    confidence_label: str
    warning: Optional[str]


class HallucinationDetector:
    def __init__(self, kb_path: str, backend: SimilarityBackend):
        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb: List[Dict] = json.load(f)
        self.backend = backend

    def retrieve_context(self, user_question: str) -> RetrievalResult:
        """Finds the KB entry whose question is most similar to the user's question."""
        results = self.retrieve_top_n(user_question, n=1)
        return results[0]

    def retrieve_top_n(self, user_question: str, n: int = 3) -> List[RetrievalResult]:

        scored = []
        for entry in self.kb:
            score = self.backend.similarity(user_question, entry["question"])
            scored.append(
                RetrievalResult(
                    kb_id=entry["id"],
                    topic=entry["topic"],
                    question=entry["question"],
                    answer=entry["answer"],
                    score=score,
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:n]

    def classify_confidence(self, score: float) -> (str, Optional[str]):

        if score >= 0.5:
            return "🟢 Reliable", None
        elif score >= 0.25:
            return "🟡 Partially Grounded", "This answer only partially matches known information. Verify before trusting it."
        else:
            return "🔴 Likely Hallucinated", "This answer does not appear to be grounded in the knowledge base. It may be inaccurate."

    def is_out_of_domain(self, retrieval_score: float, threshold: float = 0.30) -> bool:
        return retrieval_score < threshold

    def list_topics(self) -> List[Dict]:

        return self.kb

    def evaluate(self, user_question: str, generated_answer: str) -> GroundednessResult:
        retrieved = self.retrieve_context(user_question)
        groundedness_score = self.backend.similarity(generated_answer, retrieved.answer)
        label, warning = self.classify_confidence(groundedness_score)
        return GroundednessResult(
            generated_answer=generated_answer,
            retrieved=retrieved,
            groundedness_score=groundedness_score,
            confidence_label=label,
            warning=warning,
        )


if __name__ == "__main__":
    import os

    kb_path = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")
    detector = HallucinationDetector(kb_path, backend=LexicalSimilarity())

    test_cases = [
        # A "good" answer -- should score as grounded
        ("What is a GAN?", "A GAN trains a generator and discriminator against each other."),
        # A vague/wrong answer -- should get flagged
        ("What is a GAN?", "Bananas are a great source of potassium and fiber."),
        # A partially-correct answer -- should land in the middle
        ("What is tokenization?", "Tokenization is about splitting words somehow for computers."),
    ]

    for question, answer in test_cases:
        result = detector.evaluate(question, answer)
        print("=" * 70)
        print(f"User question:      {question}")
        print(f"Model answer:        {answer}")
        print(f"Matched KB topic:    {result.retrieved.topic} (id={result.retrieved.kb_id})")
        print(f"Retrieved answer:    {result.retrieved.answer}")
        print(f"Groundedness score:  {result.groundedness_score:.2f}")
        print(f"Confidence label:    {result.confidence_label}")
        if result.warning:
            print(f"Warning:             {result.warning}")
    print("=" * 70)

