"""
self_consistency_detector.py

A dataset-free hallucination detector based on self-consistency checking
(inspired by the SelfCheckGPT approach).

Idea: ask the model the same question multiple times with sampling enabled.
If the model actually "knows" the answer, its responses should stay
semantically consistent across samples. If it's guessing/hallucinating,
the responses tend to drift and contradict each other.

This works on ANY question in ANY domain -- no fixed knowledge base required.
"""

import itertools
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class ConsistencyResult:
    primary_answer: str
    samples: List[str]
    consistency_score: float
    confidence_label: str
    explanation: str


class SelfConsistencyDetector:
    def __init__(self, similarity_fn: Callable[[str, str], float]):
        """similarity_fn: any function that takes two strings and returns a
        0.0-1.0 similarity score (e.g. SemanticSimilarity().similarity)."""
        self.similarity_fn = similarity_fn

    def _average_pairwise_similarity(self, texts: List[str]) -> float:
        pairs = list(itertools.combinations(texts, 2))
        if not pairs:
            return 1.0
        scores = [self.similarity_fn(a, b) for a, b in pairs]
        return sum(scores) / len(scores)

    def classify(self, score: float):
        if score >= 0.65:
            return (
                "🟢 Reliable",
                "The model gave consistent answers across multiple independent attempts, "
                "suggesting confident, likely-factual knowledge.",
            )
        elif score >= 0.35:
            return (
                "🟡 Partially Consistent",
                "The model's answers varied somewhat across attempts. Treat this answer with some caution.",
            )
        else:
            return (
                "🔴 Likely Hallucinated",
                "The model gave inconsistent or contradictory answers across attempts -- "
                "a strong signal it may be guessing rather than recalling real knowledge.",
            )

    def evaluate(self, primary_answer: str, samples: List[str]) -> ConsistencyResult:
        all_texts = [primary_answer] + samples
        score = self._average_pairwise_similarity(all_texts)
        label, explanation = self.classify(score)
        return ConsistencyResult(
            primary_answer=primary_answer,
            samples=samples,
            consistency_score=score,
            confidence_label=label,
            explanation=explanation,
        )
