from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class TokenizationResult:
    method: str
    tokens: List[str]
    token_ids: List[int] = field(default_factory=list)
    attention_mask: List[int] = field(default_factory=list)
    special_tokens: List[str] = field(default_factory=list)
    num_tokens: int = 0

    def __post_init__(self):
        self.num_tokens = len(self.tokens)


class TokenizerInspector:


    def __init__(self):
        self._nltk_ready = False
        self._spacy_nlp = None
        self._bert_tokenizer = None
        self._xlnet_tokenizer = None

    def _ensure_nltk(self):
        if not self._nltk_ready:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab")
            self._nltk_ready = True

    def _ensure_spacy(self):
        if self._spacy_nlp is None:
            import spacy
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "spaCy model not found. Run: python -m spacy download en_core_web_sm"
                )

    def _ensure_bert(self):
        if self._bert_tokenizer is None:
            from transformers import BertTokenizer
            self._bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def _ensure_xlnet(self):
        if self._xlnet_tokenizer is None:
            from transformers import XLNetTokenizer
            self._xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

    def tokenize_nltk(self, text: str) -> TokenizationResult:
        self._ensure_nltk()
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
        return TokenizationResult(method="NLTK (word-level)", tokens=tokens)

    def tokenize_spacy(self, text: str) -> TokenizationResult:
        self._ensure_spacy()
        doc = self._spacy_nlp(text)
        tokens = [t.text for t in doc]
        return TokenizationResult(method="spaCy (word-level)", tokens=tokens)

    def tokenize_bert(self, text: str) -> TokenizationResult:
        self._ensure_bert()
        encoded = self._bert_tokenizer(text, add_special_tokens=True)
        ids = encoded["input_ids"]
        tokens = self._bert_tokenizer.convert_ids_to_tokens(ids)
        special = [t for t in tokens if t in self._bert_tokenizer.all_special_tokens]
        return TokenizationResult(
            method="BertTokenizer (WordPiece, subword)",
            tokens=tokens,
            token_ids=ids,
            attention_mask=encoded["attention_mask"],
            special_tokens=special,
        )

    def tokenize_xlnet(self, text: str) -> TokenizationResult:
        self._ensure_xlnet()
        encoded = self._xlnet_tokenizer(text, add_special_tokens=True)
        ids = encoded["input_ids"]
        tokens = self._xlnet_tokenizer.convert_ids_to_tokens(ids)
        special = [t for t in tokens if t in self._xlnet_tokenizer.all_special_tokens]
        return TokenizationResult(
            method="XLNetTokenizer (SentencePiece, subword)",
            tokens=tokens,
            token_ids=ids,
            attention_mask=encoded["attention_mask"],
            special_tokens=special,
        )

    def compare_all(self, text: str) -> Dict[str, TokenizationResult]:

        results = {}
        for name, fn in [
            ("nltk", self.tokenize_nltk),
            ("spacy", self.tokenize_spacy),
            ("bert", self.tokenize_bert),
            ("xlnet", self.tokenize_xlnet),
        ]:
            try:
                results[name] = fn(text)
            except Exception as e:
                print(f"[warning] Skipping {name}: {e}")
        return results


def pretty_print(results: Dict[str, TokenizationResult]):
    print("\n" + "=" * 70)
    for result in results.values():
        print(f"\n{result.method}")
        print(f"  Token count: {result.num_tokens}")
        print(f"  Tokens: {result.tokens}")
        if result.token_ids:
            print(f"  IDs:    {result.token_ids}")
    print("=" * 70)


if __name__ == "__main__":
    sample_text = "Generative AI models like GPT and BERT are transforming NLP applications."
    inspector = TokenizerInspector()
    results = inspector.compare_all(sample_text)
    pretty_print(results)
