"""AI FAQ Chatbot — offline question answering over a small knowledge base.

Retrieval is TF-IDF + cosine similarity implemented from scratch using only
the Python standard library (no scikit-learn, no external APIs, no API key).
An optional --llm mode (see llm_mode.py) uses an LLM when a key is present.

Run:  python chatbot.py          (offline mode)
      python chatbot.py --llm    (LLM mode, needs OPENAI_API_KEY)
"""

import json
import math
import os
import re
import sys
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "knowledge_base.json")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "what", "why", "how", "when", "where", "which", "who", "whom",
    "do", "does", "did", "can", "could", "should", "would", "will",
    "i", "you", "he", "she", "it", "we", "they", "me", "my", "your",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "and", "or", "but", "if", "then", "so", "than", "that", "this",
    "there", "here", "please", "tell", "explain", "about", "between",
    "vs", "me",
}

# Small-talk intents, matched before retrieval: (name, keywords, reply)
INTENTS = [
    ("greeting", ("hello", "hi", "hey", "namaste"),
     "Hello! I'm an FAQ chatbot for data engineering, Python, and SQL. "
     "Ask me a question, or type 'help' to see what I can do."),
    ("thanks", ("thanks", "thank", "great", "awesome", "nice"),
     "You're welcome! Happy to help. Ask me anything else."),
    ("bye", ("bye", "goodbye", "quit", "exit"),
     "Goodbye! Good luck with your learning."),
]

CONFIDENCE_THRESHOLD = 0.12


def tokenize(text):
    """Lowercase, strip punctuation, drop stopwords. Returns a list of tokens."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


class TfidfIndex:
    """From-scratch TF-IDF vectorizer + cosine-similarity retrieval."""

    def __init__(self, documents):
        self.documents = documents
        tokenized = [tokenize(doc) for doc in documents]
        n_docs = len(documents)

        # Inverse document frequency with +1 smoothing so unseen terms
        # (and terms in every doc) never divide by zero or vanish.
        doc_freq = Counter()
        for tokens in tokenized:
            for term in set(tokens):
                doc_freq[term] += 1
        self.idf = {
            term: math.log((1 + n_docs) / (1 + df)) + 1.0
            for term, df in doc_freq.items()
        }
        self.vectors = [self._vectorize(tokens) for tokens in tokenized]

    def _vectorize(self, tokens):
        counts = Counter(tokens)
        if not counts:
            return {}
        max_count = max(counts.values())
        vector = {}
        for term, count in counts.items():
            if term in self.idf:
                tf = count / max_count
                vector[term] = tf * self.idf[term]
        return vector

    @staticmethod
    def _cosine(a, b):
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query, top_n=3):
        """Return [(score, doc_index), ...] sorted by score, best first."""
        query_vector = self._vectorize(tokenize(query))
        scored = [
            (self._cosine(query_vector, doc_vector), i)
            for i, doc_vector in enumerate(self.vectors)
        ]
        scored.sort(reverse=True)
        return scored[:top_n]


class Chatbot:
    def __init__(self, kb_path=KB_PATH):
        with open(kb_path, encoding="utf-8") as f:
            self.entries = json.load(f)
        # Index each entry on its question plus its keywords, so natural
        # phrasings of the same topic still match well.
        docs = [
            entry["question"] + " " + " ".join(entry.get("keywords", []))
            for entry in self.entries
        ]
        self.index = TfidfIndex(docs)

    def match_intent(self, text):
        words = set(re.findall(r"[a-z]+", text.lower()))
        for _name, keywords, reply in INTENTS:
            if words & set(keywords):
                return reply
        return None

    def answer(self, question):
        intent = self.match_intent(question)
        if intent:
            return intent
        results = self.index.search(question, top_n=3)
        best_score, best_idx = results[0]
        if best_score < CONFIDENCE_THRESHOLD:
            topics = [self.entries[i]["question"] for _s, i in results]
            suggestions = "; ".join(topics)
            return (
                "I don't know that yet — it's outside my knowledge base. "
                f"Try asking about: {suggestions}. "
                "Or type 'help' to see example questions."
            )
        return self.entries[best_idx]["answer"]


HELP_TEXT = """I can answer questions about data engineering, Python, and SQL. Try:
  - What is ETL?
  - Explain the medallion architecture
  - What are the types of SQL joins?
  - What is a window function in SQL?
  - What is the difference between a list and a tuple in Python?
Type 'quit' or 'exit' to leave."""


def run_cli(use_llm=False):
    bot = Chatbot()
    llm_ask = None
    if use_llm:
        import llm_mode
        if llm_mode.available():
            llm_ask = llm_mode.ask
            print("LLM mode enabled (OPENAI_API_KEY found).")
        else:
            print(llm_mode.UNAVAILABLE_MESSAGE)
            print("Continuing in offline mode.\n")
    print("AI FAQ Chatbot (offline TF-IDF retrieval). "
          "Type 'help' for examples, 'quit' to exit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        lowered = user_input.lower()
        if lowered in ("quit", "exit"):
            print("Bot: Goodbye! Good luck with your learning.")
            break
        if lowered == "help":
            print("Bot:", HELP_TEXT)
            continue
        reply = llm_ask(user_input) if llm_ask else bot.answer(user_input)
        print("Bot:", reply)


if __name__ == "__main__":
    run_cli(use_llm="--llm" in sys.argv)
