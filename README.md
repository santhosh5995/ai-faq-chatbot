# Bass AI — AI FAQ Chatbot

A chatbot (live demo: https://santhosh5995.github.io/ai-faq-chatbot/) that answers questions about **data engineering, Python, and SQL** — built in pure Python with **TF-IDF retrieval implemented from scratch**. No scikit-learn, no external APIs, no API key: it runs fully offline using only the Python standard library.

Anyone can clone it and run it instantly:

```bash
python chatbot.py
```

## How it works

Question answering is treated as an information-retrieval problem:

1. **Tokenizer** — lowercases the text, strips punctuation, and removes common stopwords ("what", "is", "the", ...) so only meaningful terms remain.
2. **TF-IDF vectors** — every question in the knowledge base (plus its keywords) is converted into a sparse vector. Each term is weighted by its *term frequency* in the document multiplied by its *inverse document frequency* across the whole knowledge base, so rare, informative words (like "medallion" or "kafka") count far more than common ones. IDF uses +1 smoothing so unseen terms never break the math.
3. **Cosine similarity** — your question is vectorized the same way and compared against every knowledge-base entry; the entry with the highest cosine similarity is the best match.
4. **Confidence threshold** — if the best score is too low, the bot says so honestly and suggests topics it *can* answer instead of making something up.
5. **Small talk first** — greetings, thanks, and goodbyes are handled by simple pattern matching before retrieval runs.

The knowledge base (`knowledge_base.json`) holds 24 Q&A entries: ETL, medallion architecture, SQL joins, indexes, normalization, window functions, CTEs, Python lists vs tuples, decorators, pandas, Airflow, Kafka, star schema, Docker, Git, REST APIs, and more. Adding a new topic is just adding one JSON entry — no code changes.

## Example conversation

Real output from running `python chatbot.py`:

```text
AI FAQ Chatbot (offline TF-IDF retrieval). Type 'help' for examples, 'quit' to exit.

You: Hello!
Bot: Hello! I'm an FAQ chatbot for data engineering, Python, and SQL. Ask me a question, or type 'help' to see what I can do.

You: Explain the medallion architecture
Bot: The medallion architecture organizes data in a lakehouse into three layers of increasing quality. Bronze holds raw data exactly as it arrived, Silver holds cleaned and validated data, and Gold holds curated, business-ready tables and aggregates optimized for reporting and analytics.

You: What are the types of SQL joins?
Bot: The main SQL join types are INNER JOIN (only matching rows from both tables), LEFT JOIN (all rows from the left table plus matches from the right), RIGHT JOIN (the reverse of LEFT), FULL OUTER JOIN (all rows from both tables), and CROSS JOIN (every combination of rows). Choosing the right join depends on whether unmatched rows should be kept or dropped.

You: zqxjkvb blorple fnord
Bot: I don't know that yet — it's outside my knowledge base. Try asking about: What is a CTE in SQL?; What is the difference between batch and stream processing?; What is a slowly changing dimension?. Or type 'help' to see example questions.

You: quit
Bot: Goodbye! Good luck with your learning.
```

Inside the chat, type `help` for example questions and `quit` / `exit` to leave.

## Optional LLM mode

```bash
export OPENAI_API_KEY=your-key-here
python chatbot.py --llm
```

With `--llm` and `OPENAI_API_KEY` set, questions are sent to an OpenAI-compatible chat completions endpoint using only `urllib` from the standard library (`llm_mode.py`). Without the key, it prints a clear message and falls back to offline mode automatically. Offline mode is the default and needs nothing.

## Running the tests

```bash
python -m unittest test_chatbot -v
```

All 6 tests pass: tokenization, medallion retrieval, SQL-joins retrieval, the honest fallback on gibberish, the greeting intent, and knowledge-base loading (≥ 20 entries).

## Project structure

```text
ai-faq-chatbot/
├── chatbot.py           # chat engine (tokenizer, from-scratch TF-IDF, cosine retrieval, intents) + CLI
├── knowledge_base.json  # 24 Q&A entries on data engineering / Python / SQL
├── llm_mode.py          # optional LLM mode via urllib (only used with --llm + API key)
├── test_chatbot.py      # 6 unit tests (built-in unittest)
└── requirements.txt     # no dependencies — standard library only
```

## Ideas to extend

- Add stemming or bigrams to the tokenizer for better matching
- Grow the knowledge base, or load entries from your own notes/docs
- Add a small web UI (e.g. with the standard library's `http.server`)
- Use the retrieval scores to return a top-3 list of candidate answers

---

A personal portfolio project by Santhosh Reddy Angiti demonstrating NLP fundamentals — tokenization, TF-IDF weighting, and cosine-similarity retrieval — implemented from scratch in Python.
