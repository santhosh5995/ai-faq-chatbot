"""Optional LLM mode for the AI FAQ Chatbot.

If OPENAI_API_KEY is set in the environment, questions can be sent to an
OpenAI-compatible chat completions endpoint using ONLY urllib from the
standard library. Without the key, the chatbot stays in offline mode.

This module is never used by the tests and never called automatically —
it only activates when the user runs:  python chatbot.py --llm
"""

import json
import os
import urllib.request

API_URL = os.environ.get(
    "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
)
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = (
    "You are a concise assistant that answers questions about data "
    "engineering, Python, and SQL in 2-3 sentences."
)

UNAVAILABLE_MESSAGE = (
    "LLM mode requested, but OPENAI_API_KEY is not set in the environment, "
    "so no LLM is available. Falling back to offline TF-IDF mode."
)


def available():
    """True when an API key is present in the environment."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def ask(question):
    """Send a question to the chat completions endpoint; return the answer."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()
