"""Tests for the AI FAQ Chatbot. Uses only the built-in unittest module.

Run with:  python -m unittest discover -v
       or:  python -m unittest test_chatbot -v
"""

import unittest

from chatbot import Chatbot, tokenize

FALLBACK_MARKER = "I don't know that yet"


class TestChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = Chatbot()

    def test_tokenizer_removes_stopwords_and_punctuation(self):
        tokens = tokenize("What is the ETL process?!")
        self.assertIn("etl", tokens)
        self.assertIn("process", tokens)
        self.assertNotIn("what", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("the", tokens)
        self.assertTrue(all(t == t.lower() for t in tokens))

    def test_medallion_question_retrieves_medallion_answer(self):
        answer = self.bot.answer("What is the medallion architecture?")
        self.assertIn("Bronze", answer)
        self.assertIn("Silver", answer)
        self.assertIn("Gold", answer)
        self.assertNotIn(FALLBACK_MARKER, answer)

    def test_sql_joins_question_retrieves_joins_answer(self):
        answer = self.bot.answer("What are the different types of SQL joins?")
        self.assertIn("INNER JOIN", answer)
        self.assertIn("LEFT JOIN", answer)
        self.assertNotIn(FALLBACK_MARKER, answer)

    def test_gibberish_question_triggers_fallback(self):
        answer = self.bot.answer("zqxjkvb blorple fnord wizzle")
        self.assertIn(FALLBACK_MARKER, answer)

    def test_greeting_intent(self):
        answer = self.bot.answer("Hello!")
        self.assertIn("Hello", answer)
        self.assertNotIn(FALLBACK_MARKER, answer)

    def test_knowledge_base_has_at_least_20_entries(self):
        self.assertGreaterEqual(len(self.bot.entries), 20)
        for entry in self.bot.entries:
            self.assertIn("question", entry)
            self.assertIn("answer", entry)


if __name__ == "__main__":
    unittest.main()
