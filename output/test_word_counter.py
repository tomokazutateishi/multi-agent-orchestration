# test_word_counter.py

import unittest
from word_counter import count_words_and_characters


class TestWordCounter(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(count_words_and_characters(""), (0, 0))

    def test_single_word(self):
        self.assertEqual(count_words_and_characters("Hello"), (1, 5))

    def test_multiple_words(self):
        self.assertEqual(count_words_and_characters("Hello world"), (2, 11))

    def test_spaces_only(self):
        self.assertEqual(count_words_and_characters("    "), (0, 4))

    def test_mixed_characters(self):
        self.assertEqual(count_words_and_characters("Hello, world!"), (2, 13))


if __name__ == "__main__":
    unittest.main()
