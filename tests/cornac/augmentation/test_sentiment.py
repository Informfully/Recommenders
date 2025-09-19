import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from cornac.augmentation.sentiment import get_sentiment

class TestGetSentiment(unittest.TestCase):

    def test_positive_sentiment(self):
        text = "This is a fantastic news article!"
        result = get_sentiment(text)
        self.assertGreater(result, 0)

    def test_negative_sentiment(self):
        text = "The article was disappointing and frustrating."
        result = get_sentiment(text)
        self.assertLess(result, 0)

    def test_neutral_sentiment(self):
        text = "The report is about computer science."
        result = get_sentiment(text)
        self.assertAlmostEqual(result, 0, delta=1e-1)

    @patch('cornac.augmentation.sentiment._sentiment_analyzer')
    def test_positive_sentiment_mock(self, mock_sentiment_analyzer):
        text = "This is a fantastic news article!"
        mock_sentiment_analyzer.return_value = [
            [{'label': 'positive', 'score': 0.9}, {'label': 'negative', 'score': 0.1}]]

        result = get_sentiment(text)
        self.assertGreater(result, 0)

    @patch('cornac.augmentation.sentiment._sentiment_analyzer')
    def test_negative_sentiment_mock(self, mock_sentiment_analyzer):
        text = "The article was disappointing and frustrating."
        mock_sentiment_analyzer.return_value = [
            [{'label': 'positive', 'score': 0.1}, {'label': 'negative', 'score': 0.9}]]

        result = get_sentiment(text)
        self.assertLess(result, 0)

    @patch('cornac.augmentation.sentiment._sentiment_analyzer')
    def test_neutral_sentiment_mock(self, mock_sentiment_analyzer):
        text = "This is an informative piece of writing."
        mock_sentiment_analyzer.return_value = [
            [{'label': 'positive', 'score': 0.5}, {'label': 'negative', 'score': 0.5}]]

        result = get_sentiment(text)
        self.assertAlmostEqual(result, 0, delta=1e-2)

    def test_empty_text(self):
        result = get_sentiment("")
        self.assertIsNone(result)

    def test_non_string_input(self):
        result = get_sentiment(12345)  # Non-string input
        self.assertIsNone(result)

    def test_none_input(self):
        result = get_sentiment(None)  # None as input
        self.assertIsNone(result)

    @patch('cornac.augmentation.sentiment._sentiment_analyzer')
    def test_long_text(self, mock_sentiment_analyzer):
        text = "This is a sample sentence. " * 100  # Long text to test chunking
        mock_sentiment_analyzer.return_value = [
            [{'label': 'positive', 'score': 0.7}, {'label': 'negative', 'score': 0.3}]]

        result = get_sentiment(text)
        self.assertGreater(result, 0)

    @patch('cornac.augmentation.sentiment._sentiment_analyzer')
    def test_special_characters_text(self, mock_sentiment_analyzer):
        text = "!@#$%^&*()_+12345"  # Text with special characters and no clear sentiment
        mock_sentiment_analyzer.return_value = [
            [{'label': 'positive', 'score': 0.5}, {'label': 'negative', 'score': 0.5}]]

        result = get_sentiment(text)
        self.assertAlmostEqual(result, 0, delta=1e-2)

    @patch('cornac.augmentation.sentiment._sentiment_analyzer')
    def test_text_with_no_sentiment_label(self, mock_sentiment_analyzer):
        text = "A text with unknown labels."
        mock_sentiment_analyzer.return_value = [[{'label': 'neutral', 'score': 1.0}]]

        result = get_sentiment(text)
        self.assertIsNone(result)

    @patch('cornac.augmentation.sentiment._sentiment_analyzer')
    def test_large_text_chunk_handling(self, mock_sentiment_analyzer):
        text = "This is a very large text meant to test chunking. " * 200  # Extremely long text
        mock_sentiment_analyzer.return_value = [
            [{'label': 'positive', 'score': 0.7}, {'label': 'negative', 'score': 0.3}]]

        result = get_sentiment(text)
        self.assertGreater(result, 0)

    @patch('cornac.augmentation.sentiment._sentiment_analyzer')
    def test_error_handling_in_sentiment_analysis(self, mock_sentiment_analyzer):
        text = "This text will cause an error in sentiment analysis."
        mock_sentiment_analyzer.side_effect = Exception("Sentiment analysis error")

        # result = get_sentiment(text)
        with self.assertRaises(RuntimeError) as context:
            get_sentiment(text) 
        # self.assertIsNone(result)
        self.assertIn("Error calculating sentiment", str(context.exception))



if __name__ == "__main__":
    unittest.main()
