import unittest
import pandas as pd
from unittest.mock import patch
from cornac.augmentation.readability import get_readability

class TestGetReadability(unittest.TestCase):

    @patch('textstat.flesch_reading_ease')
    def test_valid_text_english(self, mock_flesch_reading_ease):
        sample_text = 'Some Supreme Court justices thought Donald Trump was setting them up...'
        mock_flesch_reading_ease.return_value = 65.0  # Sample readability score

        result = get_readability(sample_text, lang="en")
        self.assertAlmostEqual(result, 65.0, delta=1e-2)

    @patch('textstat.flesch_reading_ease')
    def test_valid_text_german(self, mock_flesch_reading_ease):
        sample_text = 'Einige Oberste Richter glaubten, dass Donald Trump sie austrickste...'
        mock_flesch_reading_ease.return_value = 50.0  # Sample readability score

        result = get_readability(sample_text, lang="de")
        self.assertAlmostEqual(result, 50.0, delta=1e-2)

    def test_valid_text(self):
        sample_text = 'Some Supreme Court justices thought Donald Trump was setting them up. Two days after the official swearing-in of Justice Brett Kavanaugh in October 2018, the president arranged a televised ceremony at the White House and invited all the justices.'
        result = get_readability(sample_text, lang="en")
        self.assertIsInstance(result, (float, int))

    def test_empty_text(self):
        result = get_readability("", lang="en")
        self.assertIsNone(result)

    def test_invalid_text_type(self):
        # result = get_readability(12345, lang="en")  # Non-string input
        # self.assertIsNone(result)
        """Test with invalid input type (non-string)"""
        with self.assertRaises(TypeError):
            get_readability(12345, lang="en")  # Integer input should raise TypeError

        with self.assertRaises(TypeError):
            get_readability(12.34, lang="en")  # Float input should raise TypeError

        with self.assertRaises(TypeError):
            get_readability(None, lang="en")  # None input should raise TypeError

        with self.assertRaises(TypeError):
            get_readability([], lang="en")  # List input should raise TypeError



    @patch('textstat.avg_sentence_length', return_value=20)
    @patch('textstat.avg_syllables_per_word', return_value=1.5)
    def test_fallback_formula_for_unknown_language(self, mock_avg_sentence_length, mock_avg_syllables_per_word):
        sample_text = 'Este es un texto en un idioma no soportado por textstat.'  # Assume it's Galician (gl)

        result = get_readability(sample_text, lang="gl")
        expected_score = 227 - (1.04 * 20) - (72 * 1.5)
        self.assertAlmostEqual(result, expected_score, delta=1e-2)

    def test_readability_text_with_special_characters(self):
        sample_text = "!@#$%^&*()_+12345"  # Text with special characters and no readable content
        result = get_readability(sample_text, lang="en")
        self.assertIsNone(result)

    def test_non_latin_language(self):
        sample_text = "ఈ వచనం లాటిన్ యేతర వర్ణమాల భాషలో ఉంది, అయితే ఇది ఇప్పటికీ చదవగలిగే స్కోర్‌ను అవుట్‌పుట్ చేయాలి."
        result = get_readability(sample_text, lang="te")
        self.assertIsInstance(result, (float, int))

    def test_readability_long_text(self):
        sample_text = " ".join(["This is a sample sentence."] * 100)  # Very long text
        result = get_readability(sample_text, lang="en")
        self.assertIsInstance(result, (float, int))

    def test_readability_for_multi_region_lang_code(self):
        sample_text = "Some example text for multi-region language code."
        result = get_readability(sample_text, lang="en_US")  # Test for multi-region code handling
        self.assertIsInstance(result, (float, int))

    def test_invalid_language_code(self):
        sample_text = "A random text with invalid language code."
        # Unsupported language code
        with self.assertRaises(ValueError):
            get_readability(sample_text, lang="xyz")


if __name__ == "__main__":
    unittest.main()
