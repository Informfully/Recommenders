import unittest
import pandas as pd
from unittest.mock import patch
from cornac.augmentation.party import get_party

class TestGetParty(unittest.TestCase):

    def test_no_parties(self):
        ne_list = [
            {"John": {"frequency": 2}},
            {"Alice": {"frequency": 3}}
        ]
        result, lookup = get_party(ne_list, lang="en", lookup_parties={})
        self.assertEqual(result, {})

    def test_single_party(self):
        ne_list = [
            {"Joe": {"frequency": 4, "party": ["Democratic Party"]}},
            {"Alice": {"frequency": 3}}
        ]
        result, lookup = get_party(ne_list, lang="en", lookup_parties={})
        self.assertEqual(result, {"Democratic Party": 4})

    def test_multiple_parties(self):
        ne_list = [
            {"John": {"frequency": 2, "party": ["Republican Party"]}},
            {"Alice": {"frequency": 3}},
            {"Bob": {"frequency": 1, "party": ["Republican Party", "Independent"]}}
        ]
        result, lookup = get_party(ne_list, lang="en", lookup_parties={})
        self.assertEqual(result, {"Republican Party": 3, "independent politician": 1})

    def test_invalid_ne_list(self):
        # Set up non-list input
        ne_list = "invalid input"
        lang = "en"
        lookup_parties = {}
        with self.assertRaises(ValueError) as context:
            result, lookup = get_party(ne_list, lang=lang, lookup_parties=lookup_parties)
        # self.assertEqual(result, {})
        self.assertIn("Error: when extraing party, expected ne_list to be a list", str(context.exception))


    @patch('cornac.augmentation.party.get_english_label')
    def test_translation_of_parties(self, mock_get_english_label):
        # Mocking translation
        mock_get_english_label.side_effect = lambda party, lang: f"{party}_translated" if lang == "es" else party

        ne_list = [
            {"Joe": {"frequency": 2, "party": ["Partido Republicano"]}},
            {"Alice": {"frequency": 3, "party": ["Partido Democrático"]}}
        ]
        lookup_parties = {}
        result, lookup_parties = get_party(ne_list, lang="es", lookup_parties=lookup_parties)

        expected = {"Partido Republicano_translated": 2, "Partido Democrático_translated": 3}
        self.assertEqual(result, expected)
        self.assertEqual(lookup_parties, {
            "Partido Republicano": "Partido Republicano_translated",
            "Partido Republicano_translated": "Partido Republicano_translated",
            "Partido Democrático": "Partido Democrático_translated",
            "Partido Democrático_translated": "Partido Democrático_translated"
        })

    @patch('cornac.augmentation.party.get_english_label')
    def test_cached_translations(self, mock_get_english_label):
        # Translation should not call `get_english_label` for cached entries
        mock_get_english_label.side_effect = lambda party, lang: f"{party}_translated"

        ne_list = [
            {"Joe": {"frequency": 2, "party": ["Partido Republicano"]}},
            {"Alice": {"frequency": 3, "party": ["Partido Democrático"]}}
        ]
        lookup_parties = {
            "Partido Republicano": "Partido Republicano_translated",
            "Partido Democrático": "Partido Democrático_translated"
        }
        result, updated_lookup = get_party(ne_list, lang="es", lookup_parties=lookup_parties)

        expected = {"Partido Republicano_translated": 2, "Partido Democrático_translated": 3}
        self.assertEqual(result, expected)
        mock_get_english_label.assert_not_called()  # No new translations should be fetched


if __name__ == "__main__":
    unittest.main()
