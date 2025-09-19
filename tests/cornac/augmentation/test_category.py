import unittest
import pandas as pd
from unittest.mock import patch
from cornac.augmentation.category import get_category

class TestGetCategory(unittest.TestCase):

    @patch('cornac.augmentation.category._classifier')
    def test_with_candidate_labels_high_confidence(self, mock_classifier):
        user_labels = ["news", "sports", "life"]
        sample_text = "In an adrenaline-charged match, the Springville Strikers snatched a thrilling win."

        # Mock classifier response with high confidence in 'sports' label
        mock_classifier.return_value = {'labels': ["sports", "news", "life"], 'scores': [0.8, 0.15, 0.05]}

        result = get_category(sample_text, candidate_labels=user_labels)
        self.assertEqual(result, "sports")

    @patch('cornac.augmentation.category._classifier')
    def test_with_candidate_labels_low_confidence(self, mock_classifier):
        user_labels = ["news", "sports", "life"]
        sample_text = "A very ambiguous statement."

        # Mock classifier response with low confidence
        mock_classifier.return_value = {'labels': ["news", "sports", "life"], 'scores': [0.3, 0.2, 0.1]}

        result = get_category(sample_text, candidate_labels=user_labels)
        self.assertEqual(result, -1)

    def test_with_candidate_labels(self):
        user_labels = ["news", "sports", "life"]
        sample_text = "In an adrenaline-charged match, the Springville Strikers snatched a thrilling win against the Riverside Rovers with a dramatic last-minute goal. The scoreboard read 2-1 in favor of the Strikers as the final whistle blew, leaving fans in awe."
        result = get_category(sample_text, candidate_labels=user_labels)
        self.assertEqual(result, "sports")

    def test_with_candidate_labels_not_found(self):
        user_labels = ["news", "sports", "life"]
        sample_text = "A very ambiguous statement."
        result = get_category(sample_text, candidate_labels=user_labels)
        self.assertEqual(result, -1)

    def test_without_candidate_labels(self):
        meta_data = pd.DataFrame({
            "id": [1, 2, 3],
            "category": ["news", "sports", "life"]
        })
        sample_id = 3
        result = get_category(sample_id, meta_data=meta_data)
        self.assertEqual(result, "life")

    def test_without_candidate_labels_non_matching_id(self):
        meta_data = pd.DataFrame({
            "id": [1, 2, 3],
            "category": ["news", "sports", "life"]
        })
        sample_id = 5
        result = get_category(sample_id, meta_data=meta_data)
        self.assertEqual(result, -1)

    def test_empty_candidate_labels(self):
        sample_text = "This is a sample text about science and technology."
        result = get_category(sample_text, candidate_labels=[])
        self.assertEqual(result, -1)

    def test_invalid_input_type_for_row(self):
        user_labels = ["news", "sports", "life"]
        sample_text = 12345  # invalid type, expecting string or int
        with self.assertRaises(TypeError):
            get_category(sample_text, candidate_labels=user_labels)

    def test_missing_both_candidate_labels_and_meta_data(self):
        sample_text = "This is a random news text without labels or metadata."
        result = get_category(sample_text)
        self.assertEqual(result, -1)

    def test_with_mismatched_meta_data_structure(self):
        # Test case with a DataFrame missing 'category' column
        meta_data = pd.DataFrame({
            "id": [1, 2, 3],
            "title": ["Title1", "Title2", "Title3"]
        })
        sample_id = 2
        with self.assertRaises(KeyError):
            get_category(sample_id, meta_data=meta_data)




if __name__ == "__main__":
    unittest.main()
