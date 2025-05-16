import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from concurrent.futures import Future
from cornac.augmentation.story import get_story

class TestGetStory(unittest.TestCase):

    def test_get_story_single_category(self):
        text1 = 'The sun sets over the horizon, painting the sky with hues of orange and pink.'
        text2 = 'As the day comes to a close, the sun dips below the edge of the earth.'
        text3 = 'Beneath a sky painted with the last remnants of daylight, the sun bids its farewell.'

        sample_data = {
            'id': [1, 2, 3],
            'date': ['2019-10-01', '2019-10-02', '2019-10-03'],
            'category': ['A', 'A', 'A'],
            'text': [text1, text2, text3],
        }
        sample_df = pd.DataFrame(sample_data)

        result_df = get_story(sample_df)

        # Assert that all stories belong to the same cluster (story ID 0)
        self.assertTrue((result_df['story'] == 0).all())

    def test_get_story_multiple_categories(self):
        text1 = 'The sun sets over the horizon, painting the sky with hues of orange and pink. Birds fly homeward, their silhouettes darkening against the vibrant backdrop. A sense of calm settles in as the world transitions from day to night.'
        text2 = 'As the day comes to a close, the sun dips below the edge of the earth, casting a warm glow across the landscape. Flocks of birds soar through the air, heading towards their nests. The tranquil beauty of twilight envelops everything in a serene embrace.'
        text3 = 'Beneath a sky painted with the last remnants of daylight, the sun bids its farewell, casting an amber glow over the surroundings. The air becomes cooler, and the sounds of rustling leaves and distant streams fill the atmosphere. Natures rhythm embraces the world in a soothing lullaby as night gradually takes hold.'

        sample_data = {
            'id': [1, 2, 3],
            'date': ['2019-10-01', '2019-10-10', '2019-10-03'],
            'category': ['A', 'B', 'A'],
            'text': [text1, text2, text3],
        }
        sample_df = pd.DataFrame(sample_data)

        result_df = get_story(sample_df)

        # Assert that stories are correctly identified across categories
        self.assertEqual(result_df['story'].tolist(), [0, 1, 0])

    def test_get_story_empty_dataframe(self):
        sample_df = pd.DataFrame(columns=['id', 'date', 'category', 'text'])
        result_df = get_story(sample_df)

        # Assert that the output DataFrame is empty
        self.assertTrue(result_df.empty)

    def test_get_story_missing_columns(self):
        sample_data = {'date': ['2019-10-01'], 'category': ['A']}
        sample_df = pd.DataFrame(sample_data)

        with self.assertRaises(ValueError) as context:
            get_story(sample_df)

        # Assert the error message
        self.assertIn("The input DataFrame must contain a 'id' column", str(context.exception))

    @patch("cornac.augmentation.story.TfidfVectorizer.fit_transform")
    def test_get_story_mocked_vectorizer(self, mock_fit_transform):
        mock_fit_transform.return_value = [[1, 0], [0, 1]]  # Simulate a valid output
        sample_data = {
            'id': [1],
            'date': ['2019-10-01'],
            'category': ['A'],
            'text': ['Mocked text for testing.'],
        }
        sample_df = pd.DataFrame(sample_data)

        result_df = get_story(sample_df)
        self.assertEqual(result_df['story'].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
