import unittest
import pandas as pd
from unittest.mock import patch
from cornac.augmentation.min_maj import get_min_maj_ratio
from cornac.augmentation.story import get_story

class TestGetMinMajRatio(unittest.TestCase):

    def test_gender_scores_with_one_item(self):
        major_genders = ['male']
        major_citizens = []
        major_ethnicities = []
        major_place_of_births = []
        ne_list = [
            {'Kay Hagan': {'key': 'Kay Hagan', 'gender': ['female'], 'citizen': ['United States of America'],
                           'ethnicity': [], 'place_of_birth': ['United States of America'], 'frequency': 1}},
            {'Elizabeth Dole': {'key': 'Elizabeth Dole', 'gender': ['male'],
                                'citizen': ['United States of America'], 'ethnicity': [],
                                'place_of_birth': ['United States of America'], 'frequency': 1}}
        ]

        result = get_min_maj_ratio(ne_list, major_gender=major_genders, major_citizen=major_citizens,
                                   major_ethnicity=major_ethnicities, major_place_of_birth=major_place_of_births)
        self.assertListEqual(result['gender'], [0.5, 0.5])

    def test_gender_scores_with_multiple_item(self):
        major_genders = ['male', 'female']
        major_citizens = ['United States of America']
        major_ethnicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [
            {'Kay Hagan': {'key': 'Kay Hagan', 'gender': ['female'], 'citizen': ['United States of America'],
                           'ethnicity': [], 'place_of_birth': ['United States of America'], 'frequency': 1}},
            {'Elizabeth Dole': {'key': 'Elizabeth Dole', 'gender': ['male'],
                                'citizen': ['United States of America'],
                                'ethnicity': [], 'place_of_birth': ['United States of America'],
                                'frequency': 1}}
        ]

        result = get_min_maj_ratio(ne_list, major_gender=major_genders, major_citizen=major_citizens,
                                   major_ethnicity=major_ethnicities, major_place_of_birth=major_place_of_births)
        self.assertListEqual(result['gender'], [0, 1])

    def test_ethnicity_scores_only_with_citizen(self):
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethnicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [
            {'Kay Hagan': {'key': 'Kay Hagan', 'gender': ['female'], 'citizen': ['France'], 'ethnicity': [],
                           'place_of_birth': [], 'frequency': 1}},
            {'Elizabeth Dole': {'key': 'Elizabeth Dole', 'gender': ['male'], 'citizen': ['Mexico'],
                                'ethnicity': [], 'place_of_birth': [], 'frequency': 1}},
            {'Jackson': {'key': 'Jackson', 'gender': ['male'], 'citizen': ['United States of America'],
                         'ethnicity': [], 'place_of_birth': [], 'frequency': 1}}
        ]
        result = get_min_maj_ratio(ne_list, major_gender=major_genders, major_citizen=major_citizens,
                                   major_ethnicity=major_ethnicities, major_place_of_birth=major_place_of_births)
        self.assertListEqual(result['ethnicity'], [0.6667, 0.3333])

    def test_ethnicity_scores_with_citizen_and_ethnicity(self):
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethnicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [
            {'Kay Hagan': {'key': 'Kay Hagan', 'gender': ['female'], 'citizen': ['United States of America'],
                           'ethnicity': ['white people'], 'place_of_birth': [], 'frequency': 1}},
            {'Elizabeth Dole': {'key': 'Elizabeth Dole', 'gender': ['male'],
                                'citizen': ['United States of America'],
                                'ethnicity': [], 'place_of_birth': [], 'frequency': 1}},
            {'Joe Biden': {'key': 'Joe Biden', 'gender': ['male'], 'citizen': ['United States of America'],
                           'ethnicity': ['Jewish'], 'place_of_birth': [], 'frequency': 1}}
        ]
        result = get_min_maj_ratio(ne_list, major_gender=major_genders, major_citizen=major_citizens,
                                   major_ethnicity=major_ethnicities, major_place_of_birth=major_place_of_births)
        self.assertListEqual(result['ethnicity'], [0.3333, 0.6667])

    def test_ethnicity_scores_with_citizen_and_place_of_birth(self):
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethnicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [
            {'Kay Hagan': {'key': 'Kay Hagan', 'gender': ['female'], 'citizen': ['United States of America'],
                           'ethnicity': ['white people'], 'place_of_birth': ['United States of America'],
                           'frequency': 1}},
            {'Elizabeth Dole': {'key': 'Elizabeth Dole', 'gender': ['male'],
                                'citizen': ['United States of America'],
                                'ethnicity': [], 'place_of_birth': ['Spain'], 'frequency': 1}}
        ]
        result = get_min_maj_ratio(ne_list, major_gender=major_genders, major_citizen=major_citizens,
                                   major_ethnicity=major_ethnicities, major_place_of_birth=major_place_of_births)
        self.assertListEqual(result['ethnicity'], [0.5, 0.5])

    def test_mainstream_scores(self):
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethnicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [
            {'Kay Hagan': {'key': 'Kay Hagan', 'givenname': ['Kay'], 'gender': ['female'],
                           'citizen': ['United States of America'], 'ethnicity': [],
                           'place_of_birth': ['United States of America'], 'frequency': 1}},
            {'Greensboro': {'key': 'Greensboro', 'frequency': 1}}
        ]
        result = get_min_maj_ratio(ne_list, major_gender=major_genders, major_citizen=major_citizens,
                                   major_ethnicity=major_ethnicities, major_place_of_birth=major_place_of_births)
        self.assertListEqual(result['mainstream'], [0.5, 0.5])

    def test_empty_ne_list(self):
        """Test with an empty named entity list."""
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethnicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = []

        result = get_min_maj_ratio(
            ne_list,
            major_gender=major_genders,
            major_citizen=major_citizens,
            major_ethnicity=major_ethnicities,
            major_place_of_birth=major_place_of_births
        )
        self.assertEqual(result, {})

    def test_invalid_ne_list(self):
        """Test with invalid input for named entity list."""
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethnicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = "invalid_input"

        with self.assertRaises(TypeError):
            get_min_maj_ratio(
                ne_list,
                major_gender=major_genders,
                major_citizen=major_citizens,
                major_ethnicity=major_ethnicities,
                major_place_of_birth=major_place_of_births
            )

    def test_missing_keys_in_entities(self):
        """Test when some entities lack expected keys."""
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethnicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [
            {'Unknown': {'key': 'Unknown', 'frequency': 1}},
            {'John Doe': {'key': 'John Doe', 'gender': ['male'], 'frequency': 1}}
        ]

        result = get_min_maj_ratio(
            ne_list,
            major_gender=major_genders,
            major_citizen=major_citizens,
            major_ethnicity=major_ethnicities,
            major_place_of_birth=major_place_of_births
        )
        self.assertEqual(result['gender'], [0, 1])

    def test_missing_key_in_entities(self):
        """Test when some entities lack expected keys."""
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethnicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [
            {'Unknown': {'gender': ['female'], 'citizen': ['United States of America'], 'ethnicity': ['white people'],
                         'place_of_birth': ['United States of America'], 'givenname': ['Unknown'], 'frequency': 1}},
            {'John Doe': {'key': 'Elizabeth Dole', 'gender': ['male'], 'citizen': ['United Kingdom'],
                          'ethnicity': ['black people'], 'place_of_birth': ['Spain'], 'frequency': 1}}
        ]

        result = get_min_maj_ratio(
            ne_list,
            major_gender=major_genders,
            major_citizen=major_citizens,
            major_ethnicity=major_ethnicities,
            major_place_of_birth=major_place_of_births
        )
        self.assertEqual(result, {'gender': [0, 1], 'ethnicity': [1, 0], 'mainstream': [1, 0]})

    def test_combined_scores_with_partial_data(self):
        """Test when entities have partial data for various attributes."""
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethnicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [
            {'Jane Doe': {'key': 'Jane Doe', 'gender': ['female'], 'frequency': 1}},
            {'John Smith': {'key': 'John Smith', 'citizen': ['United States of America'], 'frequency': 1}},
            {'Chris': {'key': 'Chris', 'place_of_birth': ['United States of America'], 'frequency': 1}}
        ]

        result = get_min_maj_ratio(
            ne_list,
            major_gender=major_genders,
            major_citizen=major_citizens,
            major_ethnicity=major_ethnicities,
            major_place_of_birth=major_place_of_births
        )
        self.assertEqual(result['gender'], [1.0, 0.0])
        self.assertEqual(result['ethnicity'], [0.0, 1.0])
        self.assertEqual(result['mainstream'], [1.0, 0.0])

    def test_frequency_aggregation(self):
        """Test if frequencies are aggregated correctly."""
        major_genders = ['male']
        major_citizens = ['United States of America']
        major_ethnicities = ['white people']
        major_place_of_births = ['United States of America']
        ne_list = [
            {'Person A': {'key': 'Person A', 'gender': ['male'], 'frequency': 3}},
            {'Person B': {'key': 'Person B', 'gender': ['female'], 'frequency': 2}},
            {'Person C': {'key': 'Person C', 'gender': ['male'], 'frequency': 1}}
        ]

        result = get_min_maj_ratio(
            ne_list,
            major_gender=major_genders,
            major_citizen=major_citizens,
            major_ethnicity=major_ethnicities,
            major_place_of_birth=major_place_of_births
        )
        self.assertEqual(result['gender'], [0.3333, 0.6667])

    @patch("cornac.augmentation.min_maj.get_min_maj_ratio")
    def test_with_mocked_function(self, mock_get_min_maj_ratio):
        """Test with a mocked version of get_min_maj_ratio."""
        mock_get_min_maj_ratio.return_value = {
            'gender': [0.4, 0.6],
            'ethnicity': [0.2, 0.8],
            'mainstream': [0.5, 0.5]
        }
        result = mock_get_min_maj_ratio([], major_gender=[], major_citizen=[], major_ethnicity=[],
                                        major_place_of_birth=[])
        self.assertEqual(result['gender'], [0.4, 0.6])
        self.assertEqual(result['ethnicity'], [0.2, 0.8])
        self.assertEqual(result['mainstream'], [0.5, 0.5])

if __name__ == "__main__":
    unittest.main()
