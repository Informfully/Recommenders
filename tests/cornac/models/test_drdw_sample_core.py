import unittest
import numpy as np
import pandas as pd
from cornac.models.drdw.sample_core import DistributionSampler, processPartyData


class TestSampleCore(unittest.TestCase):
    def setUp(self):
        self.article_df = pd.DataFrame({
            'category': ['Politics', 'Sports', 'Technology', 'Politics', 'Technology', 'Sports', 'Politics'],
            'outlet': ['Outlet A', 'Outlet B', 'Outlet A', 'Outlet A', 'Outlet B', 'Outlet A', 'Outlet A'],
            'sentiment': [-0.5, -0.3, -0.2, 0.1, 0.8, 0.5, -0.7],
            'entities': [
                "['Republican', 'Democratic']",  # Both parties
                "['Republican']",  # Only Republican
                "['Democratic']",  # Only Democratic
                "['Independent']",  # Minority
                None,  # No party
                "['Republican', 'Democratic']",  # Both parties
                "['Republican', 'Democratic']"]  # Both parties

        }, index=[0, 1, 2, 3, 4, 5, 6])
        self.rdw_sampler = DistributionSampler(
            item_dataframe=self.article_df)

    def test_processPartyData(self):
        # Test for processing party data strings
        test_str = "['Republican', 'Democratic']"
        result = processPartyData(test_str)
        expected = ['Republican', 'Democratic']
        self.assertEqual(result, expected)

        # Test for missing party data
        test_str = None
        result = processPartyData(test_str)
        # self.assertIsNone(result)
        self.assertEqual(len(result), 0)

        # Test when party data is a list
        test_list = ['Independent']
        result = processPartyData(test_list)
        self.assertEqual(result, test_list)

    def test_items_per_discrete_attribute(self):
        tar = {"sports": 0.3, "music": 0.1, "politics": 0.6}
        targetSize = 100
        description = "category"
        expected_output = {"category,sports": 30,
                           "category,music": 10, "category,politics": 60}
        result = self.rdw_sampler.items_per_discrete_attribute(
            tar, targetSize, description)
        self.assertEqual(result, expected_output)

    # Test case with valid distribution, rounding adjustment needed.
    def test_items_per_discrete_attribute_rounding(self):
        tar = {"sports": 0.33, "music": 0.16, "politics": 0.51}
        targetSize = 55
        description = "category"
        expected_output = {"category,sports": 18,
                           "category,music": 9, "category,politics": 28}
        result = self.rdw_sampler.items_per_discrete_attribute(
            tar, targetSize, description)
        self.assertEqual(result, expected_output)

    def test_items_per_discrete_attribute_invalid(self):
       # Test case where sum of distribution does not equal 1
        tar = {"sports": 0.33, "music": 0.16, "politics": 0.1}  # Sum is 0.59
        targetSize = 55
        description = "category"
        with self.assertRaises(ValueError, msg="Sum of the distribution values must equal 1."):
            self.rdw_sampler.items_per_discrete_attribute(
                tar, targetSize, description)

        # Test case where one distribution value is greater than 1
        tar = {"sports": 1.2, "music": 0.2, "politics": 0.7}  # sports > 1
        targetSize = 100
        description = "category"
        with self.assertRaises(ValueError, msg="Distribution value for 'sports' is not between 0 and 1."):
            self.rdw_sampler.items_per_discrete_attribute(
                tar, targetSize, description)

        # Test case where one distribution value is negative
        tar = {"sports": -0.1, "music": 0.5, "politics": 0.6}  # sports < 0
        targetSize = 100
        description = "category"
        with self.assertRaises(ValueError, msg="Distribution value for 'sports' is not between 0 and 1."):
            self.rdw_sampler.items_per_discrete_attribute(
                tar, targetSize, description)

    def test_items_per_continous_attribute(self):

        tarList = [
            {'prob': 0.33, 'min': 0, 'max': 10},
            {'prob': 0.33, 'min': 10, 'max': 20},
            {'prob': 0.34, 'min': 20, 'max': 30}
        ]
        targetSize = 100
        description = 'sentiment'

        result = self.rdw_sampler.items_per_continous_attribute(
            tarList, targetSize, description)

        self.assertEqual(sum(result.values()), 100)
        expected_output = {
            'sentiment,0,10': 33,
            'sentiment,10,20': 33,
            'sentiment,20,30': 34
        }
        self.assertEqual(result, expected_output)

        # Test when rounding and adjustment is needed
        tarList = [
            {'prob': 0.23, 'min': 0, 'max': 10},
            {'prob': 0.37, 'min': 10, 'max': 20},
            {'prob': 0.4, 'min': 20, 'max': 30}
        ]
        targetSize = 10
        description = 'sentiment'

        result = self.rdw_sampler.items_per_continous_attribute(
            tarList, targetSize, description)

        self.assertEqual(sum(result.values()), targetSize)
        expected_output = {
            'sentiment,0,10': 2,
            'sentiment,10,20': 4,
            'sentiment,20,30': 4
        }
        self.assertEqual(result, expected_output)

        # Test case where sum of distribution does not equal 1
        tarList = [
            {'prob': 0.4, 'min': 0, 'max': 10},
            {'prob': 0.3, 'min': 10, 'max': 20},  # Sum = 0.9 (should be 1)
            {'prob': 0.2, 'min': 20, 'max': 30}
        ]
        targetSize = 100
        description = 'sentiment'
        with self.assertRaises(ValueError, msg="Sum of the distribution values must equal 1."):
            self.rdw_sampler.items_per_continous_attribute(
                tarList, targetSize, description)

        # Test case where one distribution value is greater than 1
        tarList = [
            {'prob': 1.1, 'min': 0, 'max': 10},  # prob > 1
            {'prob': 0.2, 'min': 10, 'max': 20},
            {'prob': 0.7, 'min': 20, 'max': 30}
        ]
        targetSize = 100
        description = 'sentiment'
        with self.assertRaises(ValueError, msg="Distribution value for range 0-10 is not between 0 and 1."):
            self.rdw_sampler.items_per_continous_attribute(
                tarList, targetSize, description)

        # Test case where one distribution value is negative
        tarList = [
            {'prob': -0.1, 'min': 0, 'max': 10},  # prob < 0
            {'prob': 0.5, 'min': 10, 'max': 20},
            {'prob': 0.6, 'min': 20, 'max': 30}
        ]
        targetSize = 100
        description = 'sentiment'
        with self.assertRaises(ValueError, msg="Distribution value for range 0-10 is not between 0 and 1."):
            self.rdw_sampler.items_per_continous_attribute(
                tarList, targetSize, description)

    def test_items_per_party_classification(self):
        tarList = [
            {'prob': 0.24, 'description': 'only mention',
                'contain': ['Republican']},
            {'prob': 0.25, 'description': 'only mention',
                'contain': ['Democratic']},
            {'prob': 0.30, 'description': 'only mention',
                'contain': ['Republican', 'Democratic']},
            {'prob': 0.11, 'description': 'minority but can also mention',
                'contain': ['Republican', 'Democratic']},
            {'prob': 0.1, 'description': 'No parties',
                'contain': []}
        ]
        targetSize = 100
        description = 'party'
        # Expected output based on the input target list
        expected_output = {
            'party,only mention:Republican': 24,
            'party,only mention:Democratic': 25,
            'party,only mention:Republican,Democratic': 30,
            'party,minority but can also mention:Republican,Democratic': 11,
            'party,No parties:': 10
        }

        result = self.rdw_sampler.items_per_party_classification(
            tarList, targetSize, description)
        self.assertEqual(result, expected_output)

        # Now test for rounding where the floor sum is less than the targetSize and adjustment is needed
        tarList = [
            {'prob': 0.33, 'description': 'only mention',
                'contain': ['Republican']},
            {'prob': 0.33, 'description': 'only mention',
                'contain': ['Democratic']},
            {'prob': 0.34, 'description': 'only mention',
                'contain': ['Republican', 'Democratic']}
        ]
        targetSize = 90
        description = 'party'

        # For targetSize = 90:
        # np.floor(0.33 * 90) = 29 for both 'Republican' and 'Democratic'
        # np.floor(0.34 * 90) = 30 for 'Republican,Democratic'
        # Total = 29 + 29 + 30 = 88, so we need to adjust by increasing the largest proportion categories by 2
        # The largest categories ('party,only mention:Republican' and 'party,only mention:Democratic') should be adjusted.

        expected_output = {
            'party,only mention:Republican': 30,
            'party,only mention:Democratic': 30,
            'party,only mention:Republican,Democratic': 30
        }

        result = self.rdw_sampler.items_per_party_classification(
            tarList, targetSize, description)
        self.assertEqual(result, expected_output)

        # Test where the sum of probabilities doesn't equal 1
        tarList = [
            {'prob': 0.2, 'description': 'only mention',
                'contain': ['Republican']},
            {'prob': 0.25, 'description': 'only mention',
                'contain': ['Democratic']},
            {'prob': 0.3, 'description': 'only mention', 'contain': [
                'Republican', 'Democratic']}  # Sum = 0.75
        ]
        targetSize = 100
        description = 'party'

        with self.assertRaises(ValueError, msg="Sum of the distribution values must equal 1."):
            self.rdw_sampler.items_per_party_classification(
                tarList, targetSize, description)

        # Test case where one distribution value is greater than 1
        tarList = [
            {'prob': 1.2, 'description': 'only mention',
                'contain': ['Republican']},  # Invalid prob > 1
            {'prob': 0.2, 'description': 'only mention',
                'contain': ['Democratic']}
        ]
        targetSize = 100
        description = 'party'

        with self.assertRaises(ValueError, msg="Distribution value for party only mention is not between 0 and 1."):
            self.rdw_sampler.items_per_party_classification(
                tarList, targetSize, description)

        # Test case where one distribution value is negative
        tarList = [
            {'prob': -0.1, 'description': 'only mention',
                'contain': ['Republican']},  # Invalid prob < 0
            {'prob': 0.6, 'description': 'only mention',
                'contain': ['Democratic']}
        ]
        targetSize = 100
        description = 'party'

        with self.assertRaises(ValueError, msg="Distribution value for party only mention is not between 0 and 1."):
            self.rdw_sampler.items_per_party_classification(
                tarList, targetSize, description)

    def test_cache_with_different_parameters(self):
        # Test caching for discrete attributes
        tar1 = {"sports": 0.5, "music": 0.3, "politics": 0.2}
        tar2 = {"sports": 0.4, "music": 0.4, "politics": 0.2}
        targetSize = 10
        description = "category"

        # Call with first set of parameters for discrete attributes
        self.rdw_sampler.items_per_discrete_attribute(
            tar1, targetSize, description)
        # Ensure it's cached
        cache_key_1 = self.rdw_sampler._generate_cache_key(
            'discrete', description, tar1)
        self.assertIn(
            cache_key_1, self.rdw_sampler.target_num_items_per_category)

        # Call with a different set of parameters
        self.rdw_sampler.items_per_discrete_attribute(
            tar2, targetSize, description)
        # Ensure the new result is cached under a different key
        cache_key_2 = self.rdw_sampler._generate_cache_key(
            'discrete', description, tar2)
        self.assertIn(
            cache_key_2, self.rdw_sampler.target_num_items_per_category)
        self.assertNotEqual(cache_key_1, cache_key_2)

        # check the return value is same with cached value
        # Call with a different set of parameters
        result = self.rdw_sampler.items_per_discrete_attribute(
            tar1, targetSize, description)
        # Ensure the new result is cached under a different key
        expected_result = self.rdw_sampler.target_num_items_per_category[cache_key_1]
        self.assertEqual(result, expected_result)

        # Test caching for continuous attributes
        tarList1 = [
            {'prob': 0.33, 'min': 0, 'max': 10},
            {'prob': 0.33, 'min': 10, 'max': 20},
            {'prob': 0.34, 'min': 20, 'max': 30}
        ]
        tarList2 = [
            {'prob': 0.4, 'min': 0, 'max': 10},
            {'prob': 0.3, 'min': 10, 'max': 20},
            {'prob': 0.3, 'min': 20, 'max': 30}
        ]
        targetSize = 100
        description = 'sentiment'

        # Call with first set of parameters for continuous attributes
        self.rdw_sampler.items_per_continous_attribute(
            tarList1, targetSize, description)
        cache_key_cont1 = self.rdw_sampler._generate_cache_key(
            'continuous', description, tarList1)
        self.assertIn(cache_key_cont1,
                      self.rdw_sampler.target_num_items_per_category)

        # Call with a different set of parameters
        self.rdw_sampler.items_per_continous_attribute(
            tarList2, targetSize, description)
        cache_key_cont2 = self.rdw_sampler._generate_cache_key(
            'continuous', description, tarList2)
        self.assertIn(cache_key_cont2,
                      self.rdw_sampler.target_num_items_per_category)
        self.assertNotEqual(cache_key_cont1, cache_key_cont2)

        # Test caching for party classification
        tarParty1 = [
            {'prob': 0.1, 'description': 'only mention',
                'contain': ['Republican']},
            {'prob': 0.9, 'description': 'minority but can also mention',
                'contain': ['Republican', 'Democratic']}
        ]
        tarParty2 = [
            {'prob': 0.2, 'description': 'only mention',
                'contain': ['Republican']},
            {'prob': 0.8, 'description': 'no parties', 'contain': []}
        ]
        targetSize = 100
        description = 'party'

        # Call with first set of parameters for party classification
        self.rdw_sampler.items_per_party_classification(
            tarParty1, targetSize, description)
        cache_key_party1 = self.rdw_sampler._generate_cache_key(
            'party', description, tarParty1)
        self.assertIn(cache_key_party1,
                      self.rdw_sampler.target_num_items_per_category)

        # Call with a different set of parameters
        self.rdw_sampler.items_per_party_classification(
            tarParty2, targetSize, description)
        cache_key_party2 = self.rdw_sampler._generate_cache_key(
            'party', description, tarParty2)
        self.assertIn(cache_key_party2,
                      self.rdw_sampler.target_num_items_per_category)
        self.assertNotEqual(cache_key_party1, cache_key_party2)

    def test_generateMaskedMatrixDiscrete(self):

        itemPool = np.asarray([0, 1, 2, 4, 6])

        items_per_category = {"category,Sports": 1,
                              "category,Technology": 2, "category,Politics": 2}

        # Map the index in itemPool (0, 1, ..., len(itemPool) - 1) to original item IDs
        newIndex = np.arange(len(itemPool))
        newId_to_cornacId = dict(enumerate(itemPool))

        cornacId_to_newId = dict(zip(itemPool, newIndex))

        expected_dict = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4}
        self.assertEqual(cornacId_to_newId, expected_dict)

        result = self.rdw_sampler.generateMaskedMatrixDiscrete(
            self.article_df, itemPool, 'category', items_per_category, cornacId_to_newId)
        # Expecting masks for Politics, Sports, and Technology categories

        self.assertTrue(np.array_equal(
            result['category,Politics'], np.array([1, 0, 0, 0, 1])))
        self.assertTrue(np.array_equal(
            result['category,Sports'], np.array([0, 1, 0, 0, 0])))
        self.assertTrue(np.array_equal(
            result['category,Technology'], np.array([0, 0, 1, 1, 0])))

    def test_generateMaskedMatrixContinous(self):
        # Setup for continuous attribute
        self.article_df['age'] = [5, 15, 25,
                                  35, 11, 55, 1]  # Suppose Age ranges
        items_per_category = {
            'age_range,0,10': 1,
            'age_range,10,30': 2,
            'age_range,30,50': 3
        }
        itemPool = np.asarray([0, 3, 4, 6, 2])
        newIndex = np.arange(len(itemPool))
        newId_to_cornacId = dict(enumerate(itemPool))
        cornacId_to_newId = dict(zip(itemPool, newIndex))

        result = self.rdw_sampler.generateMaskedMatrixContinous(
            self.article_df, itemPool, 'age', items_per_category, cornacId_to_newId)

        # Test mask for different age ranges
        self.assertTrue(np.array_equal(
            result['age_range,0,10'], np.array([1, 0, 0, 1, 0])))
        self.assertTrue(np.array_equal(
            result['age_range,10,30'], np.array([0, 0, 1, 0, 1])))
        self.assertTrue(np.array_equal(
            result['age_range,30,50'], np.array([0, 1, 0, 0, 0])))

        items_per_category = {
            'sentiment,-1,-0.5': 1,
            'sentiment,-0.5,0': 2,
            'sentiment,0,0.5': 2
        }
        itemPool = np.asarray([1, 2, 0, 3, 6])
        newIndex = np.arange(len(itemPool))
        newId_to_cornacId = dict(enumerate(itemPool))
        cornacId_to_newId = dict(zip(itemPool, newIndex))
        result = self.rdw_sampler.generateMaskedMatrixContinous(
            self.article_df, itemPool, 'sentiment', items_per_category, cornacId_to_newId)

        # Test mask for different age ranges
        self.assertTrue(np.array_equal(
            result['sentiment,-1,-0.5'], np.array([0, 0, 0, 0, 1])))
        self.assertTrue(np.array_equal(
            result['sentiment,-0.5,0'], np.array([1, 1, 1, 0, 0])))
        self.assertTrue(np.array_equal(
            result['sentiment,0,0.5'], np.array([0, 0, 0, 1, 0])))

    def test_generateMaskedMatrixParties(self):
        # Setup for party classification
        # tarList = [
        #     {'prob': 0.24, 'description': 'only mention',
        #         'contain': ['Republican']},
        #     {'prob': 0.25, 'description': 'only mention',
        #         'contain': ['Democratic']},
        #     {'prob': 0.30, 'description': 'only mention',
        #         'contain': ['Republican', 'Democratic']},
        #     {'prob': 0.11, 'description': 'minority but can also mention',
        #         'contain': ['Republican', 'Democratic']},
        #     {'prob': 0.10, 'description': 'No parties', 'contain': []}
        # ]

        items_per_category = {
            'party,only mention:Republican': 2,
            'party,only mention:Democratic': 2,
            # 'party,only mention:Republican,Democratic': 1,
            "party,composition:[['Republican'],['Democratic']]": 1,
            'party,minority but can also mention:Republican,Democratic': 1,
            'party,No parties:': 1
        }

        itemPool = np.asarray([0, 5, 2, 4])  # IDs from the article_df index
        # Create an array [0, 1, 2, 3, 4] for indexing in the mask
        newIndex = np.arange(len(itemPool))
        # Map from index in masked matrix to item IDs
        newId_to_cornacId = dict(enumerate(itemPool))
        # Map from item IDs to masked matrix indices
        cornacId_to_newId = dict(zip(itemPool, newIndex))

        # Generate the masked matrix for party classification
        result = self.rdw_sampler.generateMaskedMatrixParties(
            self.article_df, itemPool, 'entities', items_per_category, cornacId_to_newId
        )

        # Expected results:
        # 'only mention:Republican' - Should match items 1  (Republican only)
        # 'only mention:Democratic' - Should match items 2 (Democratic only)
        # 'only mention:Republican,Democratic' - Should match items 0, 5,6
        # 'minority but can also mention:Republican,Democratic' - Should match item 3 (Independent)
        # 'No parties' - Should match item 4 (No parties)

        print(f"result:{result}")
        self.assertTrue(np.array_equal(
            result['party,only mention:Republican'], np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(
            result['party,only mention:Democratic'], np.array([0, 0, 1, 0])))
        # self.assertTrue(np.array_equal(
        #     result['party,only mention:Republican,Democratic'], np.array([1, 1, 0, 0])))
        self.assertTrue(np.array_equal(
            result["party,composition:[['Republican'],['Democratic']]"], np.array([1, 1, 0, 0])))
        self.assertTrue(np.array_equal(
            result['party,minority but can also mention:Republican,Democratic'], np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(
            result['party,No parties:'], np.array([0, 0, 0, 1])))

    def test_prepareLinearProgramming(self):
        # Setup test data
        # Adding an 'age' column for continuous testing
        self.article_df['age'] = [5, 15, 25, 35, 45, 55, 65]
        itemPool = np.asarray([0, 1, 2, 4, 6])  # Example item pool

        # Target distributions:
        targetDistributions = [
            {"type": "discrete", "distr": {
                "Politics": 0.4, "Sports": 0.3, "Technology": 0.3}},
            {"type": "continuous", "distr": [
                {'prob': 0.5, 'min': 0, 'max': 30}, {'prob': 0.5, 'min': 30, 'max': 60}]},
            {"type": "party", "distr": [
                {'prob': 0.24, 'description': 'only mention',
                    'contain': ['Republican']},
                {'prob': 0.25, 'description': 'only mention',
                    'contain': ['Democratic']},
                # {'prob': 0.30, 'description': 'only mention',
                #     'contain': ['Republican', 'Democratic']},
                  {'prob': 0.30, 'description': 'composition',
                    'contain': [['Republican'], ['Democratic']]},
                {'prob': 0.11, 'description': 'minority but can also mention',
                    'contain': ['Republican', 'Democratic']},
                {'prob': 0.10, 'description': 'No parties', 'contain': []}
            ]}
        ]

        # Target dimensions: corresponding to category, age, and entities
        targetDimensions = ['category', 'age', 'entities']

        # Define the target size
        targetSize = 10

        # Call the prepareLinearProgramming function
        super_dict_matrix, super_dict_number, newId_to_cornacId, cornacId_to_newId = self.rdw_sampler.prepareLinearProgramming(
            self.article_df, itemPool, targetDimensions, targetDistributions, targetSize)

        # Verify that the mappings were created correctly
        expected_cornacId_to_newId = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4}
        self.assertEqual(cornacId_to_newId, expected_cornacId_to_newId)

        expected_newId_to_cornacId = {0: 0, 1: 1, 2: 2, 3: 4, 4: 6}
        self.assertEqual(newId_to_cornacId, expected_newId_to_cornacId)

        # Verify the discrete attribute matrix (category)
        self.assertTrue(np.array_equal(
            super_dict_matrix['category,Politics'], np.array([1, 0, 0, 0, 1])))
        self.assertTrue(np.array_equal(
            super_dict_matrix['category,Sports'], np.array([0, 1, 0, 0, 0])))
        self.assertTrue(np.array_equal(
            super_dict_matrix['category,Technology'], np.array([0, 0, 1, 1, 0])))

        # Verify the continuous attribute matrix (age)
        self.assertTrue(np.array_equal(
            super_dict_matrix['age,0,30'], np.array([1, 1, 1, 0, 0])))
        self.assertTrue(np.array_equal(
            super_dict_matrix['age,30,60'], np.array([0, 0, 0, 1, 0])))

        # Verify the party classification matrix (entities)
        self.assertTrue(np.array_equal(
            super_dict_matrix['entities,only mention:Republican'], np.array([0, 1, 0, 0, 0])))
        self.assertTrue(np.array_equal(
            super_dict_matrix['entities,only mention:Democratic'], np.array([0, 0, 1, 0, 0])))
        # self.assertTrue(np.array_equal(
        #     super_dict_matrix['entities,only mention:Republican,Democratic'], np.array([1, 0, 0, 0, 1])))
        self.assertTrue(np.array_equal(
            super_dict_matrix["entities,composition:[['Republican'], ['Democratic']]"], np.array([1, 0, 0, 0, 1])))
        self.assertTrue(np.array_equal(
            super_dict_matrix['entities,minority but can also mention:Republican,Democratic'], np.array([0, 0, 0, 0, 0])))
        self.assertTrue(np.array_equal(
            super_dict_matrix['entities,No parties:'], np.array([0, 0, 0, 1, 0])))

        # Verify the target number of items per category for each target distribution
        expected_super_dict_number = {
            'category,Politics': 4, 'category,Sports': 3, 'category,Technology': 3,
            'age,0,30': 5, 'age,30,60': 5,
            'entities,only mention:Republican': 2, 'entities,only mention:Democratic': 3, "entities,composition:[['Republican'], ['Democratic']]": 3,
            'entities,minority but can also mention:Republican,Democratic': 1, 'entities,No parties:': 1
        }
        self.assertDictEqual(super_dict_number, expected_super_dict_number)


if __name__ == '__main__':
    unittest.main(buffer=False)
