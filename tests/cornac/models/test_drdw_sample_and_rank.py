import unittest
import numpy as np
import pandas as pd
from cornac.models.drdw.sample_and_rank import Sample_And_Rank
from cornac.models.drdw.graph_recommender import GraphRec
from cornac.models.drdw.sample_core import DistributionSampler
from pandas.testing import assert_frame_equal
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal
from unittest.mock import patch


class TestSampleAndRank(unittest.TestCase):
    def setUp(self):
        # Create a 3x5 user-item interaction matrix
        # Example matrix:
        self.train_set_rating = np.array([
            [1, 0, 0, 1, 0],  # User 0
            [0, 1, 0, 1, 1],  # User 1
            [1, 1, 1, 0, 0],  # User 2
            [0, 0, 0, 0, 0],  # User 3
        ])

        self.articlesDataframe = pd.DataFrame({
            'category': ['Politics', 'Sports', 'Technology', 'Politics', 'Sports'],
            'sentiment': [-0.5, 0.1, 0.8, -0.7, 0.3],
            'outlet': ['CNN', 'BBC', 'CNN', 'NBC', 'FOX'],
            'entities': [
                "['Republican', 'Democratic']",  # Both parties
                "['Republican']",  # Only Republican
                "['Democratic']",  # Only Democratic
                "['Independent','Democratic']",  # Minority
                None],  # No party,
            'age': [5, 15, 35,
                        11, 55]
        }, index=[0, 1, 2, 3, 4])

        self.sampler_ranker = Sample_And_Rank(
            self.train_set_rating, self.articlesDataframe)

    def test_init(self):
        sample_rank_instance = Sample_And_Rank(
            self.train_set_rating, self.articlesDataframe)

        self.assertTrue(np.array_equal(
            sample_rank_instance.train_set_rating, self.train_set_rating))
        self.assertTrue(sample_rank_instance.articlesDataframe.equals(
            self.articlesDataframe))

        # Total number of articles
        self.assertEqual(sample_rank_instance.articleNum,
                         self.train_set_rating.shape[1])
        # Empty item pool after initialization
        self.assertEqual(np.size(sample_rank_instance.itemPool), 0)
        self.assertEqual(np.size(sample_rank_instance.articleRdwScore), 0)
        # Empty list for old candidates
        self.assertEqual(sample_rank_instance.CANDIDATESOLD,
                         [])

    def test_sampleArticles(self):
        # Test sampling of articles based on target dimensions and distributions
        targetDimensions = ['category']
        targetDistributions = [{
            'type': 'discrete',
            'distr': {"Politics": 0.4, "Sports": 0.3, "Technology": 0.3}
        }]
        targetSize = 3
        linear_program_coefficient = 'rdw_score'
        # here will use the random walk score (not a feature from the article dataframe)

        self.sampler_ranker.itemPool = np.asarray([0, 1, 4, 3, 2])
        self.sampler_ranker.articleRdwScore = np.array(
            [0.5, 0.7, 0.3, 0.4, 0.6])
        # Sample articles
        target_num_items, sampled_Items = self.sampler_ranker.sampleArticles(
            targetDimensions, targetDistributions, targetSize, linear_program_coefficient)
        # Check if the sampled number of items corresponds to the target size
        self.assertEqual(len(sampled_Items), targetSize)
        # maximize rdw_score
        expected_sampled_items = [0, 1, 2]
        self.assertEqual(sampled_Items, expected_sampled_items)

        linear_program_coefficient = 'sentiment'
        # Sample articles
        target_num_items, sampled_Items = self.sampler_ranker.sampleArticles(
            targetDimensions, targetDistributions, targetSize, linear_program_coefficient)
        self.assertEqual(len(sampled_Items), targetSize)
        # minimize sentiment
        expected_sampled_items = [1, 2, 3]
        self.assertTrue(set(sampled_Items) == set(expected_sampled_items))

    def test_sampleArticles_multiple_dim(self):
        # Define target dimensions and distributions
        targetDimensions = ['category', 'sentiment']
        targetDistributions = [
            {'type': 'discrete', 'distr': {
                "Politics": 0.4, "Sports": 0.3, "Technology": 0.3}},
            {'type': 'continuous', 'distr': [
                {'min': -0.5, 'max': 0, 'prob': 0.33},  {'min': 0, 'max': 0.5, 'prob': 0.33}, {'min': 0.5, 'max': 1, 'prob': 0.34}]}
        ]
        targetSize = 3  # Number of articles to be sampled
        self.sampler_ranker.itemPool = np.asarray([1, 4, 0, 3, 2])

        # Run sampleArticles method
        target_num_items, candidateItems = self.sampler_ranker.sampleArticles(
            targetDimensions=targetDimensions,
            targetDistributions=targetDistributions,
            targetSize=targetSize,
            linear_program_coefficient='sentiment'  # minimize sentiment
        )

        # Assertions
        # Check if the sampled items meet the expected size
        self.assertEqual(len(candidateItems), targetSize)

        # Check if the sampled categories and sentiment values match the target distributions
        # Example: Check if category "Politics" is in the candidate items
        sampled_categories = self.articlesDataframe.loc[candidateItems, 'category'].tolist(
        )
        self.assertIn("Politics", sampled_categories)

        expected_sampled_items = [0, 2, 1]
        self.assertTrue(set(candidateItems) == set(expected_sampled_items))

        # Verify the target number of items per category
        expected_target_num_items = {
            "category,Politics": 1, "category,Sports": 1, "category,Technology": 1}
        self.assertEqual(
            target_num_items['category,Politics'], expected_target_num_items["category,Politics"])

    def test_sampleArticles_multiple_dim_entities_outlet(self):
        # Define target dimensions and distributions
        targetDimensions = ['entities', 'outlet']
        targetDistributions = [
            {"type": "parties",
             "distr": [{"description": "only mention", "contain": ["Republican"], "prob": 0.2},
                       {"description": "only mention",
                       "contain": ["Democratic"], "prob": 0.2},
                    #    {"description": "only mention", "contain": [
                    #        "Republican", "Democratic"], "prob": 0.2},
                         {"description": "composition", "contain": [
                          ["Republican"], ["Democratic"]], "prob": 0.2},
                       {"description": "minority but can also mention", "contain": [
                           "Republican", "Democratic"], "prob": 0.2},  # item 2 and item 6
                       {"description": "no parties", "contain": [], "prob": 0.2}]},
            {'type': 'discrete', 'distr': {
                "CNN": 0.4, "BBC": 0.3, "FOX": 0.3}},
        ]
        targetSize = 3  # Number of articles to be sampled
        self.sampler_ranker.itemPool = np.asarray([3, 4, 0, 1, 2])

        # Run sampleArticles method
        target_num_items, candidateItems = self.sampler_ranker.sampleArticles(
            targetDimensions=targetDimensions,
            targetDistributions=targetDistributions,
            targetSize=targetSize,
            linear_program_coefficient='sentiment'  # minimize sentiment
        )

        # Assertions
        # Check if the sampled items meet the expected size
        self.assertEqual(len(candidateItems), 0)

        # change target distribution dict:
        targetDistributions = [
            {"type": "parties",
             "distr": [{"description": "only mention", "contain": ["Republican"], "prob": 0.2},
         
                       {"description": "no parties", "contain": [], "prob": 0.2},
                       {"description": "composition", "contain": [
                           ["Republican"], ["Democratic"]], "prob": 0.2},
                    {"description": "only mention",
                       "contain": ["Democratic"], "prob": 0.2},
                       {"description": "minority but can also mention", "contain": [
                           "Republican", "Democratic"], "prob": 0.2}  # item 2 and item 6
                       ]},
            {'type': 'discrete', 'distr': {
                "CNN": 0.4, "BBC": 0.3, "FOX": 0.3}},
            #    {'type': 'discrete', 'distr': {
            #     "CNN": 0.4, "BBC": 0.2, "FOX": 0.2, "NBC":0.2}},
        ]
        targetSize = 3  # Number of articles to be sampled
        self.sampler_ranker.itemPool = np.asarray([3, 4, 0, 1, 2])

        # Run sampleArticles method
        target_num_items, candidateItems = self.sampler_ranker.sampleArticles(
            targetDimensions=targetDimensions,
            targetDistributions=targetDistributions,
            targetSize=targetSize,
            linear_program_coefficient='sentiment'  # minimize sentiment
        )

        # Assertions
        # Check if the sampled items meet the expected size
        self.assertEqual(len(candidateItems), targetSize)

        # Check if the sampled categories and sentiment values match the target distributions
        # Example: Check if category "Politics" is in the candidate items
        sampled_outlet = self.articlesDataframe.loc[candidateItems, 'outlet'].tolist(
        )
        self.assertIn("CNN", sampled_outlet)
        self.assertIn("BBC", sampled_outlet)
        self.assertIn("FOX", sampled_outlet)

        # expected_sampled_items = [0, 1, 2, 3, 4]
        expected_sampled_items = [0, 1, 4]
        self.assertTrue(set(candidateItems) == set(expected_sampled_items))

        # Verify the target number of items per category
        expected_target_num_items = {
            "outlet,CNN": 1, "outlet,BBC": 1,  "outlet,FOX": 1}
        self.assertEqual(
            target_num_items["outlet,CNN"], expected_target_num_items["outlet,CNN"])

    def test_rankArticles_by_rdw_score(self):

        candidateItems = np.asarray([0, 1, 2, 3, 4])
        rankingType = 'rdw_score'
        self.sampler_ranker.articleRdwScore = np.array(
            [0.5, 0.7, 0.3, 0.4, 0.6])

        # Rank articles based on RDW score
        rankedArticles, scores = self.sampler_ranker.rankArticles(
            candidateItems, 3, rankingType)  # target size =3

        # The RDW scores in the articlesDataframe: [0.5, 0.7, 0.3, 0.4, 0.6]
        # Articles sorted by RDW score (descending)
        expected_ranked = np.array([1, 4, 0])
        expected_scores = np.array([0.7, 0.6, 0.5])

        assert_array_equal(rankedArticles, expected_ranked)
        assert_array_equal(scores, expected_scores)

    def test_rankArticles_graph_coloring(self):
        # Test rankArticles with graph_coloring
        candidateItems = [0, 1, 2, 3, 4]  # Sample candidate items
        rankingType = 'graph_coloring'
        rankingObjectives = 'category'
        self.sampler_ranker.articleRdwScore = np.array(
            [0.5, 0.7, 0.3, 0.4, 0.6])

        # Rank articles based on graph coloring
        rankedArticles, scores = self.sampler_ranker.rankArticles(
            candidateItems, 3, rankingType, rankingObjectives)

        self.assertEqual(len(rankedArticles), 3)
        expected_rankedArticles = np.array([0, 1, 3])
        expected_scores = np.array([0.5, 0.7, 0.4])
        assert_array_equal(rankedArticles, expected_rankedArticles)
        assert_array_equal(scores, expected_scores)

    def test_rankArticles_multi_objectives(self):
        candidateItems = np.array([0, 1, 2, 3, 4])

        # Define the ranking type and objectives
        rankingType = "multi_objectives"
        # Rank by sentiment first, then category
        rankingObjectives = ['sentiment', 'category']
        targetSize = 3
        ascending = [False, True]  # Sentiment descending, Category ascending
        mappingList = [{},
                       {'category': {'Politics': 1, 'Sports': 2, 'Technology': 3}}]

        # Mock articleRdwScore
        self.sampler_ranker.articleRdwScore = np.array(
            [0.5, 0.7, 0.3, 0.4, 0.6])

        # Run the rankArticles method
        rankedArticles, scores = self.sampler_ranker.rankArticles(
            candidateItems=candidateItems,
            targetSize=targetSize,
            rankingType=rankingType,
            rankingObjectives=rankingObjectives,
            mappingList=mappingList,
            ascending=ascending
        )

        # Expected ranking based on sentiment (descending) and category (ascending)
        # Sentiment values: [-0.5, 0.1, 0.8, -0.7, 0.3],
        # Categories: ['Politics', 'Sports', 'Technology', 'Politics', 'Sports'],
        # Articles sorted by sentiment and category: [1, 4, 0]
        expected_ranked = [2, 4, 1]
        expected_scores = [0.3, 0.6, 0.7]

        # Assert that the articles are ranked correctly
        self.assertEqual(rankedArticles, expected_ranked)
        # Assert that the scores match the expected RDW scores
        np.testing.assert_array_equal(scores, expected_scores)

    def test_rankArticles_multi_objectives_mappinglist(self):
        candidateItems = np.array([0, 1, 2, 3, 4])

        # Define the ranking type and objectives
        rankingType = "multi_objectives"
        # Rank by outlet first, then category
        rankingObjectives = ['outlet', 'category']
        targetSize = 3
        ascending = [True, True]  # outlet, Category ascending
        # 'category': ['Politics', 'Sports', 'Technology', 'Politics', 'Sports'],
        # 'outlet': ['CNN', 'BBC', 'CNN', 'NBC', 'FOX'],
        # mappingList records the preferences (priority)
        mappingList = [{'outlet': {'CNN': 1, 'BBC': 2, 'NBC': 3, 'FOX': 4}},
                       {'category': {'Politics': 1, 'Sports': 2, 'Technology': 3}}]

        self.sampler_ranker.articleRdwScore = np.array(
            [0.5, 0.7, 0.3, 0.4, 0.6])

        # Run the rankArticles method
        rankedArticles, scores = self.sampler_ranker.rankArticles(
            candidateItems=candidateItems,
            targetSize=targetSize,
            rankingType=rankingType,
            rankingObjectives=rankingObjectives,
            mappingList=mappingList,
            ascending=ascending
        )

        # Articles sorted by outlet and category
        expected_ranked = [0, 2, 1]
        expected_scores = [0.5, 0.3, 0.7]

        # Assert that the articles are ranked correctly
        self.assertEqual(rankedArticles, expected_ranked)
        # Assert that the scores match the expected RDW scores
        np.testing.assert_array_equal(scores, expected_scores)

    def test_filterHeuristics(self):
        # Test the filterHeuristics method
        filteringCriteria = {
            'filterDim': 'age',
            'filterThreshold': 25,
            'comparison': 'less'
        }
        user_idx = 0
        itemPool = [0, 1, 2, 3, 4]

        filtered_items = self.sampler_ranker.filterHeuristics(
            user_idx=user_idx, itemPool=itemPool, filteringCriteria=filteringCriteria)

        # Expecting only items with article Age < 25, and not in user 0's history
        self.assertEqual(filtered_items.tolist(), [1])

        # Test the filterHeuristics method when filteringCriteria is None
        filteringCriteria = None
        user_idx = 0
        itemPool = [0, 1, 2, 3, 4]

        filtered_items = self.sampler_ranker.filterHeuristics(
            user_idx=user_idx, itemPool=itemPool, filteringCriteria=filteringCriteria)

        # Expecting only items with article not in user 0's history
        self.assertEqual(filtered_items.tolist(), [1, 2, 4])

    @patch('random.sample')
    @patch('random.randint')
    @patch.object(Sample_And_Rank, 'sampleArticles')
    def test_newHop_empty_history(self, mock_sample_articles, mock_randint, mock_random_sample):
        # Test for the case where the user has no interaction history (isEmptyHistory=True)
        user_id = 3  # User 3 has no history in train_set_rating
        targetDimensions = ['category']
        targetDistributions = [{'type': 'discrete', 'distr': {
            "Politics": 0.4, "Sports": 0.3, "Technology": 0.3}}]
        targetSize = 3
        sampleObjective = 'rdw_score'
        currentHop = 1
        filteringCriteria = None

        # Mock random calls for user with no interaction history
        mock_randint.return_value = 15  # Simulating 10-20 times targetSize
        # Mock random sampling from articles
        mock_random_sample.return_value = [0, 1, 2, 3, 4]

        # Mock the output of sampleArticles
        mock_sample_articles.return_value = (
            {'category,Politics': 1}, [0, 1, 2])

        # Call the method
        candidateItems = self.sampler_ranker.newHop(
            user_id, targetDimensions, targetDistributions, targetSize, sampleObjective, currentHop, filteringCriteria)

        # Check if the random sample was called with the correct parameters
        mock_randint.assert_called_once_with(10, 20)
        mock_random_sample.assert_called_once_with(
            range(0, self.sampler_ranker.articleNum), 5)

        # Check if sampleArticles was called with the correct arguments
        mock_sample_articles.assert_called_once_with(
            targetDimensions, targetDistributions, targetSize, sampleObjective)

        # Assert that candidateItems are correct
        self.assertEqual(candidateItems, [0, 1, 2])

    @patch.object(GraphRec, 'performMultiHop')
    def test_newHop_with_history(self, mock_performMultiHop):
        # Test for the case where the user has interaction history (isEmptyHistory=False)

        user_id = 0  # User 0 has history item 0 and item 3 in train_set_rating
        targetDimensions = ['category']
        targetDistributions = [{'type': 'discrete', 'distr': {
            "Politics": 0.4, "Sports": 0.3, "Technology": 0.3}}]
        targetSize = 3
        sampleObjective = 'rdw_score'
        currentHop = 3
        filteringCriteria = {
            'filterDim': 'sentiment',
            'filterThreshold': 0,
            'comparison': 'larger'
        }

        fake_probs = np.array([
            # User 0: Item 0, 1 ,2, 3 and 4
            [0, 0, 0, 0, 0.1, 0.4, 0.2, 0.1, 0.2],
            [0, 0, 0, 0, 0.0, 0.4, 0.0, 0.3, 0.3],  # User 1: Items 1, 3, and 4
            # User 2 prefers Items 0, 1, and 2
            [0, 0, 0, 0, 0.3, 0.3, 0.4, 0.0, 0.0],
            # User 3 no interactions (empty)
            [0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        # Mock the result of Model_RDW.performMultiHop()
        mock_performMultiHop.return_value = csr_matrix(fake_probs)

        # Call the method
        sampledItems = self.sampler_ranker.newHop(
            user_id, targetDimensions, targetDistributions, targetSize, sampleObjective, currentHop, filteringCriteria)

        # Check if performMultiHop was called
        mock_performMultiHop.assert_called_once_with(currentHop)

        # User 0: history item 0 and 3.
        self.assertEqual(list(self.sampler_ranker.itemPool), [1, 2, 4])
        # Assert that candidateItems are correct
        self.assertEqual(sampledItems, [])

        targetDistributions = [{'type': 'discrete', 'distr': {
            "Sports": 0.6, "Technology": 0.4}}]
        # Call the method
        sampledItems = self.sampler_ranker.newHop(
            user_id, targetDimensions, targetDistributions, targetSize, sampleObjective, currentHop, filteringCriteria)
        self.assertEqual(sampledItems, [1, 2, 4])

    def test_performSampling(self):
        user_id = 1
        listSize = 2
        targetDimensions = ['category']
        targetDistribution = [{'type': 'discrete', 'distr': {
            "Politics": 0.4,  "Technology": 0.35, "Sports": 0.25}}]
        maxHops = 3  # maxHops must be large or larger equal than 3.
        filteringCriteria = None
        sampleObjective = 'rdw_score'
        rankingType = 'multi_objectives'
        rankingObjectives = ['sentiment', 'category']
        mappingList = None
        # first look at sentiment (descending), when tie look at category
        ascending = [False, True]

        # Call performSampling
        ranked_items, scores = self.sampler_ranker.performSampling(
            user_id=user_id,
            listSize=listSize,
            targetDimensions=targetDimensions,
            targetDistribution=targetDistribution,
            maxHops=maxHops,
            filteringCriteria=filteringCriteria,
            sampleObjective=sampleObjective,
            rankingType=rankingType,
            rankingObjectives=rankingObjectives,
            mappingList=mappingList,
            ascending=ascending
        )

        # Verify the results
        expected_ranked_items = [2,  0]

        self.assertEqual(ranked_items, expected_ranked_items)

        listSize = 3
        ranked_items, scores = self.sampler_ranker.performSampling(
            user_id=user_id,
            listSize=listSize,
            targetDimensions=targetDimensions,
            targetDistribution=targetDistribution,
            maxHops=maxHops,
            filteringCriteria=filteringCriteria,
            sampleObjective=sampleObjective,
            rankingType=rankingType,
            rankingObjectives=rankingObjectives,
            mappingList=mappingList,
            ascending=ascending
        )

        self.assertEqual(len(ranked_items), listSize)
        expected_containing_items = [0, 2]
        self.assertTrue(
            all(item in ranked_items for item in expected_containing_items))

    @patch.object(Sample_And_Rank, 'sampleArticles')
    def test_addRandomArticles(self, mock_sample_articles):
        # Mock return values for sampleArticles to simulate incomplete sampling
        mock_sample_articles.side_effect = [
            ({}, [0, 1, 2]),  # First call returns 3 items
            ({}, [0, 1]),    # Second call returns 2 items
            ({}, [0]),       # Third call returns 1 item
            ({}, []),        # Fourth call returns no items
        ]

        targetDimensions = ['category']
        targetDistributions = [{'type': 'discrete',
                                'distr': {"Politics": 0.5, "Sports": 0.5}}]
        targetSize = 4
        sampleObjective = 'sentiment'  # need a continous value for minimization

        # mock self.sampler_ranker.articleNum
        self.sampler_ranker.articleNum = 4
        result = self.sampler_ranker.addRandomArticles(
            targetDimensions, targetDistributions, targetSize, sampleObjective)

        # Check that the final result contains 4 items (3 sampled + 1 random)
        self.assertEqual(len(result), 4)
        # Verify the exact contents of the result: 0, 1, 2 from sample, and 3 added randomly
        expected_result = [0, 1, 2, 3]
        self.assertEqual(result, expected_result)

        # Verify that sampleArticles was called one times
        self.assertEqual(mock_sample_articles.call_count, 1)

    @patch.object(Sample_And_Rank, 'sampleArticles')
    def test_addRandomArticles_empty_sample(self, mock_sample_articles):
        # Mock return values for sampleArticles to simulate incomplete sampling
        mock_sample_articles.side_effect = [
            ({}, []),        # First call returns no items
            ({}, []),        # Second call returns no items
            ({}, []),        # Thrid call returns no items
            ({}, []),        # Fourth call returns no items
        ]

        targetDimensions = ['category']
        targetDistributions = [{'type': 'discrete',
                                'distr': {"Politics": 0.5, "Sports": 0.5}}]
        targetSize = 4
        sampleObjective = 'sentiment'  # need a continous value for minimization
        # mock  self.sampler_ranker.articleNum
        self.sampler_ranker.articleNum = 4
        result = self.sampler_ranker.addRandomArticles(
            targetDimensions, targetDistributions, targetSize, sampleObjective)

        # Check that the final result contains 4 items
        self.assertEqual(len(result), 4)
        expected_result = [0, 1, 2, 3]
        self.assertCountEqual(result, expected_result)

        # Verify that sampleArticles was called three times
        self.assertEqual(mock_sample_articles.call_count, 3)

    def test_checkListParity(self):
        # Test case where the lists are identical
        candidatesOld = [1, 2, 3, 4, 5]
        candidatesNew = [5, 4, 3, 2, 1]
        result = self.sampler_ranker.checkListParity(
            candidatesOld, candidatesNew)
        self.assertTrue(result)

        # Test case where the lists are different
        candidatesOld = [1, 2, 3]
        candidatesNew = [4, 5, 6]
        result = self.sampler_ranker.checkListParity(
            candidatesOld, candidatesNew)
        self.assertFalse(result)

        # Test case where lists have some overlap but are not identical
        candidatesOld = [1, 2, 3]
        candidatesNew = [3, 4, 5]
        result = self.sampler_ranker.checkListParity(
            candidatesOld, candidatesNew)
        self.assertFalse(result)

        # Test case where both lists are empty
        candidatesOld = []
        candidatesNew = []
        result = self.sampler_ranker.checkListParity(
            candidatesOld, candidatesNew)
        self.assertTrue(result)

        # Test case where one list is empty, and the other is not
        candidatesOld = [1, 2, 3]
        candidatesNew = []
        result = self.sampler_ranker.checkListParity(
            candidatesOld, candidatesNew)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
