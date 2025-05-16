import unittest
import numpy as np
from cornac.models import D_RDW
from cornac.eval_methods import BaseMethod
from cornac.metrics import Activation
from cornac.metrics import Calibration
from cornac.metrics import Representation
from cornac.metrics import AlternativeVoices
import pandas as pd
import json
from cornac.experiment.experiment import Experiment
from cornac.models.drdw.rank_core import ItemRanker
from cornac.models.drdw.sample_and_rank import Sample_And_Rank


class TestModels(unittest.TestCase):
    def setUp(self):
        self.data = [('user1', 'item1', 1.0),
                     ('user2', 'item2', 1.0),
                     ('user1', 'item3', 1.0),
                     ('user2', 'item3', 1.0),
                     ('user2', 'item4', 1.0),
                     ('user3', 'item3', 1.0),
                     ('user3', 'item5', 1.0),
                     ('user3', 'item6', 1.0),
                     ('user3', 'item7', 1.0),
                     ('user4', 'item2', 1.0),
                     ('user4', 'item8', 1.0),
                     ('user4', 'item9', 1.0),
                     ('user5', 'item8', 1.0),
                     ('user5', 'item10', 1.0),
                     ('user5', 'item11', 1.0),
                     ('user4', 'item12', 1.0),
                     ('user1', 'item5', 1.0),
                     ('user4', 'item6', 1.0),
                     ('user5', 'item9', 1.0)]
        self.bm = BaseMethod.from_splits(
            train_data=self.data[:-3], test_data=self.data[-3:],  exclude_unknowns=True)
        # Item_feature = {0: np.array([1, 2, 0]), 1: np.array([1, 2, 3]), 2: np.array([4, 2, 9]), 3: np.array([4, 2, 5]),
        #                 4: np.array([4, 3, 5]), 5: np.array([0, 2, 3])}
        self.Item_category = {0: "music", 1: "sports", 2: "entertainment", 3: "politics",
                              4: "sports", 5: "politics", 6: "politics", 7: "politics", 8: "music", 9: "politics", 10: "entertainment", 11: "sports"}
        self.Item_sentiment = {0: 0.2, 1: -0.2, 2: 0.6, 3: 0.8,
                               4: -0.3, 5: -0.7, 6: -0.8, 7: 0.1, 8: 0.4, 9: -0.4, 10: -0.6, 11: 0.7}
        # with missing value
        self.Item_entities = {
            3: ["AnyParty", "Democratic", "Democratic", "Democratic"], 4: ["Democratic", "Democratic", "Democratic"],
            5: ["Party1", "Republican", "Party1"], 6: ["Republican", "Democratic"], 7: ["Republican", "Democratic"], 9: ["Republican", "Republican"]}
        self.Item_min = {0:  0.1,  1: 0.2, 2: 0,
                         3: 0.5, 4: 0.25, 5: 0.4, 6: 0.7, 7: 0.8, 8: 0.6, 9: 0.3, 10: 0.6, 11: 0.4}

        self.Item_min_major = {0:  np.array([0.1,  0.9]), 1: np.array([0.2, 0.8]), 2: np.array([0, 1]),
                               3: np.array([0.5, 0.5]), 4: np.array([0.25, 0.75]), 5: np.array([0.4, 0.6]),
                               6:  np.array([0.7,  0.3]), 7: np.array([0.8, 0.2]), 8: np.array([0.6, 0.4]),
                               9: np.array([0.3, 0.7]), 10: np.array([0.6, 0.4]), 11: np.array([0.4, 0.6])}
        self.Item_stories = {0: 1, 1: 3, 2: 18, 3: 2, 4: 22,
                             5: 15, 6: 29, 7: 20, 8: 14, 9: 36, 10: 32, 11: 26}
        self.Item_popularity = {0: 1, 1: 2, 2: 3, 3: 1, 4: 1,
                                5: 1, 6: 1, 7: 2, 8: 1, 9: 1, 10: 1, 11: 1}

        self.Item_age = {0: 1, 1: 3, 2: 18, 3: 2, 4: 22,
                         5: 15, 6: 29, 7: 20, 8: 14, 9: 36, 10: 32, 11: 26}
        # prepare dataframe
        out_pd = (pd.Series(self.Item_category).to_frame('category')
                  .join(pd.Series(self.Item_min).to_frame('minority_proportion'), how='outer'))
        out_pd = out_pd.join(
            pd.Series(self.Item_entities).to_frame('entities'), how='outer')
        out_pd = out_pd.join(
            pd.Series(self.Item_sentiment).to_frame('sentiment'), how='outer')
        out_pd = out_pd.join(
            pd.Series(self.Item_popularity).to_frame('popularity'), how='outer')
        out_pd = out_pd.join(
            pd.Series(self.Item_stories).to_frame('story'), how='outer')
        out_pd = out_pd.join(
            pd.Series(self.Item_age).to_frame('articleAge'), how='outer')
        self.article_dataframe = out_pd
        self.config_file =  "./tests/cornac/models/test_drdw_parameter.ini"
    
    def test_init(self):
        """
        Test initialization with all parameters provided.
        """
        model = D_RDW(
            item_dataframe=self.article_dataframe,
            diversity_dimension=["category"],
            target_distributions={"category": {"type": "discrete", "distr": {"weather": 0.8, "news": 0.2}}},
            targetSize=10,
            maxHops=15,
            sampleObjective="rdw_score",
            rankingType="graph_coloring",
            rankingObjectives=["category"]
        )

        self.assertEqual(model.targetSize, 10)
        self.assertEqual(model.maxHops, 15)
        self.assertEqual(model.diversity_dimension, ["category"])
        self.assertIsNotNone(model.targetDistribution)
        self.assertEqual(model.sampleObjective, "rdw_score")
    
    def test_config_file_loading(self):
        """
        Test loading configurations from a config file.
        """
        model = D_RDW(
            item_dataframe=self.article_dataframe,
            config_file=self.config_file
        )
        self.assertEqual(model.targetSize, 5)
        self.assertEqual(model.maxHops, 6)
        self.assertEqual(model.diversity_dimension, ["category", "sentiment"])
        self.assertEqual(model.targetDistribution, {
    "category": {
        "type": "discrete",
        "distr": {
            "entertainment": 0.2,
            "sports": 0.2,
            "politics": 0.4,
            "music": 0.2
        }
    },
    "sentiment": {
        "type": "continuous",
        "distr": [
            {"min": -1, "max": -0.5, "prob": 0.2},
            {"min": -0.5, "max": 0, "prob": 0.3},
            {"min": 0, "max": 0.5, "prob": 0.3},
            {"min": 0.5, "max": 1, "prob": 0.2}
        ]
    },
    "entities": {
        "type": "parties",
        "distr": [
            {"description": "only mention", "contain": ["Republican Party"], "prob": 0.15},
            {"description": "only mention", "contain": ["Democratic Party"], "prob": 0.15},
            # {"description": "only mention", "contain": ["Republican Party", "Democratic Party"], "prob": 0.15},
            {"description": "composition", "contain": [["Republican Party"], ["Democratic Party"]], "prob": 0.15},
            {"description": "minority but can also mention", "contain": ["Republican Party", "Democratic Party"], "prob": 0.25},
            {"description": "no parties", "contain": [], "prob": 0.3}
        ]
    }
})

        self.assertEqual(model.sampleObjective, "rdw_score")
        self.assertEqual(model.rankingType, "graph_coloring")
        self.assertEqual(model.rankingObjectives, ["category"])
    def test_with_dummy_data(self):
        targetSize = 4
        cal = Calibration(item_feature=self.Item_category,
                          data_type="category", k=targetSize)
        act = Activation(item_sentiment=self.Item_sentiment, k=targetSize)

        repre = Representation(item_entities=self.Item_entities, k=targetSize)
        alt = AlternativeVoices(
            item_minor_major=self.Item_min_major, k=targetSize)
        # prepare model
        # with open('./experiments/configs/reranker_configs/target_distr_dummy.json') as f:
        #     target_distr = json.load(f)
        target_distr = {
  "category": {
    "type": "discrete",
    "distr": {
      "politics": 0.5,
      "music": 0.25,
      "sports": 0.25
    }
  },
  "sentiment": {
    "type": "continuous",
    "distr": [
      {
        "min": -1,
        "max": -0.5,
        "prob": 0.25
      },
      {
        "min": -0.5,
        "max": 0,
        "prob": 0.25
      },
      {
        "min": 0,
        "max": 0.5,
        "prob": 0.25
      },
      {
        "min": 0.5,
        "max": 1,
        "prob": 0.25
      }
    ]
  },
  "entities": {
    "type": "parties",
    "distr": [
      {
        "description": "only mention",
        "contain": ["Republican Party", "Democratic Party"],
        "prob": 0.5
      },
      {
        "description": "minority but can also mention",
        "contain": ["Republican Party", "Democratic Party"],
        "prob": 0.5
      }
    ]
  },
  "minority_proportion": {
    "type": "continuous",
    "distr": [
      {
        "min": 0,
        "max": 0.5,
        "prob": 0.5
      },
      {
        "min": 0.5,
        "max": 1.0,
        "prob": 0.5
      }
    ]
  },
  "complexity": {
    "type": "continuous",
    "distr": [
      {
        "min": 0,
        "max": 10,
        "prob": 0.2
      },
      {
        "min": 10,
        "max": 20,
        "prob": 0.3
      },
      {
        "min": 20,
        "max": 30,
        "prob": 0.3
      },
      {
        "min": 30,
        "max": 100,
        "prob": 0.2
      }
    ]
  }
}

        targetDim = ["category", "sentiment"]
        rankingObjectives = ['category', 'articleAge']
        mappingList = [
            {'category': {'politics': 1, 'sports': 2, 'entertainment': 3, 'music': 4}}, {}]
        ascending = [True, True]
        filteringCriteria = {
            "filterDim": "articleAge", "filterThreshold": 48, "comparison": "less"}
        sampleObjective = "popularity"
        model = D_RDW(item_dataframe=self.article_dataframe, diversity_dimension=targetDim, target_distributions=target_distr, targetSize=targetSize, maxHops=11,
                      filteringCriteria=filteringCriteria, sampleObjective=sampleObjective, rankingType='multi_objectives', rankingObjectives=rankingObjectives, mappingList=mappingList, ascending=ascending,
                      config_file=None)
        ############################################
        # check random walk transition probability
        ###########################################
        P1 = np.array([[0, 0, 0, 0, 0, 1/2, 0, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1/3, 1/3, 1/3, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1/4, 0, 1/4, 1/4, 1/4, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1/4, 0, 0, 0, 0, 0, 1/4, 1/4, 0, 0, 1/4],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/3, 0, 1/3, 1/3, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1/2, 0, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1/3, 1/3, 1/3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                       ])

        model.fit(self.bm.train_set)
        np.testing.assert_allclose(
            model.sampleRank.Model_RDW.P.toarray(), P1, atol=1e-6)
        ranked_list_for_user1, score = model.rank(0, None)
        ranked_list_for_user4, score = model.rank(3, None)
        ranked_list_for_user5, score = model.rank(4, None)
        expected_list = ['politics', 'politics',
                         'sports', 'music']  # Expected contents
        category_user1 = [self.Item_category[key]
                          for key in ranked_list_for_user1]
        category_user4 = [self.Item_category[key]
                          for key in ranked_list_for_user4]
        category_user5 = [self.Item_category[key]
                          for key in ranked_list_for_user5]
        self.assertListEqual(category_user1, expected_list)
        self.assertListEqual(category_user4, expected_list)
        self.assertListEqual(category_user5, expected_list)

        def count_num_in_range(numbers, min_val, max_val):
            return sum(1 for x in numbers if min_val <= x < max_val)
        sentiment_user1 = [self.Item_sentiment[key]
                           for key in ranked_list_for_user1]
        sentiment_user4 = [self.Item_sentiment[key]
                           for key in ranked_list_for_user4]
        sentiment_user5 = [self.Item_sentiment[key]
                           for key in ranked_list_for_user5]
        self.assertEqual(count_num_in_range(sentiment_user1, -1, -0.5), 1)
        self.assertEqual(count_num_in_range(sentiment_user1, -0.5, 0), 1)
        self.assertEqual(count_num_in_range(sentiment_user1, 0, 0.5), 1)
        self.assertEqual(count_num_in_range(sentiment_user1, 0.5, 1), 1)
        self.assertEqual(count_num_in_range(sentiment_user4, -1, -0.5), 1)
        self.assertEqual(count_num_in_range(sentiment_user4, -0.5, 0), 1)
        self.assertEqual(count_num_in_range(sentiment_user4, 0, 0.5), 1)
        self.assertEqual(count_num_in_range(sentiment_user4, 0.5, 1), 1)
        self.assertEqual(count_num_in_range(sentiment_user5, -1, -0.5), 1)
        self.assertEqual(count_num_in_range(sentiment_user5, -0.5, 0), 1)
        self.assertEqual(count_num_in_range(sentiment_user5, 0, 0.5), 1)
        self.assertEqual(count_num_in_range(sentiment_user5, 0.5, 1), 1)
        Experiment(eval_method=self.bm, models=[model],
                   metrics=[act, cal, repre, alt]
                   ).run()

    def test_graph_coloring(self):
        candidateItems = [0, 1, 2, 4, 5, 6, 8, 10, 11]
        articleDataframe = self.article_dataframe
        dimension = 'category'
        rank_object = ItemRanker(
            candidateItems=candidateItems, item_dataframe=articleDataframe, dimension=dimension)
        ordered_list = rank_object.rank()
        category_list = [self.Item_category[key]
                         for key in ordered_list]
        print("After ranking the categories in the recommendation list are: {}".format(
            category_list))
        for i in range(len(category_list) - 1):
            # Assert that the current element is not equal to the next element
            self.assertNotEqual(category_list[i], category_list[i + 1],
                                f"Category at positions {i} and {i + 1} should not be equal.")

    def test_round_robin(self):
        candidateItems = [3, 5, 7, 9, 0, 1]
        articleDataframe = self.article_dataframe
        dimension = 'category'
        rank_object = ItemRanker(
            candidateItems=candidateItems, item_dataframe=articleDataframe, dimension=dimension)
        ordered_list = rank_object.rank()
        category_list = [self.Item_category[key]
                         for key in ordered_list]
        print("After ranking the categories in the recommendation list are: {}".format(
            category_list))
        expected_list = ["politics", "music", "sports",
                         "politics", "politics", "politics"]
        self.assertListEqual(category_list, expected_list)

    def test_sampling(self):
        train_set = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]])
        sampleObject = Sample_And_Rank(
            train_set_rating=train_set, articlesDataframe=self.article_dataframe)
        # Test filterHeuristics
        itemPool = np.arange(0, 12)
        filteringCriteria = {
            "filterDim": "articleAge", "filterThreshold": 24, "comparison": "less"}
        # filter for item 0, history: 0 and 2
        filteredItems = sampleObject.filterHeuristics(
            0, itemPool, filteringCriteria)
        print("filteredItems:{}".format(filteredItems))
        expectedItems = np.asarray([1, 3, 4, 5, 7, 8])
        assert np.array_equal(np.sort(filteredItems), np.sort(expectedItems))
        # Test sampling
        sampleObject.itemPool = itemPool
        targetDimensions = ["entities", "minority_proportion"]
        targetDistributions = {
            "entities": {
                "type": "parties",
                "distr": [
                    {
                        "description": "only mention",
                        "contain": ["Republican"],
                        "prob": 0.25
                    },
                    {
                        "description": "only mention",
                        "contain": ["Democratic"],
                        "prob": 0.25
                    },
                      {
                        "description": "composition",
                        "contain": [["Republican"], ["Democratic"]],
                        "prob": 0.25
                    },
                    # {
                    #     "description": "only mention",
                    #     "contain": ["Republican", "Democratic"],
                    #     "prob": 0.25
                    # },
                    {
                        "description": "minority but can also mention",
                        "contain": ["Democratic", "Republican"],
                        "prob": 0.25
                    }
                ]
            },
            "minority_proportion": {
                "type": "continuous",
                "distr": [
                    {
                        "min": 0,
                        "max": 0.5,
                        "prob": 0.5
                    },
                    {
                        "min": 0.5,
                        "max": 1.0,
                        "prob": 0.5
                    }
                ]
            }
        }
        selectedTarget = []
        for i in targetDimensions:
            selectedTarget.append(targetDistributions[i])
        target_num_items, candidateItems = sampleObject.sampleArticles(
            targetDimensions, selectedTarget, 4, 'articleAge')
        print(target_num_items)
        expectedItems = [3, 4, 9, 7]
        print("candidateItems in sampling is:{}".format(candidateItems))
        self.assertCountEqual(candidateItems, expectedItems)


if __name__ == '__main__':
    unittest.main(buffer=False)
