import unittest
import numpy as np
from cornac.eval_methods import  BaseMethod
from cornac.experiment.experiment import Experiment
from cornac.models import UserKNN

from cornac.rerankers import ReRanker
from cornac.datasets import mind as mind


class TestRerank(unittest.TestCase):
    def setUp(self):
        self.dummy_data = [('u0', 'item0', 3.0), 
                           ('u1', 'item1', 1.0),
                           ('u2', 'item0', 2.0),
                           ('u3', 'item1', 4.0),
                           ('u4', 'item1', 2.0),
                           ('u5', 'item2', 5.0),
                           ('u3', 'item2', 3.0),
                           ('u3', 'item3', 3.0),
                           ('u0', 'item4', 3.0),
                           ('u2', 'item2', 3.0),
                           ('u4', 'item3', 3.0),
                           ('u5', 'item3', 3.0),
                           ('u3', 'item5', 3.0),
                           ('u4', 'item5', 3.0),
                           ('u5', 'item6', 2.0),
                           ('u0', 'item6', 1.0),
                           ('u2', 'item3', 2.0),
                           ('u1', 'item2', 3.0),
                           ('u6', 'item1', 2.0),
                          ('u6', 'item7', 2.0), # new item
                           ]
        
        self.item_feature_vectors = {
            0: np.array([0.1, 0.2, 0.3]), 
            1: np.array([0.1, 0.2, 0.3]), ## 0 and 1 has same feature vector
            2: np.array([0.4, 0.5, -0.6]),
            3: np.array([0.7, 0.8, 0.9]),
            4: np.array([0.2, 0.3, 0.4]),
            5: np.array([0.5, 0.6, 0.7]),
            6: np.array([0.6, -0.7, 0.6]),
            7: np.array([0.6, -0.7, 0.6]), ## 6 and 7 has same feature vector

        }

        self.ratio_split = BaseMethod.from_splits(data=self.dummy_data,
                                                  train_data=self.dummy_data[:-
                                                                             6], test_data=self.dummy_data[-6:],
                                                  exclude_unknowns=False, verbose=True, seed=123, rating_threshold=1)
        # self.user_item_history = {
        #     0: [0, 1],  ## because Mind dataset has a separate history for the first four weeks. additional filtering needed.
        #     1: [0, 4],
        #     6: [0], 
        # }
        self.user_item_history = {
            'u0': ['item0', 'item1'],  ## because Mind dataset has a separate history for the first four weeks. additional filtering needed.
            'u1': ['item0', 'item4'],
            'u6': ['item0'], 
        }

        # self.impression_items = [0,1,2,4,5,7] ## assume only these items appear in the test set impression logs.
        self.impression_items = ['item0','item1','item2','item4','item5','item7']

        self.reranker_normal = ReRanker(name="test_reranker", top_k=3)

        self.reranker = ReRanker(name="test_reranker", top_k=3, 
                                      item_feature_vectors=self.item_feature_vectors, 
                                      user_item_history=self.user_item_history)

        self.reranker_given_item_pool = ReRanker(name="test_reranker_limited_pool", top_k=3, 
                                      item_feature_vectors=self.item_feature_vectors, 
                                      user_item_history=self.user_item_history,
                                      rerankers_item_pool = self.impression_items
                                )
        
       
                             

    def test_init(self):
        # Assert that attributes are correctly initialized
        self.assertEqual(self.reranker_normal.top_k, 3)
        self.assertIsNone(self.reranker_normal.rerankers_item_pool, "Expected value to be None")
        self.assertIsNone(self.reranker_normal.user_item_history, "Expected 'user_item_history' to be absent")

        self.assertEqual(self.reranker.top_k, 3)
        self.assertEqual(self.reranker.pool_size, -1)

        self.assertTrue(np.array_equal(self.reranker.item_feature_vectors, self.item_feature_vectors), "The arrays are not equal")
        self.assertEqual(self.reranker.user_item_history, self.user_item_history, "The dictionaries are not equal")

          
        self.assertEqual(self.reranker_given_item_pool.rerankers_item_pool, self.impression_items)
        self.assertEqual(self.reranker_given_item_pool.user_item_history, self.user_item_history, "The dictionaries are not equal")
   
    def test_filter_items(self):
        
        ranked_items = [2, 3, 0, 1, 4, 6, 5, 7 ]
        ranked_itemscores = np.arange(len(ranked_items) - 1, -1, -1)
        ranked_itemscores = ranked_itemscores.tolist()

        user_idx = 1

        self.reranker.rerank(user_idx = user_idx, interaction_history=self.ratio_split.train_set, candidate_items=ranked_items, prediction_scores=ranked_itemscores )
        self.assertDictEqual( self.reranker.candidate_items_raw, {1:ranked_items}, "Dictionaries do not match")
        self.assertDictEqual( self.reranker.candidate_scores_raw, {1:ranked_itemscores}, "Dictionaries do not match")
        self.reranker.execute_filters(user_idx)

        expected_items_dict = {1: [2, 3, 0, 4, 6, 5, 7 ]}
        expected_scores_dict = {1: [7, 6, 5, 3, 2, 1, 0]}

        self.assertDictEqual(self.reranker.candidate_items, expected_items_dict)

        self.reranker.retrieve_prediction_scores(user_idx)
        self.assertDictEqual(self.reranker.candidate_scores, expected_scores_dict)

        self.reranker.filter_items_in_additional_history(user_idx)
        expected_items_dict = {1: [2, 3, 6, 5, 7 ]}
        expected_scores_dict = {1: [7,6,2,1,0]}
        self.assertDictEqual(self.reranker.candidate_items, expected_items_dict)
        self.reranker.retrieve_prediction_scores(user_idx)
        self.assertDictEqual(self.reranker.candidate_scores, expected_scores_dict)


        
    def test_filter_impression_items(self):
        self.assertListEqual(self.reranker_given_item_pool.rerankers_item_pool, self.impression_items, "The lists do not match")
        
      
        ranked_items_user4 = [1, 7, 3, 4, 6, 2, 0, 5]
        # itemscores = np.arange(len(ranked_items))
        # itemscores = itemscores.tolist()
        ranked_itemscores_user4 = np.arange(len(ranked_items_user4) - 1, -1, -1)

        user_idx = 4

        self.reranker_given_item_pool.rerank(user_idx = user_idx, interaction_history=self.ratio_split.train_set, candidate_items=ranked_items_user4, prediction_scores=ranked_itemscores_user4 )
        self.assertDictEqual( self.reranker_given_item_pool.candidate_items_raw, {4 : ranked_items_user4}, "Dictionaries do not match")
        self.assertDictEqual( self.reranker_given_item_pool.candidate_scores_raw, {4 : ranked_itemscores_user4}, "Dictionaries do not match")


        self.reranker_given_item_pool.execute_filters(user_idx)

        expected_items_dict = {4: [7,  4, 2,  0]}
        expected_scores_dict = {4: [ 6, 4, 2, 1]}


        self.assertDictEqual(self.reranker_given_item_pool.candidate_items, expected_items_dict)

        self.reranker_given_item_pool.retrieve_prediction_scores(user_idx)
        self.assertDictEqual(self.reranker_given_item_pool.candidate_scores, expected_scores_dict)

        self.reranker_given_item_pool.filter_items_in_additional_history(user_idx)
        self.assertDictEqual(self.reranker_given_item_pool.candidate_items, expected_items_dict)
        self.reranker_given_item_pool.retrieve_prediction_scores(user_idx)
        self.assertDictEqual(self.reranker_given_item_pool.candidate_scores, expected_scores_dict)

        ########## test for another user
        user_idx = 6
        
        ranked_items = [3,6,5,7,1,2,0,4]
        # itemscores = np.arange(len(ranked_items))
        # itemscores = itemscores.tolist()
        ranked_itemscores = np.arange(len(ranked_items) - 1, -1, -1)

        self.reranker_given_item_pool.rerank(user_idx = user_idx, interaction_history=self.ratio_split.train_set, candidate_items=ranked_items, prediction_scores=ranked_itemscores )
        
        self.assertDictEqual( self.reranker_given_item_pool.candidate_items_raw, {4 : ranked_items_user4, 6: ranked_items }, "Dictionaries do not match")
        self.assertDictEqual( self.reranker_given_item_pool.candidate_scores_raw, {4 : ranked_itemscores_user4, 6: ranked_itemscores}, "Dictionaries do not match")

        self.reranker_given_item_pool.execute_filters(user_idx)


        expected_items =  [5,7,1,2,0,4]
        expected_scores = [ 5,4,3,2,1,0]

        self.assertEqual(self.reranker_given_item_pool.candidate_items[6], expected_items)

        self.reranker_given_item_pool.retrieve_prediction_scores(user_idx)
        self.assertEqual(self.reranker_given_item_pool.candidate_scores[6], expected_scores)



        self.reranker_given_item_pool.filter_items_in_additional_history(user_idx)

        expected_items =  [5,7,1,2,4]
        expected_scores = [ 5,4,3,2,0] 
        self.assertEqual(self.reranker_given_item_pool.candidate_items[6], expected_items)
        self.reranker_given_item_pool.retrieve_prediction_scores(user_idx)
        self.assertEqual(self.reranker_given_item_pool.candidate_scores[6], expected_scores)

        
        ## check result for user 4 doesn't change
        self.assertEqual(self.reranker_given_item_pool.candidate_items[4], [7,  4, 2,  0])








    
    # def test_mmr_re_ranking(self):
    #     """Test the rerank functionality."""
    #     # initial_rank = 
    #     ranked_items = self.reranker.rerank(user_idx=1, 
    #                                         candidate_items= , 
    #                                         prediction_scores=self.prediction_scores)
    #     print(f"Ranked Items for User 1: {ranked_items}")
        
    #     # Assertions
    #     self.assertEqual(len(ranked_items), 3, "The number of ranked items should be equal to top_k")
    #     self.assertEqual(ranked_items, [4, 3, 5], "Ranked items do not match the expected outcome")

    # def test_missing_item_feature_vectors(self):
    #     """Test that missing item_feature_vectors raises an error."""
    #     with self.assertRaises(ValueError) as context:
    #         reranker = MMR_ReRanker(top_k=3, item_feature_vectors=None, user_item_history={})
    #     self.assertTrue('item_feature_vectors cannot be None' in str(context.exception))

    # def test_empty_candidate_items(self):
    #     """Test that empty candidate_items raises an error."""
    #     with self.assertRaises(ValueError) as context:
    #         self.reranker.rerank(user_idx=1, candidate_items=[], prediction_scores=self.prediction_scores)
    #     self.assertTrue('Candidate items cannot be empty for user 1' in str(context.exception))

    # def test_default_user_item_history(self):
    #     """Test behavior when user_item_history is not provided."""
    #     reranker = MMR_ReRanker(top_k=2, item_feature_vectors=self.item_feature_vectors)
        
    #     ranked_items = reranker.rerank(user_idx=1, candidate_items=[1, 2], prediction_scores={1: 0.8, 2: 0.9})
    #     print(f"Ranked Items (User 1 with default user_item_history): {ranked_items}")
        
    #     self.assertEqual(len(ranked_items), 2, "Test failed: Number of ranked items is not correct.")
    #     self.assertEqual(ranked_items, [1, 2], "Ranked items do not match the expected outcome")

        


if __name__ == '__main__':
    unittest.main()
