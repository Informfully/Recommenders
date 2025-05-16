import unittest
import numpy as np
from cornac.eval_methods import BaseMethod
from cornac.rerankers.mmr import MMR_ReRanker
from cornac.datasets import mind as mind

def cosine_similarity_np(vec1, vec2):
    """Compute cosine similarity between two vectors using NumPy."""
    dot_product = np.dot(vec1, vec2)  # Compute dot product
    norm_vec1 = np.linalg.norm(vec1)  # Compute L2 norm (magnitude) of vec1
    norm_vec2 = np.linalg.norm(vec2)  # Compute L2 norm (magnitude) of vec2
    
    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # Define similarity as 0 if either vector is zero
    
    return dot_product / (norm_vec1 * norm_vec2)



class TestRerank(unittest.TestCase):
    def setUp(self):
        self.dummy_data = [('u0', 'item0', 3.0), 
                           ('u1', 'item1', 1.0),
                           ('u2', 'item0', 2.0),
                           ('u3', 'item1', 4.0),
                           ('u4', 'item1', 2.0),
                           ('u5', 'item2', 5.0),
                        #    ('u3', 'item2', 3.0),
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
            0: np.array([1, 0, 0]), 
            1: np.array([1, 0, 0]), ## 0 and 1 has same feature vector
            2: np.array([0, 1, -1]),
            3: np.array([0, 0, 1]),
            4: np.array([0, 1, 1]),
            5: np.array([0, 0, 1]),
            6: np.array([0.6, -0.7, 0.6]),
            7: np.array([0.6, -0.7, 0.6]), ## 6 and 7 has same feature vector

        }

        self.ratio_split = BaseMethod.from_splits(data=self.dummy_data,
                                                  train_data=self.dummy_data[:-
                                                                             6], test_data=self.dummy_data[-6:],
                                                  exclude_unknowns=False, verbose=True, seed=123, rating_threshold=1)
        
        # self.user_item_history = {
        #     0: [0, 1],  ## because Mind dataset has a separate history for the first four weeks
        #     1: [0, 4],
        #     2: [3],
        #     6: [0], 
        # }
        self.user_item_history = {
            'u0': ['item0', 'item1'],  ## because Mind dataset has a separate history for the first four weeks
            'u1': ['item0', 'item4'],
            'u2': ['item3'],
            'u6': ['item0'], 
        }

        # self.impression_items = [0,1,2,4,5,7] ## assume only these items appear in the test set impression logs.
        self.impression_items = ['item0','item1','item2','item4','item5','item7']

        self.reranker_normal = MMR_ReRanker(top_k=3, 
                                      item_feature_vectors=self.item_feature_vectors,
                                      lamda = 0.1
                                )
        self.reranker = MMR_ReRanker(top_k=3, 
                                      item_feature_vectors=self.item_feature_vectors, 
                                      user_item_history=self.user_item_history,
                                        lamda = 0.2)

        self.reranker_given_item_pool = MMR_ReRanker(top_k=3, 
                                      item_feature_vectors=self.item_feature_vectors, 
                                      user_item_history=self.user_item_history,
                                      rerankers_item_pool = self.impression_items,
                                        lamda = 0.3
                                )
                        
    def test_init(self):
        # Assert that attributes are correctly initialized

        self.assertEqual(self.reranker_normal.top_k, 3)
        self.assertIsNone(self.reranker_normal.rerankers_item_pool, "Expected value to be None")
        self.assertIsNone(self.reranker_normal.user_item_history)

        self.assertEqual(self.reranker.top_k, 3)
        self.assertEqual(self.reranker.pool_size, -1)

        self.assertTrue(np.array_equal(self.reranker.item_feature_vectors, self.item_feature_vectors), "The arrays are not equal")
        self.assertEqual(self.reranker.user_item_history, self.user_item_history, "The dictionaries are not equal")

          
        self.assertEqual(self.reranker_given_item_pool.rerankers_item_pool, self.impression_items)
        self.assertEqual(self.reranker_given_item_pool.item_feature_vectors, self.item_feature_vectors)
        self.assertTrue(np.array_equal(self.reranker_given_item_pool.item_feature_vectors, self.item_feature_vectors), "The arrays are not equal")


    def assert_mmr_score(self, item_feature_vectors, prediction_scores, remaining_items, selected_items, lamda, actual_scores):
        """Helper function to assert MMR score correctness for a given item index."""
        if selected_items:
            for item_idx, item_id in enumerate(remaining_items):
            
                max_similarity = max(
                    cosine_similarity_np(item_feature_vectors[item_id], item_feature_vectors[selected_item])
                    for selected_item in selected_items
                )
                expected_score = lamda * prediction_scores[item_idx] - (1 - lamda) * max_similarity
                actual_score = actual_scores[item_idx]
                self.assertAlmostEqual(
                actual_score, expected_score, places=5,
                msg=f"MMR score mismatch at position {item_idx} (Item ID: {item_id}): "
                    f"Expected {expected_score}, but got {actual_score}.")
        else:
            prediction_scores = np.asarray(prediction_scores)
            max_diversity_scores = np.zeros(len(remaining_items))
            expected_score = lamda * prediction_scores - (1 - lamda) * max_diversity_scores
            np.testing.assert_array_almost_equal(expected_score, actual_scores)

                
        

       
    
    def test_mmr_diversity_scores(self):
        """Test the rerank functionality."""
        remaining_items = [100, 2, 98, 34, 65]
        selected_items = [0, 15, 36, 1000]

        item_feature_vectors = {
            0: np.array([1, 0, 0]),
            2: np.array([0.5, 0.5, 0]),
            15: np.array([0, 0.5, 0.6]),
            34: np.array([-0.1, 0.5, 0]),
            36: np.array([0, 0.5, 0.6]),
            65: np.array([1, 0, 0]),
            98: np.array([0.5, 0, 0]), 
            100: np.array([0, 1, 0]),
            1000: np.array([0, 0, 1])
        }

        prediction_scores = [0.9,0.8, 0.7, 0.6, 0.5]  ## the scores corresponding to remaining items
        lamda=0.1

        actual_scores = self.reranker.diversityScores(
            remaining_items, selected_items, item_feature_vectors, prediction_scores, lamda
        )
       
        self.assert_mmr_score(item_feature_vectors, prediction_scores, remaining_items, selected_items, lamda, actual_scores )


        ### Test empty selected
        remaining_items = [100, 2, 98, 34, 65]
        selected_items = []

        prediction_scores = [0.95,0.75, 0.7, 0.55, 0.5]  ## the scores corresponding to remaining items
        lamda=0.2

        actual_scores = self.reranker.diversityScores(
            remaining_items, selected_items, item_feature_vectors, prediction_scores, lamda
        )
       
        self.assert_mmr_score(item_feature_vectors, prediction_scores, remaining_items, selected_items, lamda, actual_scores )


        ## Test lamda 0

        remaining_items = [0, 15, 36, 1000]
        selected_items = []

        prediction_scores = [0.95,0.75, 0.7, 0.55]  ## the scores corresponding to remaining items
        lamda=0

        actual_scores = self.reranker.diversityScores(
            remaining_items, selected_items, item_feature_vectors, prediction_scores, lamda
        )
       
        self.assert_mmr_score(item_feature_vectors, prediction_scores, remaining_items, selected_items, lamda, actual_scores )



    def test_missing_item_feature_vectors(self):
        """Test that missing item_feature_vectors raises an error."""
        with self.assertRaises(ValueError) as context:
            reranker = MMR_ReRanker(top_k=3, item_feature_vectors=None, user_item_history={})
        self.assertTrue('item_feature_vectors cannot be None' in str(context.exception))

    def test_empty_candidate_items(self):
        """Test that empty candidate_items raises an error."""

        
        with self.assertRaises(ValueError) as context:
            self.reranker.rerank(user_idx=1, interaction_history=self.ratio_split.train_set,  
                                 candidate_items=[], prediction_scores=None)
        self.assertTrue('Candidate items cannot be empty for user 1' in str(context.exception))

    def test_empty_user_item_history(self):
        """Test behavior when user_item_history is not provided."""
        
        ranked_items = self.reranker_normal.rerank(user_idx=1,interaction_history=self.ratio_split.train_set,  
                                                    candidate_items=[0,1,2,3,4,5,6,7], 
                                                    prediction_scores=[0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
                                                    )

        expected_candidate_items_raw = {1:[0,1,2,3,4,5,6,7] }
        expected_candidate_score_raw = {1:[0.7,0.6,0.5,0.4,0.3,0.2,0.1,0] }
        self.assertDictEqual(self.reranker_normal.candidate_items_raw, expected_candidate_items_raw)
        self.assertDictEqual(self.reranker_normal.candidate_scores_raw, expected_candidate_score_raw)

        ## check filtering:
        expected_candidate_items = {1:[0,2,3,4,5,6,7] }
        expected_candidate_score = {1:[0.7,0.5,0.4,0.3,0.2,0.1,0] }
        self.assertDictEqual(self.reranker_normal.candidate_items, expected_candidate_items)
        self.assertDictEqual(self.reranker_normal.candidate_scores, expected_candidate_score)



        self.assertEqual(len(ranked_items), 3, "Test failed: Number of ranked items is not correct.")
        self.assertEqual(ranked_items, [0, 2, 3], "Ranked items do not match the expected outcome")

    def test_empty_impression_pool(self):
        """Test behavior when user_item_history is not provided."""
        
        ranked_items = self.reranker.rerank(user_idx=2, interaction_history=self.ratio_split.train_set,  
                                            
                                            candidate_items=[0, 1, 2, 3, 5,7], prediction_scores=[0.9,0.8,0.7,0.6,0.5,0.4])
    
        

        expected_candidate_items_raw = {2:[0, 1, 2, 3, 5,7] }
        expected_candidate_score_raw = {2:[0.9,0.8,0.7,0.6,0.5,0.4] }
        self.assertDictEqual(self.reranker.candidate_items_raw, expected_candidate_items_raw)
        self.assertDictEqual(self.reranker.candidate_scores_raw, expected_candidate_score_raw)

        ## check filtering:
        expected_candidate_items = {2:[1,5,7] }
        expected_candidate_score = {2:[0.8,0.5,0.4] }
        self.assertDictEqual(self.reranker.candidate_items, expected_candidate_items)
        self.assertDictEqual(self.reranker.candidate_scores, expected_candidate_score)

        self.assertEqual(len(ranked_items), 3, "Test failed: Number of ranked items is not correct.")
        self.assertEqual(ranked_items, [1, 5, 7], "Ranked items do not match the expected outcome")


    def test_user_history_impression_pool(self):
        """Test behavior when user_item_history is not provided."""
        print("test impression items and history")
        ranked_items = self.reranker_given_item_pool.rerank(user_idx=3, 
                                                            interaction_history=self.ratio_split.train_set,  
                                                            candidate_items=[7,6,5,4,3,2,1,0], 
                                                            prediction_scores = [1,0.9,0.8,0.52,0.51,0.5,0.4,0.3] )
        self.assertEqual(self.reranker_given_item_pool.lamda, 0.3)
        expected_candidate_items_raw = {3:[7,6,5,4,3,2,1,0] }
        expected_candidate_score_raw = {3:[1,0.9,0.8,0.52,0.51,0.5,0.4,0.3] }
        self.assertDictEqual(self.reranker_given_item_pool.candidate_items_raw, expected_candidate_items_raw)
        self.assertDictEqual(self.reranker_given_item_pool.candidate_scores_raw, expected_candidate_score_raw)

        ## check filtering:
        expected_candidate_items = {3:[7,4,2,0] }
        expected_candidate_score = {3:[1,0.52,0.5,0.3] }
        self.assertDictEqual(self.reranker_given_item_pool.candidate_items, expected_candidate_items)
        self.assertDictEqual(self.reranker_given_item_pool.candidate_scores, expected_candidate_score)
        
        self.assertEqual(len(ranked_items), 3, "Test failed: Number of ranked items is not correct.")
        self.assertEqual(ranked_items, [7,2,4], "Ranked items do not match the expected outcome")


        ranked_items = self.reranker_given_item_pool.rerank(user_idx=6, 
                                                            interaction_history=self.ratio_split.train_set,  
                                                            candidate_items=[5,6,7,4,3,2,1,0], 
                                                            prediction_scores = [1,0.9,0.8,0.52,0.51,0.5,0.4,0.3] )
        
        expected_candidate_items_raw = {3:[7,6,5,4,3,2,1,0],6: [5,6,7,4,3,2,1,0]}
        expected_candidate_score_raw = {3:[1,0.9,0.8,0.52,0.51,0.5,0.4,0.3], 6:[1,0.9,0.8,0.52,0.51,0.5,0.4,0.3] }
        self.assertDictEqual(self.reranker_given_item_pool.candidate_items_raw, expected_candidate_items_raw)
        self.assertDictEqual(self.reranker_given_item_pool.candidate_scores_raw, expected_candidate_score_raw)
        expected_candidate_items = {3:[7,4,2,0], 6:[5,7,4,2,1] }
        expected_candidate_score = {3:[1,0.52,0.5,0.3], 6:[1,0.8,0.52,0.5,0.4] }
        self.assertDictEqual(self.reranker_given_item_pool.candidate_items, expected_candidate_items)
        self.assertDictEqual(self.reranker_given_item_pool.candidate_scores, expected_candidate_score)
        self.assertEqual(ranked_items, [5,2,1], "Ranked items do not match the expected outcome")

if __name__ == '__main__':
    unittest.main()
