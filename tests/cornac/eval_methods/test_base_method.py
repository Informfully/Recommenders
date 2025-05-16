# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import unittest

from cornac.eval_methods import BaseMethod
from cornac.eval_methods import ranking_eval, diversity_eval
from cornac.data import FeatureModality, TextModality, ReviewModality, ImageModality, SentimentModality
from cornac.data import (
    FeatureModality,
    TextModality,
    ReviewModality,
    ImageModality,
    SentimentModality,
)
from cornac.data import Dataset, Reader
from cornac.metrics import MAE, AUC, Recall
from cornac.models import MF, MostPop
from cornac.metrics import NDCG_score
from cornac.metrics import GiniCoeff
from cornac.metrics import ILD
from cornac.metrics import EILD
from cornac.metrics import Activation
from cornac.metrics import Calibration
from cornac.metrics import Fragmentation
from cornac.metrics import Representation
from cornac.metrics import AlternativeVoices
from cornac.metrics import Binomial
from cornac.models import UserKNN
import numpy as np
import os

class TestBaseMethod(unittest.TestCase):
    def setUp(self):
        """Set up reusable datasets and BaseMethod instance."""
        self.data = [
            ('1', '1', 5.0), ('1', '2', 3.0), ('2', '1', 4.0), ('2', '3', 1.0),
            ('3', '2', 2.0), ('3', '3', 5.0), ('4', '1', 2.0), ('4', '3', 4.0)
        ]
        self.train_data = self.data[:5]
        self.test_data = self.data[5:]
        self.method = BaseMethod.from_splits(
            train_data=self.train_data,
            test_data=self.test_data,
            fmt="UIR",
            verbose=True
        )

    def test_init(self):
        bm = BaseMethod(None, verbose=True)
        self.assertTrue(bm.exclude_unknowns)
        self.assertEqual(bm.rating_threshold, 1.0)

    def test_trainset_none(self):
        bm = BaseMethod(None, verbose=True)
        try:
            bm.evaluate(None, {}, False)
        except ValueError:
            assert True

    def test_testset_none(self):
        bm = BaseMethod(None, verbose=True)
        bm.train_set = Dataset.from_uir(data=Reader().read("./tests/data.txt"))
        try:
            bm.evaluate(None, {}, False)
        except ValueError:
            assert True

    def test_from_splits(self):
        data = Reader().read("./tests/data.txt")
        try:
            BaseMethod.from_splits(train_data=None, test_data=None)
        except ValueError:
            assert True

        try:
            BaseMethod.from_splits(train_data=data, test_data=None)
        except ValueError:
            assert True

        try:
            BaseMethod.from_splits(
                train_data=data, test_data=[], exclude_unknowns=True)
        except ValueError:
            assert True

        bm = BaseMethod.from_splits(train_data=data[:-1], test_data=data[-1:])
        self.assertEqual(bm.total_users, 10)
        self.assertEqual(bm.total_items, 10)

        bm = BaseMethod.from_splits(
            train_data=data[:-1],
            test_data=data[-1:],
            val_data=[(data[0][0], data[1][1], 5.0)],
            verbose=True,
        )
        self.assertEqual(bm.total_users, 10)
        self.assertEqual(bm.total_items, 10)

    def test_with_modalities(self):
        data = Reader().read("./tests/data.txt")
        sentiment_data = Reader().read(
            "./tests/sentiment_data.txt", fmt="UITup", sep=",", tup_sep=":"
        )
        review_data = Reader().read("./tests/review.txt", fmt="UIReview")
        bm = BaseMethod.from_splits(train_data=data[:-1], test_data=data[-1:])

        self.assertIsNone(bm.user_feature)
        self.assertIsNone(bm.user_text)
        self.assertIsNone(bm.item_feature)
        self.assertIsNone(bm.item_text)
        self.assertIsNone(bm.user_image)
        self.assertIsNone(bm.item_image)
        self.assertIsNone(bm.user_graph)
        self.assertIsNone(bm.item_graph)
        self.assertIsNone(bm.sentiment)

        bm.user_feature = FeatureModality()
        bm.user_text = TextModality()
        bm.item_text = ReviewModality(data=review_data, filter_by="item")
        bm.item_image = ImageModality()
        bm.sentiment = SentimentModality(data=sentiment_data)
        bm._build_modalities()

        try:
            bm.user_feature = ()
        except ValueError:
            assert True


        try:
            bm.item_feature = ()
        except ValueError:
            assert True

        try:
            bm.user_text = ImageModality()
        except ValueError:
            assert True

        try:
            bm.item_text = ImageModality()
        except ValueError:
            assert True

        try:
            bm.user_image = TextModality()
        except ValueError:
            assert True

        try:
            bm.item_image = TextModality()
        except ValueError:
            assert True

        try:
            bm.user_graph = TextModality()
        except ValueError:
            assert True

        try:
            bm.item_graph = ImageModality()
        except ValueError:
            assert True

        try:
            bm.sentiment = TextModality()
        except ValueError:
            assert True

        try:
            bm.sentiment = ImageModality()
        except ValueError:
            assert True

    def test_organize_metrics(self):
        rating_metrics, ranking_metrics,diversity_metrics = BaseMethod.organize_metrics([MAE(), AUC()])
        self.assertEqual(len(rating_metrics), 1)  # MAE
        self.assertEqual(len(ranking_metrics), 1)  # AUC

        try:
            BaseMethod.organize_metrics(None)
        except ValueError:
            assert True

    def test_evaluate(self):
        data = Reader().read("./tests/data.txt")
        bm = BaseMethod.from_splits(train_data=data[:-1], test_data=data[-1:])
        model = MF(k=1, max_iter=0)
        result = bm.evaluate(model, metrics=[MAE()], user_based=False)
        result.__str__()

    def test_evaluate_diversity(self):
        data = [('196', '242', 3.0),
                ('22', '377', 1.0),
                ('244', '51', 2.0),
                ('298', '474', 4.0),
                ('115', '474', 2.0),
                ('253', '346', 5.0),
                ('253', '242', 3.0),
                ('298', '302', 3.0),
                ('196', '302', 3.0),
                ('22', '346', 1.0),]
        bm = BaseMethod.from_splits(train_data=data[:-3], test_data=data[-3:])
        model = UserKNN(k=3, similarity="pearson", name="UserKNN-Pearson")
        ndcg = NDCG_score(k=2)
        Item_genre = {0: np.array([0, 0, 1, 0]), 1: np.array([0, 0, 1, 0]), 2: np.array([1, 0, 0, 0]), 3: np.array([0, 0, 0, 1]),
                      4: np.array([0, 1, 0, 0]), 5: np.array([0, 0, 1, 0])}
        Item_feature = {0: np.array([1, 2, 3]), 1: np.array([1, 2, 3]), 2: np.array([4, 2, 5]), 3: np.array([4, 2, 5]),
                        4: np.array([4, 2, 5]), 5: np.array([0, 2, 3])}
        Item_category = {0: 2, 1: 2, 2: 0, 3: 3,
                         4: 1, 5: 3}
        Item_sentiment = {0: 0.5, 1: -0.2, 2: 0, 3: 0.8,
                          4: 1, 5: -0.7}
        Item_stories = {0: 1, 1: 2, 2: 15, 3: 2, 4: 10, 5: 15}
        Item_entities = {0: ["Democrat", "Republican", "Republican", "Party1"], 1: ["AnyParty", "Republican", "Republican", "Party1"], 2: ["Party1", "Republican", "Republican", "Republican"],
                         3: ["AnyParty", "Democrat", "Democrat", "Democrat"], 4: ["Republican", "Democrat", "Democrat", "Democrat"],
                         5: ["Party1", "Republican", "Party1", "Party1"]}
        Item_min_major = {0:  np.array([0.1,  0.9]), 1: np.array([0.2, 0.8]), 2: np.array([0, 1]),
                          3: np.array([0.5, 0.5]), 4: np.array([0.25, 0.75]), 5: np.array([0.4, 0.6])}
        eild = EILD(item_feature=Item_feature, k=2)

        gini = GiniCoeff(item_genre=Item_genre, k=2)
        ild = ILD(item_feature=Item_feature, k=2)
        cal = Calibration(item_feature=Item_category,
                          data_type="category", k=2)
        act = Activation(item_sentiment=Item_sentiment, k=2)
        frag = Fragmentation(item_story=Item_stories, n_samples=1, k=2)
        repre = Representation(item_entities=Item_entities, k=2)
        alt = AlternativeVoices(item_minor_major=Item_min_major, k=2)
        bino = Binomial(item_genre=Item_genre, k=2)
        result = bm.evaluate(model, metrics=[
                             ndcg, gini, eild, ild, act, cal, frag, repre, alt, bino], user_based=True)
        result.__str__()

    def test_save_and_load(self):
        """Test saving and loading the BaseMethod instance."""
        save_path = "./tests/eval_methods/saved_dataset"
        self.method.save(save_path)
        loaded_method = BaseMethod.load(save_path)
        self.assertEqual(self.method.total_users, loaded_method.total_users)
        self.assertEqual(self.method.total_items, loaded_method.total_items)

    def test_cache_usage(self):
        """Test that caching is working correctly."""
        model = UserKNN(k=2)
        self.method.evaluate(model, metrics=[Recall(k=2)], user_based=True)
        self.assertTrue(hasattr(model, 'ranked_items'))
        self.assertTrue(hasattr(model, 'item_scores'))

    def test_ranking_eval(self):
        """Test ranking evaluation."""

        model = UserKNN(k=2)
        model.fit(self.method.train_set)
        recall = Recall(k=2)

        # Run evaluation
        avg_results, user_results = ranking_eval(
            model=model,
            metrics=[recall],
            train_set=self.method.train_set,
            test_set=self.method.test_set,
            verbose=True
        )

        # Check if results are generated
        self.assertGreater(len(avg_results), 0,
                           "No average results were generated.")
        self.assertGreater(len(user_results), 0,
                           "No user-specific results were generated.")


    def test_diversity_eval(self):
        """Test diversity evaluation."""

        # Define mock item features
        item_features = {0: np.array([1, 2]), 1: np.array(
            [3, 4]), 2: np.array([5, 6])}
        item_genres = {0: np.array([1, 0]), 1: np.array(
            [0, 1]), 2: np.array([1, 1])}

        # Define diversity metrics
        ild = ILD(item_feature=item_features, k=2)
        gini = GiniCoeff(item_genre=item_genres, k=2)

        model = UserKNN(k=2)
        model.fit(self.method.train_set)
        # Run diversity evaluation
        avg_results, user_results = diversity_eval(
            model=model,
            metrics=[ild, gini],
            train_set=self.method.train_set,
            test_set=self.method.test_set,
            verbose=True
        )

        # Check if results are generated
        self.assertGreater(len(avg_results), 0,
                           "No average results were generated.")
        self.assertGreater(len(user_results), 0,
                           "No user-specific results were generated.")
    
    def test_save_load_dataset(self):
        test_dir = './tests/output'
        self.method.save(test_dir)
        expected_files = ['train_set.pkl', 'test_set.pkl', 'dataset_attributes.pkl']
        for filename in expected_files:
            file_path = os.path.join(test_dir, filename)
            self.assertTrue(os.path.exists(file_path), f"{filename} should exist after save")

        loaded_dataset = BaseMethod.load(test_dir)
        self.assertEqual(self.method.verbose, loaded_dataset.verbose)
        self.assertEqual(self.method.global_uid_map, loaded_dataset.global_uid_map)





if __name__ == "__main__":
    unittest.main()
