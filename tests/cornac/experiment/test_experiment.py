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
import numpy as np
from cornac.data import Reader
from cornac.eval_methods import RatioSplit, CrossValidation
from cornac.eval_methods import BaseMethod
from cornac.models import PMF
from cornac.models import ENMF
from cornac.metrics import MAE, RMSE, Recall, FMeasure
from cornac.experiment.experiment import Experiment
from cornac.metrics import NDCG
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
from cornac.datasets import mind as mind
from cornac.datasets import amazon_clothing
from cornac.datasets import movielens


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.data = Reader().read("./tests/data.txt")

    def test_with_ratio_split(self):
        Experiment(
            eval_method=RatioSplit(
                self.data + [(self.data[0][0], self.data[1][1], 5.0)],
                exclude_unknowns=True,
                seed=123,
                verbose=True,
            ),
            models=[PMF(1, 0), UserKNN(k=3, similarity="pearson",
                                       name="UserKNN")],
            metrics=[MAE(), RMSE()],
            verbose=True
        ).run()

        try:
            Experiment(None, None, None)
        except ValueError:
            assert True

        try:
            Experiment(None, [PMF(1, 0)], None)
        except ValueError:
            assert True

    def test_with_enmf_model(self):
        feedback = amazon_clothing.load_feedback(
            reader=Reader(bin_threshold=1.0))
        models = [ENMF()]
        rs = RatioSplit(
            data=feedback,
            test_size=0.2,
            rating_threshold=1.0,
            seed=123,
            exclude_unknowns=True,
            verbose=True,
        )
        Experiment(
            eval_method=rs,
            models=[ENMF(num_epochs=1)],
            metrics=[Recall(10), FMeasure(10), MAE()],
            verbose=True,
        ).run()

    def test_with_enmf_movielens(self):
        feedback = movielens.load_feedback(
            reader=Reader(bin_threshold=1.0))
        # movielens dataset min rating = 1
        # bin_threshold (float ) –
        # The rating threshold to binarize rating values (turn explicit feedback to implicit feedback).
        # For example, if bin_threshold = 3.0, all rating values >= 3.0 will be set to 1.0, and the rest (< 3.0) will be discarded.
        rs = RatioSplit(
            data=feedback,
            test_size=0.2,
            rating_threshold=1,
            seed=123,
            exclude_unknowns=True,
            verbose=True,
        )
        Experiment(
            eval_method=rs,
            models=[ENMF(num_epochs=1)],
            metrics=[Recall(50), NDCG(50)],
            verbose=True,
        ).run()

   

    def test_with_cross_validation(self):
        Experiment(
            eval_method=CrossValidation(
                self.data + [(self.data[0][0], self.data[1][1], 5.0)],
                exclude_unknowns=False,
                verbose=True,
            ),
            models=[PMF(1, 0)],
            metrics=[Recall(1), FMeasure(1)],
            verbose=True
        ).run()

    def test_mind(self):
        feedback = mind.load_feedback(
            fpath="./examples/example_news_files/example_impression_all_uir.csv")
        # Mind dataset min rating = 1
        sentiment = mind.load_sentiment(
            fpath="./examples/example_news_files/example_sentiment.json")
        category = mind.load_category(
            fpath="./examples/example_news_files/example_category.json")
        complexity = mind.load_complexity(
            fpath="./examples/example_news_files/example_readability.json")
        story = mind.load_story(fpath="./examples/example_news_files/example_story.json")
        entities = mind.load_entities(fpath="./examples/example_news_files/example_party.json")

        genre = mind.load_category_multi(
            fpath="./examples/example_news_files/example_category.json")
        min_maj = mind.load_min_maj(fpath="./examples/example_news_files/example_minor_major_ratio.json")

        rs = RatioSplit(
            data=feedback,
            test_size=0.2,
            rating_threshold=1,
            seed=123,
            exclude_unknowns=True,
            verbose=True,
        )

        Item_sentiment = mind.build(
            data=sentiment, id_map=rs.train_set.iid_map)
        Item_category = mind.build(data=category, id_map=rs.train_set.iid_map)
        Item_complexity = mind.build(
            data=complexity, id_map=rs.train_set.iid_map)
        Item_stories = mind.build(data=story, id_map=rs.train_set.iid_map)
        Item_genre = mind.build(data=genre, id_map=rs.train_set.iid_map)
        Item_entities = mind.build(data=entities, id_map=rs.train_set.iid_map)
        Item_min_major = mind.build(data=min_maj, id_map=rs.train_set.iid_map)
        Item_feature = Item_genre
        print(len(Item_complexity))

        act = Activation(item_sentiment=Item_sentiment, k=600)
        cal = Calibration(item_feature=Item_category,
                          data_type="category", k=600)
        
        cal_complexity = Calibration(
            item_feature=Item_complexity, data_type="complexity", k=600)
        bino = Binomial(item_genre=Item_genre, k=600)
        alt = AlternativeVoices(item_minor_major=Item_min_major, k=600)
        repre = Representation(item_entities=Item_entities, k=600)
        frag = Fragmentation(item_story=Item_stories, n_samples=1, k=600)
        ild = ILD(item_feature=Item_feature, k=200)
        ndcg = NDCG_score(k=200)
        eild = EILD(item_feature=Item_feature, k=200)
        gini = GiniCoeff(item_genre=Item_genre, k=200)
        Experiment(eval_method=rs,
                   models=[UserKNN(k=3, similarity="pearson",
                                   name="UserKNN")],
                   metrics=[act, cal, cal_complexity, bino,
                            gini, frag, ild, eild, repre, alt]
                   ).run()

    def test_with_diversity(self):
        data1 = [('196', '242', 3.0),  # 0
                 ('22', '377', 1.0),  # 1
                 ('244', '51', 2.0),  # 2
                 ('298', '474', 4.0),  # 3
                 ('115', '474', 2.0),  # 4
                 ('253', '346', 5.0),  # 5
                 ('253', '242', 3.0),  # 5
                 ('298', '51', 3.0),  # 3
                 ('196', '302', 3.0),  # 0
                 ('22', '346', 1.0)]  # 1
        Item_genre = {0: np.array([0, 0, 1, 0]), 1: np.array([0, 0, 1, 0]), 2: np.array([1, 0, 0, 0]), 3: np.array([0, 0, 0, 1]),
                      4: np.array([0, 1, 0, 0]), 5: np.array([0, 0, 1, 0])}
        Item_feature = {0: np.array([1, 2, 0]), 1: np.array([1, 2, 3]), 2: np.array([4, 2, 9]), 3: np.array([4, 2, 5]),
                        4: np.array([4, 3, 5]), 5: np.array([0, 2, 3])}
        Item_category = {0: 2, 1: 1, 2: 0, 3: 0,
                         4: 1, 5: 3}
 
        Item_sentiment = {0: 0.5, 1: -0.2, 2: 0, 3: 0.8,
                          4: 1, 5: -0.7}
        Item_stories = {0: 1, 1: 3, 2: 18, 3: 2, 4: 22, 5: 15}
        Item_entities = {0: ["Democrat", "Republican", "Republican", "Party1"], 1: ["AnyParty", "Republican", "Republican", "Party1"], 2: ["Party1", "Republican", "Republican", "Republican"],
                         3: ["AnyParty", "Democrat", "Democrat", "Democrat"], 4: ["Republican", "Democrat", "Democrat", "Democrat"],
                         5: ["Party1", "Republican", "Party1", "Party1"]}
        Item_min_major = {0:  np.array([0.1,  0.9]), 1: np.array([0.2, 0.8]), 2: np.array([0, 1]),
                          3: np.array([0.5, 0.5]), 4: np.array([0.25, 0.75]), 5: np.array([0.4, 0.6])}
        ndcg = NDCG_score(k=2)
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
        ratio_split = RatioSplit(
            data=data1, test_size=3, exclude_unknowns=True, verbose=True, seed=123, rating_threshold=3)

        Experiment(eval_method=ratio_split, models=[UserKNN(k=3, similarity="pearson", name="UserKNN"), PMF(1, 0)],

                   metrics=[bino, ndcg]
                   ).run()


if __name__ == "__main__":
    unittest.main()
