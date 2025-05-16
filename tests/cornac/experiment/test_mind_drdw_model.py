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
from cornac.models import NRMS, MostPop, DAE
from cornac.metrics import MAE, RMSE, Recall, FMeasure
from cornac.experiment.experiment import Experiment
from cornac.metrics import NDCG
from cornac.metrics import NDCG_score
from cornac.metrics import GiniCoeff
from cornac.metrics import ILD
from cornac.metrics import EILD, Precision
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
from cornac.data import TextModality
from cornac.data.text import BaseTokenizer
from cornac.models import D_RDW
import pandas as pd
import json


class TestExperiment(unittest.TestCase):

    def setUp(self):
        # Load feedback and create dataset split
        self.feedback = mind.load_feedback(
            fpath="./examples/example_news_files/example_impression_all_uir.csv")

        self.ratio_split = RatioSplit(
            data=self.feedback,
            test_size=0.2,
            exclude_unknowns=True,
            verbose=True,
            seed=123,
            rating_threshold=0.5,
        )

        # Load item features
        self.sentiment = mind.load_sentiment(fpath="./examples/example_news_files/example_sentiment.json")
        self.category = mind.load_category(fpath="./examples/example_news_files/example_category.json")
        self.complexity = mind.load_complexity(fpath="./examples/example_news_files/example_readability.json")
        self.story = mind.load_story(fpath="./examples/example_news_files/example_story.json")
        self.entities = mind.load_entities(fpath="./examples/example_news_files/example_party.json", keep_empty=True)
        self.genre = mind.load_category_multi(fpath="./examples/example_news_files/example_category.json")
        self.min_maj = mind.load_min_maj(fpath="./examples/example_news_files/example_minor_major_ratio.json")

        # Build item feature mappings using the ID map
        iid_map = self.ratio_split.train_set.iid_map

        self.Item_sentiment = mind.build(data=self.sentiment, id_map=iid_map)
        self.Item_category = mind.build(data=self.category, id_map=iid_map)
        self.Item_complexity = mind.build(data=self.complexity, id_map=iid_map)
        self.Item_stories = mind.build(data=self.story, id_map=iid_map)
        self.Item_entities = mind.build(data=self.entities, id_map=iid_map, keep_empty = False)
        self.Item_entities_keep_no_party = mind.build(data=self.entities, id_map=iid_map, keep_empty = True)
        self.Item_min_major = mind.build(data=self.min_maj, id_map=iid_map)
        self.Item_genre = mind.build(data=self.genre, id_map=iid_map)

        # Compose article feature DataFrame
        self.article_feature_dataframe = (
            pd.Series(self.Item_category).to_frame('category')
            .join(pd.Series(self.Item_entities_keep_no_party).to_frame('entities'), how='outer')
            .join(pd.Series(self.Item_complexity).to_frame('complexity'), how='outer')
            .join(pd.Series(self.Item_sentiment).to_frame('sentiment'), how='outer')
            .join(pd.Series(self.Item_stories).to_frame('story'), how='outer')
        )
    def test_with_drdw(self):
        
        # feedback = mind.load_feedback(
        #     fpath="./examples/example_news_files/example_impression_all_uir.csv")
        # mind_ratio_split = RatioSplit(
        #     data=feedback,
        #     test_size=0.2,
        #     exclude_unknowns=True,
        #     verbose=True,
        #     seed=123,
        #     rating_threshold=0.5,
        # )
        # # metrics
        # sentiment = mind.load_sentiment(
        #     fpath = "./examples/example_news_files/example_sentiment.json")
        # category = mind.load_category(
        #     fpath = "./examples/example_news_files/example_category.json")
        
        # complexity = mind.load_complexity(
        #     fpath="./examples/example_news_files/example_readability.json")
        # story = mind.load_story(fpath="./examples/example_news_files/example_story.json")
        # # Important! For dynamic re-ranking,  `keep_empty` need to be set as True
        # entities = mind.load_entities(fpath="./examples/example_news_files/example_party.json", keep_empty = True)

        # genre = mind.load_category_multi(
        #     fpath = "./examples/example_news_files/example_category.json")
        # min_maj = mind.load_min_maj(fpath="./examples/example_news_files/example_minor_major_ratio.json")

        # Item_sentiment = mind.build(
        #     data=sentiment, id_map=mind_ratio_split.train_set.iid_map)

        # Item_category = mind.build(
        #     data=category, id_map=mind_ratio_split.train_set.iid_map)
        # Item_complexity = mind.build(
        #     data=complexity, id_map=mind_ratio_split.train_set.iid_map)
        # Item_stories = mind.build(
        #     data=story, id_map=mind_ratio_split.train_set.iid_map)
        # Item_entities = mind.build(
        #     data=entities, id_map=mind_ratio_split.train_set.iid_map)
        # Item_min_major = mind.build(
        #     data=min_maj, id_map=mind_ratio_split.train_set.iid_map)
        # Item_genre = mind.build(
        #     data=genre, id_map=mind_ratio_split.train_set.iid_map)

        # Item_feature = Item_genre
       
        # # prepare dataframe file for d_drw model
        # article_feature_dataframe = (pd.Series(Item_category).to_frame('category')
        #                              .join( pd.Series(Item_entities).to_frame('entities'), how='outer'))

        # article_feature_dataframe = article_feature_dataframe.join(
        #     pd.Series(Item_complexity).to_frame('complexity'), how='outer')
        # article_feature_dataframe = article_feature_dataframe.join(
        #     pd.Series(Item_sentiment).to_frame('sentiment'), how='outer')
        # article_feature_dataframe = article_feature_dataframe.join(
        #     pd.Series(Item_stories).to_frame('story'), how='outer')

        drdw_model = D_RDW(
            item_dataframe=self.article_feature_dataframe, config_file='./tests/configs/model_configs/parameters.ini')

        targetSize = 20

        metrics = [
            Recall(k=targetSize),
            Precision(k=targetSize),
            Activation(item_sentiment = self.Item_sentiment,
                       divergence_type='JS', k=targetSize),
            Calibration(item_feature = self.Item_category,
                        data_type="category", divergence_type='JS', k=targetSize),
            Calibration(item_feature = self.Item_complexity,
                        data_type="complexity", divergence_type='JS', k=targetSize),

        ]

        Experiment(eval_method = self.ratio_split, models=[drdw_model],
                   metrics=metrics
                   ).run()

    def test_with_multi_objective_ranking(self):
        # Generate numbers with skewed distribution to simulate hours since news articles were published.

        def generate_skewed_data():
            # Generate a random float
            random_prob = np.random.random()
            # 70% chance to generate number < 24
            if random_prob < 0.8:
                return np.random.uniform(1, 24)
            # Additional 15% chance (total 85%) to generate number < 48
            elif random_prob < 0.95:
                return np.random.uniform(24, 48)
            # Remaining 5% chance to generate number between 48 and 72
            else:
                return np.random.uniform(48, 72)

        # Create the Dummy column with weighted random values
        self.article_feature_dataframe['articleAge'] = [generate_skewed_data()
                                for _ in range(len( self.article_feature_dataframe))]

        self.article_feature_dataframe['popularity'] = np.round(np.random.random(
            size=len( self.article_feature_dataframe)) * (1 - 1e-10) + 1e-10, 3)
        # List of outlets, dummy data
        outlets = ['CNN', 'BBC', 'NBC', 'FOX']

        # Add a new column with random values from the outlets list
        np.random.seed(42)  # for reproducibility
        self.article_feature_dataframe['outlet'] = np.random.choice(outlets, size=len( self.article_feature_dataframe))

        self.article_feature_dataframe.to_csv('out.csv', sep='\t')

        with open('./tests/configs/reranker_configs/target_distr_even.json') as f:
            target_distr = json.load(f)
        targetDim = ["category", "sentiment"]
        targetSize = 8
        rankingObjectives = ['outlet', 'popularity']
        mappingList = [
            {'outlet': {'CNN': 1, 'BBC': 2, 'NBC': 3, 'FOX': 4}}, {}]
        ascending = [False, True]
        filteringCriteria = {
            "filterDim": "articleAge", "filterThreshold": 48, "comparison": "less"}
        models1 = [
            D_RDW(item_dataframe= self.article_feature_dataframe, diversity_dimension=targetDim, target_distributions=target_distr, targetSize=targetSize, maxHops=15, filteringCriteria=filteringCriteria, 
                  rankingType='multi_objectives', rankingObjectives=rankingObjectives, mappingList=mappingList, ascending=ascending)]

        # Recall(k=targetSize), NDCG(k=targetSize),

        # FMeasure(k=targetSize), Precision(k=targetSize),
        metrics = [
            Activation( item_sentiment = self.Item_sentiment,
                       divergence_type='JS', k=targetSize),
            Calibration(item_feature=self.Item_category,
                        data_type="category", divergence_type='JS', k=targetSize),
            Calibration(item_feature=self.Item_complexity,
                        data_type="complexity", divergence_type='JS', k=targetSize),
            Fragmentation(item_story=self.Item_stories, n_samples=1,
                          divergence_type='JS', k=targetSize),
            ILD(item_feature=self.Item_genre, k=targetSize),
            NDCG_score(k=targetSize),
            EILD(item_feature=self.Item_genre, k=targetSize),
            GiniCoeff(item_genre=self.Item_genre, k=targetSize),
            AlternativeVoices(item_minor_major=self.Item_min_major,
                              divergence_type='JS', k=targetSize),
            Representation(item_entities=self.Item_entities,
                           divergence_type='JS', k=targetSize)
        ]

        Experiment(eval_method=self.ratio_split, models=models1,
                   metrics=metrics
                   ).run()


if __name__ == "__main__":
    unittest.main()
