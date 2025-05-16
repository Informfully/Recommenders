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
from cornac.eval_methods import RatioSplit
from cornac.models import NRMS, MostPop, DAE
from cornac.metrics import MAE, RMSE, Recall, FMeasure
from cornac.experiment.experiment import Experiment

from cornac.metrics import GiniCoeff
from cornac.metrics import ILD
from cornac.metrics import EILD, Precision
from cornac.metrics import Activation
from cornac.metrics import Calibration
from cornac.metrics import Fragmentation
from cornac.metrics import Representation
from cornac.metrics import AlternativeVoices

from cornac.datasets import mind as mind
from cornac.models import D_RDW
import pandas as pd


class TestExperiment(unittest.TestCase):

    def test_with_multi_models(self):
        feedback = mind.load_feedback(
            fpath="./examples/example_news_files/example_impression_all_uir.csv")
      
        mind_ratio_split = RatioSplit(
            data=feedback,
            test_size=0.2,
            exclude_unknowns=True,
            verbose=True,
            seed=123,
            rating_threshold=0.5,
        )
    
        sentiment = mind.load_sentiment(
            fpath = "./examples/example_news_files/example_sentiment.json")
        category = mind.load_category(
            fpath = "./examples/example_news_files/example_category.json")
        
        complexity = mind.load_complexity(
            fpath="./examples/example_news_files/example_readability.json")
        story = mind.load_story(fpath="./examples/example_news_files/example_story.json")
        # Important! For dynamic re-ranking,  `keep_empty` need to be set as True
        entities = mind.load_entities(fpath="./examples/example_news_files/example_party.json", keep_empty = True)

        genre = mind.load_category_multi(
            fpath = "./examples/example_news_files/example_category.json")
        min_maj = mind.load_min_maj(fpath="./examples/example_news_files/example_minor_major_ratio.json")


        Item_sentiment = mind.build(
            data=sentiment, id_map=mind_ratio_split.train_set.iid_map)

        Item_category = mind.build(
            data=category, id_map=mind_ratio_split.train_set.iid_map)
        Item_complexity = mind.build(
            data=complexity, id_map=mind_ratio_split.train_set.iid_map)
        Item_stories = mind.build(
            data=story, id_map=mind_ratio_split.train_set.iid_map)
        Item_entities = mind.build(
            data=entities, id_map=mind_ratio_split.train_set.iid_map)
        Item_min_major = mind.build(
            data=min_maj, id_map=mind_ratio_split.train_set.iid_map)
        Item_genre = mind.build(
            data=genre, id_map=mind_ratio_split.train_set.iid_map)

        Item_feature = Item_genre
        # prepare dataframe file for d_drw model
        article_feature_dataframe = (pd.Series(Item_category).to_frame('category')
                                     .join( pd.Series(Item_entities).to_frame('entities'), how='outer'))

        article_feature_dataframe = article_feature_dataframe.join(
            pd.Series(Item_complexity).to_frame('complexity'), how='outer')
        article_feature_dataframe = article_feature_dataframe.join(
            pd.Series(Item_sentiment).to_frame('sentiment'), how='outer')
        article_feature_dataframe = article_feature_dataframe.join(
            pd.Series(Item_stories).to_frame('story'), how='outer')

        dae_model = DAE(qk_dims=[20],
                        pk_dims=[20],
                        n_epochs=3)
        most_pop_model = MostPop()
        drdw_model = D_RDW(
            item_dataframe=article_feature_dataframe, config_file='./tests/configs/model_configs/parameters.ini')

        targetSize = 20

        metrics = [
            Recall(k=targetSize),
            Precision(k=targetSize),
            Activation(item_sentiment=Item_sentiment,
                       divergence_type='JS', k=targetSize),
            Calibration(item_feature=Item_category,
                        data_type="category", divergence_type='JS', k=targetSize),
            Calibration(item_feature=Item_complexity,
                        data_type="complexity", divergence_type='JS', k=targetSize),
            Fragmentation(item_story=Item_stories, n_samples=1,
                          divergence_type='JS', k=targetSize),
            ILD(name = "cat_ILD", item_feature=Item_feature, k=targetSize),
            EILD(name = "cat_EILD", item_feature=Item_feature, k=targetSize),
            GiniCoeff(name = "cat_Gini", item_genre=Item_genre, k=targetSize),
            AlternativeVoices(item_minor_major=Item_min_major,
                              divergence_type='JS', k=targetSize),
            Representation(item_entities=Item_entities,
                           divergence_type='JS', k=targetSize)
        ]

        Experiment(eval_method=mind_ratio_split, models=[dae_model, most_pop_model, drdw_model],
                   metrics=metrics
                   ).run()


if __name__ == "__main__":
    unittest.main()
