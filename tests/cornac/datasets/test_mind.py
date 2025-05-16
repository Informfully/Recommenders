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
from cornac.datasets import mind as mind
import numpy as np
import os

class TestMind(unittest.TestCase):
    def setUp(self):
        self.example_files_location = "./examples/example_news_files"

    def test_load_uir(self):
        uir_path = os.path.join(self.example_files_location, "example_impression_all_uir.csv")
        ratings = mind.load_feedback(fpath=uir_path)
        expected = 1467
        actual = len(ratings)
        self.assertTrue(abs(expected - actual) < 0.001)

    def test_load_features_json(self):
        sentiment_path = os.path.join(self.example_files_location, "example_sentiment.json")
        sentiment = mind.load_sentiment(fpath=sentiment_path)
        self.assertTrue(abs(-0.95 - sentiment['article_53']) < 0.001)
        self.assertTrue(abs(0.15 - sentiment['article_55']) < 0.001)
        self.assertTrue(abs(0.01 - sentiment['article_129']) < 0.001)
        self.assertTrue(160 - len(sentiment) < 0.001)

        category_path = os.path.join(self.example_files_location, "example_category.json")
        category = mind.load_category(fpath=category_path)
        self.assertTrue("b" == category['article_53'])
        self.assertTrue("f" == category['article_55'])
        self.assertTrue("d" == category['article_129'])
        self.assertTrue(160 - len(category) < 0.001)


        complexity_path = os.path.join(self.example_files_location, "example_readability.json")
        complexity = mind.load_complexity(fpath=complexity_path)
        self.assertTrue(abs( 24.48 - complexity['article_53']) < 0.001)
        self.assertTrue(abs( 71.83 - complexity['article_55']) < 0.001)
        self.assertTrue(abs(96.67 - complexity['article_129']) < 0.001)
        self.assertTrue(160 - len(complexity) < 0.001)

        story_path = os.path.join(self.example_files_location, "example_story.json")
        story = mind.load_story(fpath=story_path)
        self.assertTrue(abs(33 - story['article_146']) < 0.001)
        self.assertTrue(abs(9 - story['article_138']) < 0.001)
        self.assertTrue(abs(40 - story['article_20']) < 0.001)
        self.assertTrue(160 - len(story) < 0.001)

        entity_path = os.path.join(self.example_files_location, "example_party.json")
        entities = mind.load_entities(
            fpath=entity_path)
        self.assertCountEqual(entities['article_87'], ["party1"])
        self.assertCountEqual(entities['article_133'], [
                              "party9", "party8","party7" ])



        entities = mind.load_entities(
            fpath=entity_path, keep_empty = True)
        self.assertCountEqual(entities['article_8'], [])


        self.assertTrue(160 - len(entities) < 0.001)

        genre = mind.load_category_multi(fpath=category_path)
        # print(genre)
        self.assertTrue(6 - genre['article_51'].shape[0] < 0.001)
      
        min_maj_path = os.path.join(self.example_files_location, "example_minor_major_ratio.json")
        min_maj = mind.load_min_maj(fpath=min_maj_path)
        self.assertTrue(0.38 - min_maj['article_53'][0] < 0.001)
        self.assertTrue(0.62 - min_maj['article_53'][1] < 0.001)
   






if __name__ == '__main__':
    unittest.main()
