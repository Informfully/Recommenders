import unittest

import numpy as np
from cornac.metrics.diversity import DiversityMetric
from cornac.metrics import NDCG_score
from cornac.metrics import GiniCoeff
from cornac.metrics import EILD
from cornac.metrics import ILD
from cornac.metrics import Binomial
from cornac.metrics import Activation
from cornac.metrics import Calibration
from cornac.metrics import Fragmentation
from cornac.metrics import Representation
from cornac.metrics import AlternativeVoices
from cornac.metrics import Alpha_NDCG
# Update: change float precision.


class TestDiversity(unittest.TestCase):
    def test_diversity_metric(self):
        metric = DiversityMetric()

        self.assertEqual(metric.type, "diversity")
        self.assertIsNone(metric.name)
        self.assertEqual(metric.k, -1)
        try:
            metric.compute()
        except NotImplementedError:
            assert True

    def test_gini(self):
        Item_genre = {1: np.array([0, 0, 1, 0]), 2: np.array([1, 0, 0, 0]), 3: np.array([0, 0, 0, 1]),
                      4: np.array([0, 1, 0, 0]), 5: np.array([0, 0, 1, 0]), 6: np.array([0, 0, 1, 0])}
        gini = GiniCoeff(item_genre=Item_genre)
        self.assertEqual(gini.type, "diversity")
        self.assertEqual(gini.name, "GiniCoeff@-1")
        rec_list = np.asarray([1, 6, 5, 2])  # [1, 3, 2]
        expected = 0.8333333333333334
        actual = gini.compute(np.asarray(rec_list))
        self.assertTrue(abs(expected - actual) < 0.001)
        gini_2 = GiniCoeff(item_genre=Item_genre, k=2)
        self.assertEqual(gini_2.k, 2)
        expected = 1
        actual = gini_2.compute(np.asarray(rec_list))
        self.assertTrue(abs(expected - actual) < 0.001)

    def test_binomial(self):
        Item_genre = {1: np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0]), 2: np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0]), 3: np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 4: np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0]),
                      5: np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0]), 6: np.array([1, 0, 1, 0, 0, 0, 1, 0, 1, 1]), 7: np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1]), 8: np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 0]),
                      9: np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0]), 10: np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])}
        bino = Binomial(item_genre=Item_genre, alpha=1)
        self.assertEqual(bino.type, "diversity")
        self.assertEqual(bino.name, "Binomial@-1")
        user_history = np.asarray([5, 8, 1, 2])
        rec_list = np.asarray([2, 1, 9, 10])
        user_preference = {1: [5, 8, 1, 2], 2: [1, 6, 3, 7], 3: [1, 2, 4, 6]}
        globalProb = bino.globalFeatureProbs(user_preference)
        actual = bino.compute(rec_list, globalProb, user_history)
        expected = 0.7254643159681841
        self.assertTrue(abs(expected - actual) < 0.001)

        bino_2 = Binomial(item_genre=Item_genre, k=2)
        self.assertEqual(bino_2.k, 2)
        expected = 0.7424229077314783
        actual = bino_2.compute(rec_list, globalProb, user_history)
        self.assertTrue(abs(expected - actual) < 0.001)

    def test_eild(self):
        item_vectors = {0: np.array([10, 1, 22, 0, 99, 5]), 1: np.array(
            [10, 1, 22, 3, 99, 5]), 2: np.array([1, 12, 100, 2, 40, 5]), 3: np.array([0, 2, 2, 3, 4, 9])}
        eild = EILD(item_feature=item_vectors)
        self.assertEqual(eild.type, "diversity")
        self.assertEqual(eild.name, "EILD@-1")
        rec_list = np.asarray([1, 3])
        gd_relevance = [0, 0.4, 5, 0.5]
        expected = 0.5257061364698357
        actual = eild.compute(np.asarray(rec_list),
                              gd_relevance, rating_threshold=0.1)
        self.assertTrue(abs(expected - actual) < 0.001)

        eild_2 = EILD(item_feature=item_vectors, k=1)
        self.assertEqual(eild_2.k, 1)
        expected = 0
        actual = eild_2.compute(np.asarray(rec_list),
                                gd_relevance, rating_threshold=0.1)
        self.assertTrue(abs(expected - actual) < 0.001)

    def test_ild(self):
        item_vectors = {0: np.array([1, 2, 3]), 1: np.array([1, 2, 3]), 2: np.array([4, 2, 5]), 3: np.array([4, 2, 5]),
                        4: np.array([4, 2, 5]), 5: np.array([0, 2, 3]), 6: np.array([4, 1, 5]),
                        7: np.array([3, 2, 5]), 8: np.array([0, 2, 5])}
        ild = ILD(item_feature=item_vectors)
        self.assertEqual(ild.type, "diversity")
        self.assertEqual(ild.name, "ILD@-1")
        rec_list = np.asarray([1, 2, 3, 4])
        expected = 0.041829033088482404
        actual = ild.compute(np.asarray(rec_list))
        self.assertTrue(abs(expected - actual) < 0.001)

        rec_list = np.asarray([2, 4, 3, 5])
        expected = 0
        ild2 = ILD(item_feature=item_vectors, k=2)
        self.assertEqual(ild2.k, 2)
        actual = ild2.compute(np.asarray(rec_list))
        self.assertTrue(abs(expected - actual) < 0.001)

    def test_ndcg_score(self):
        ndcg = NDCG_score()
        # Item_Relevance ={1:1, 2:2,3:3 }
        Item_Relevance = np.asarray([1, 2, 3])
        rec_list = np.asarray([0, 1, 2])
        self.assertEqual(ndcg.type, "diversity")
        self.assertEqual(ndcg.name, "NDCG_score@-1")
        expected = 0.7899980042460358
        actual = ndcg.compute(np.asarray(rec_list), Item_Relevance)
        self.assertTrue(abs(expected - actual) < 0.001)

        ndcg_2 = NDCG_score(k=2)
        self.assertEqual(ndcg_2.k, 2)
        expected = 0.8597186998521972

        actual = ndcg_2.compute(np.asarray(rec_list), Item_Relevance)
        self.assertTrue(abs(expected - actual) < 0.001)

    def test_alpha_ndcg(self):
        Item_genre = {0: np.array([0, 1, 0, 1, 0, 0]), 1: np.array(
            [0, 1, 0, 0, 0, 0]), 2: np.array([0, 1, 0, 0, 0, 0]), 3: np.array([0, 0, 0, 0, 0, 0]),
            4: np.array([1, 0, 0, 0, 0, 1]), 5: np.array([1, 0, 0, 0, 0, 0]), 6: np.array([0, 0, 1, 0, 0, 0]), 7: np.array([1, 0, 0, 0, 0, 0]), 8: np.array([1, 1, 1, 1, 1, 1]),
            9: np.array([0, 1, 0, 1, 0, 0]), 10: np.array([1, 1, 0, 0, 0, 0]),
            11: np.array([0, 1, 1, 1, 0, 0]), 12: np.array([0, 0, 0, 1, 0, 1]),
            13: np.array([0, 1, 0, 1, 1, 1]), 14: np.array([0, 0, 0, 0, 0, 0])}
        alpha_ndcg = Alpha_NDCG(item_genre=Item_genre)
        rec_list = np.asarray([0, 1, 2, 3, 4, 5, 6, 7])
        history = np.asarray([8])
        self.assertEqual(alpha_ndcg.type, "diversity")
        self.assertEqual(alpha_ndcg.name, "Alpha_NDCG@-1")
        expected = 0.8759994181683827
        actual = alpha_ndcg.compute(rec_list, history)
        self.assertTrue(abs(expected - actual) < 0.001)

        alpha_ndcg2 = Alpha_NDCG(item_genre=Item_genre, k=5, alpha=0.8)
        self.assertEqual(alpha_ndcg2.k, 5)
        actual = alpha_ndcg2.compute(rec_list, history)
        expected = 0.8641067652921767
        self.assertTrue(abs(expected - actual) < 0.001)

        rec_list1 = np.asarray([11, 12, 13, 14])
        history1 = np.asarray([9, 10])
        expected1 = 0.9879801444388844
        actual1 = alpha_ndcg.compute(rec_list1, history1)
        self.assertTrue(abs(expected1 - actual1) < 0.001)

        rec_list2 = np.asarray([14, 12])
        history2 = np.asarray([1, 2])
        expected2 = 0.0
        actual2 = alpha_ndcg.compute(rec_list2, history2)
        self.assertTrue(abs(expected2 - actual2) < 0.001)

    def test_activation(self):
        Item_sentiment = {0: 0.5, 1: -0.2, 2: 0, 3: 0.8,
                          4: 1, 5: -0.7}
        with self.assertRaises(ValueError) as excinfo:
            act = Activation(item_sentiment=Item_sentiment, n_bins=-1)
        self.assertEqual("Activation received an invalid number "
                         "of bins. Number of bins "
                         "must be at least 2, and must be an int.", str(excinfo.exception))
        act = Activation(item_sentiment=Item_sentiment)
        self.assertEqual(act.type, "diversity")
        self.assertEqual(act.name, "Activation@-1")
        rec_list = np.asarray([0, 1, 3])
        pool_list = np.asarray([0, 1, 2, 3, 4, 5])
        expected = 2.985336
        actual = act.compute(rec_list, pool_list)
        self.assertTrue(abs(expected - actual) < 0.001)

        expected = 6.107726
        act2 = Activation(item_sentiment=Item_sentiment, k=2)
        self.assertEqual(act2.k, 2)
        actual = act2.compute(rec_list, pool_list)
        self.assertTrue(abs(expected - actual) < 0.001)

        expected = 0.434599
        act3 = Activation(item_sentiment=Item_sentiment,
                          n_bins=4, k=2, divergence_type="js")
        self.assertEqual(act3.k, 2)
        actual = act3.compute(rec_list, pool_list)
        self.assertTrue(abs(expected - actual) < 0.001)

    def test_calibration(self):
        item_vectors = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
        cal = Calibration(item_feature=item_vectors, data_type="category")
        self.assertEqual(cal.type, "diversity")
        self.assertEqual(cal.name, "Calibration_category@-1")
        rec_list = np.asarray([1, 2, 3, 4, 5])
        history_list = np.asarray([1, 2, 3, 4, 5, 9, 8, 7, 6, 2, 2, 3])
        # expected = 3.039184897133893
        expected = 3.039375
        actual = cal.compute(rec_list, history_list)
        self.assertTrue(abs(expected - actual) < 0.001)

        cal1 = Calibration(item_feature=item_vectors, k=3,
                           data_type="complexity", divergence_type="js")
        self.assertEqual(cal1.type, "diversity")
        self.assertEqual(cal1.name, "Calibration_complexity@3")
        self.assertEqual(cal1.k, 3)
        rec_list = np.asarray([1, 2, 3, 4, 5])
        history_list = np.asarray([1, 2, 3, 4, 5, 9, 8, 7, 6, 2, 2, 3])
        # expected = 0.5012309744612722
        expected = 0.500962
        actual = cal1.compute(rec_list, history_list)
        self.assertTrue(abs(expected - actual) < 0.001)

        item_vectors1 = {1: "action", 2: "drama", 3: "comedy", 4: "comedy",
                         5: "western", 6: "romance", 7: "romance", 8: "drama", 9: "comedy"}
        cal2 = Calibration(item_feature=item_vectors1,
                           data_type="category", divergence_type="js")
        self.assertEqual(cal2.type, "diversity")
        self.assertEqual(cal2.name, "Calibration_category@-1")
        rec_list = np.asarray([1, 2, 3, 4, 5])
        history_list = np.asarray([1, 2, 3, 4, 5, 9, 8, 7, 6, 2, 2, 3])
        # expected = 0.36362464942732303
        expected = 0.363625
        actual = cal2.compute(rec_list, history_list)
        self.assertTrue(abs(expected - actual) < 0.001)

        cal3 = Calibration(item_feature=item_vectors, k=3,
                           data_type="complexity", divergence_type="js")
        self.assertEqual(cal3.type, "diversity")
        self.assertEqual(cal3.name, "Calibration_complexity@3")
        self.assertEqual(cal3.k, 3)
        rec_list = np.asarray([1, 2, 3, 4, 5])
        history_list = np.asarray([1])
        actual = cal3.compute(rec_list, history_list)
        self.assertTrue(actual is None)

    def test_fragmentation(self):
        item_stories = {0: 1, 1: 2, 2: 22, 3: 5,
                        4: 10, 5: 15, 6: 10, 7: 15, 8: 22, 9: 1}
        frag = Fragmentation(item_story=item_stories)
        self.assertEqual(frag.type, "diversity")
        self.assertEqual(frag.name, "Fragmentation@-1")
        rec_list = np.asarray([0, 1, 6, 5])
        pd_other_users = [np.asarray([0, 2, 6, 9]), np.asarray([4, 3, 7, 9])]
        # expected = 2.734191898820655
        expected = 2.734192
        actual = frag.compute(rec_list, pd_other_users)
        self.assertTrue(abs(expected - actual) < 0.001)

        # expected = 7.458309139542626
        expected = 7.458309
        frag2 = Fragmentation(item_story=item_stories, k=2)
        self.assertEqual(frag2.k, 2)
        pd_other_users1 = [np.asarray([0, 2]), np.asarray([4, 3])]
        actual = frag2.compute(rec_list, pd_other_users1)
        self.assertTrue(abs(expected - actual) < 0.001)

        # expected = 0.8486708608256677
        expected = 0.848671
        frag3 = Fragmentation(item_story=item_stories,
                              divergence_type="js", k=2)
        self.assertEqual(frag3.k, 2)
        pd_other_users1 = [np.asarray([0, 2]), np.asarray([4, 3])]
        actual = frag3.compute(rec_list, pd_other_users1)
        self.assertTrue(abs(expected - actual) < 0.001)

    def test_representation(self):
        Item_entities = {0: ["Democrat", "Republican", "Republican", "Party1"], 1: ["AnyParty", "Republican", "Republican", "Party1"], 2: [
            "Party1", "Republican", "Republican", "Republican"], 3: ["AnyParty", "Democrat", "Democrat", "Democrat"]}
        repre = Representation(item_entities=Item_entities)
        self.assertEqual(repre.type, "diversity")
        self.assertEqual(repre.name, "Representation@-1")
        rec_list = np.asarray([0, 2, 1])
        pool_list = np.asarray([0, 1, 2, 3])
        # expected = 0.20900290285676065
        expected = 0.209111
        actual = repre.compute(rec_list, pool_list)
        self.assertTrue(abs(expected - actual) < 0.001)

        # expected = 1.1911816116091571
        expected = 1.191182
        repre2 = Representation(item_entities=Item_entities, k=2)
        self.assertEqual(repre2.k, 2)
        actual = repre2.compute(rec_list, pool_list)
        self.assertTrue(abs(expected - actual) < 0.001)

        # expected = 0.30366279787425554
        expected = 0.303663
        repre3 = Representation(
            item_entities=Item_entities, k=2, divergence_type="js")
        self.assertEqual(repre3.k, 2)
        actual = repre3.compute(rec_list, pool_list)
        self.assertTrue(abs(expected - actual) < 0.001)

    def test_AlternativeVoices(self):
        item_min_major = {0:  np.array([1,  9]), 1: np.array([2, 8]), 2: np.array([0, 1]),
                          3: np.array([5, 8])}
        alt = AlternativeVoices(item_minor_major=item_min_major)
        self.assertEqual(alt.type, "diversity")
        self.assertEqual(alt.name, "AltVoices_mainstream@-1")
        rec_list = np.asarray([0, 2, 1])
        pool_list = np.asarray([0, 1, 2, 3])
        expected = 0.034107
        actual = alt.compute(rec_list, pool_list)
        self.assertTrue(abs(expected - actual) < 0.001)

        # expected = 0.14002872565616667
        expected = 0.140119
        alt2 = AlternativeVoices(item_minor_major=item_min_major, k=2)
        self.assertEqual(alt2.k, 2)
        actual = alt2.compute(rec_list, pool_list)
        self.assertTrue(abs(expected - actual) < 0.001)

        # expected = 0.16783069127015743
        expected = 0.167881
        alt3 = AlternativeVoices(
            item_minor_major=item_min_major, k=2, divergence_type="js")
        self.assertEqual(alt3.k, 2)
        actual = alt3.compute(rec_list, pool_list)
        self.assertTrue(abs(expected - actual) < 0.001)


if __name__ == "__main__":
    unittest.main()
