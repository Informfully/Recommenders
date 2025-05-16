import unittest
import pandas as pd
from collections import OrderedDict
from cornac.models.drdw.rank_core import ItemRanker


class TestRankCore(unittest.TestCase):
    def setUp(self):
        self.article_df = pd.DataFrame({
            'category': ['Politics', 'Sports', 'Technology', 'Politics', 'Technology', 'Sports', 'Politics'],
            'outlet': ['Outlet A', 'Outlet B', 'Outlet A', 'Outlet A', 'Outlet B', 'Outlet A', 'Outlet A'],
            'sentiment': ['Positive', 'Negative', 'Neutral', 'Negative', 'Neutral', 'Positive', 'Positive']
        }, index=[0, 1, 2, 3, 4, 5, 6])
        self.candidate_items = [0, 1, 2, 3, 4, 5, 6]

    def test_init(self):
        test_dimension = 'category'
        # Initialize the rank_core class
        test_rank_core_instance = ItemRanker(
            self.candidate_items, self.article_df, test_dimension)
        self.assertEqual(test_rank_core_instance.V, len(self.candidate_items))
        self.assertIsInstance(test_rank_core_instance.color_dict, OrderedDict)
        self.assertEqual(
            list(test_rank_core_instance.color_dict.keys()), ['Politics',  'Sports', 'Technology'])
        self.assertEqual(test_rank_core_instance.used_color, {
                         'Politics': 0,  'Sports': 0, 'Technology': 0})

        # invalid input
        with self.assertRaises(TypeError):
            test_rank_core_instance = ItemRanker(
                None, self.article_df, test_dimension)
        with self.assertRaises(TypeError):
            test_rank_core_instance = ItemRanker(
                [0, 1, 2], self.article_df, [123])
        with self.assertRaises(ValueError):
            test_rank_core_instance = ItemRanker(
                [0, 1, 2], self.article_df, 'unkown_column')
        with self.assertRaises(IndexError):
            test_rank_core_instance = ItemRanker(
                [3, 4, 5, 6, 7, 8, 9], self.article_df, 'category')

        test_rank_core_instance = ItemRanker(
            [], self.article_df, 'category')
        # empty candidate items is allowed; however, the ranking result is also [].
        self.assertEqual(test_rank_core_instance.candidateItems, [])

    def test_buildAdjMatrix(self):
        test_rank_core_instance = ItemRanker(
            self.candidate_items, self.article_df, 'outlet')
        adj_matrix = test_rank_core_instance.buildAdjMatrix()
        expected_matrix = [
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
        ]
        self.assertEqual(adj_matrix, expected_matrix)

    def test_create_color_sequence(self):

        test_rank_core_instance = ItemRanker(
            self.candidate_items, self.article_df, 'category')
        color = [0, 1, 0, 1, 2, 0, 2]
        result = test_rank_core_instance.create_color_sequence(color)
        self.assertEqual(result, ['Politics', 'Sports', 'Politics',
                                  'Sports', 'Technology', 'Politics', 'Technology'])

    def test_is_valid_color(self):
        test_rank_core_instance = ItemRanker(
            self.candidate_items, self.article_df, 'outlet')
        self.assertEqual(list(test_rank_core_instance.used_color.keys()), [
                         'Outlet A', 'Outlet B'])
        graph = test_rank_core_instance.buildAdjMatrix()
        color = [-1, -1, -1, -1, -1, -1, -1]
        # check when all color are not decided yet, for vertex 0,is color 'Outlet A' ok.
        self.assertTrue(
            test_rank_core_instance.is_valid_color(0, graph, color, 0))
        # Suppose 'Outlet A' being fully used
        test_rank_core_instance.used_color['Outlet A'] = 5
        self.assertFalse(test_rank_core_instance.is_valid_color(
            0, graph, color, 0))  # Should fail since 'Outlet A' is used
        test_rank_core_instance.used_color['Outlet B'] = 1
        # 'Outlet B' in total has two
        self.assertTrue(test_rank_core_instance.is_valid_color(
            0, graph, color, 1))

    def test_graph_coloring(self):
        # for category feature,  there is a valid graph coloring result
        test_rank_core_instance = ItemRanker(
            self.candidate_items, self.article_df, 'category')
        graph = test_rank_core_instance.buildAdjMatrix()
        color = [-1] * test_rank_core_instance.V
        result = test_rank_core_instance.graph_coloring(
            graph, len(test_rank_core_instance.color_dict), color, 0)
        self.assertTrue(result)
        # for outlet feature,  there isn't a valid graph coloring result
        test_rank_core_instance = ItemRanker(
            self.candidate_items, self.article_df, 'outlet')
        graph = test_rank_core_instance.buildAdjMatrix()
        color = [-1] * test_rank_core_instance.V
        result = test_rank_core_instance.graph_coloring(
            graph, len(test_rank_core_instance.color_dict), color, 0)
        self.assertFalse(result)

    def test_solve_graph_coloring(self):
        test_rank_core_instance = ItemRanker(
            self.candidate_items, self.article_df, 'category')
        result = test_rank_core_instance.solve_graph_coloring()
        print("result:{}".format(result))
        expected_result = ['Politics', 'Sports', 'Politics',
                           'Sports', 'Technology', 'Politics', 'Technology']
        self.assertEqual(
            result, expected_result)

        # change candidate items order
        test_rank_core_instance = ItemRanker(
            [2, 0, 5, 4, 6, 3, 1], self.article_df, 'category')
        result = test_rank_core_instance.solve_graph_coloring()
        expected_result = ['Politics', 'Technology', 'Politics',
                           'Technology', 'Sports', 'Politics', 'Sports']
        self.assertEqual(
            result, expected_result)

    def test_round_robin_rank(self):
        # Test the round-robin ranking algorithm
        test_rank_core_instance = ItemRanker(
            self.candidate_items, self.article_df, 'outlet')
        # First-In-First-Out
        ranked_list = test_rank_core_instance.round_robin_rank()
        expected_list = [0, 1, 2, 4, 3, 5, 6]
        self.assertEqual(ranked_list, expected_list)
        new_candidate_order = [0, 6, 3, 4, 2, 1, 5]
        test_rank_core_instance = ItemRanker(
            new_candidate_order, self.article_df, 'outlet')
        # First-In-First-Out
        ranked_list = test_rank_core_instance.round_robin_rank()
        expected_list = [0, 4, 6, 1, 3, 2, 5]
        self.assertEqual(ranked_list, expected_list)

    def test_rank(self):

        test_rank_core_instance = ItemRanker(
            self.candidate_items, self.article_df, 'sentiment')
        # key dict is: 'Positive', 'Negative', 'Neutral'
        result = test_rank_core_instance.rank()
        self.assertEqual(result, [0, 1, 5, 3, 2, 6, 4])
        # change items order
        test_rank_core_instance = ItemRanker(
            [2, 0, 5, 4, 6, 3, 1], self.article_df, 'category')
        result = test_rank_core_instance.rank()
        self.assertEqual(
            result, [0, 2, 6, 4, 5, 3, 1])

        # round-robin case
        test_rank_core_instance = ItemRanker(
            self.candidate_items, self.article_df, 'outlet')
        # First-In-First-Out
        ranked_list = test_rank_core_instance.rank()
        expected_list = [0, 1, 2, 4, 3, 5, 6]
        self.assertEqual(ranked_list, expected_list)


if __name__ == '__main__':
    unittest.main()
