import unittest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from cornac.models.drdw.graph_recommender import GraphRec


class TestGraphRec(unittest.TestCase):

    def setUp(self):
        # Create a 3x3 user-item interaction matrix
        # Example matrix:
        # User 1 interacted with item 1 and item 3
        # User 2 interacted with item 2
        # User 3 interacted with item 1 and item 2
        data = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0]
        ], dtype=np.float32)
        self.train_matrix = csr_matrix(data)

        self.graph_rec = GraphRec(self.train_matrix)

    def test_adjacency_matrix_A(self):
        A = self.graph_rec.A.toarray()
        # Expected adjacency matrix structure:
        # | U x U block (0) | U x I block (train_matrix) |
        # | I x U block (train_matrix.T) | I x I block (0) |
        expected_A = np.array([
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0]
        ], dtype=np.float32)

        self.assertEqual(A.shape, (6, 6))
        np.testing.assert_almost_equal(A, expected_A, decimal=4)

    def test_initialization(self):
        self.assertEqual(self.graph_rec.num_u, 3)
        self.assertEqual(self.graph_rec.num_i, 3)

        P = self.graph_rec.P.toarray()  # Convert sparse matrix to dense

        # Expected 1-hop transition probability matrix
        expected_P = np.array([
            [0,   0,   0,  1/2, 0,  1/2],
            [0,   0,   0,   0, 1.0,  0],
            [0,   0,   0, 1/2, 1/2, 0],
            [1/2, 0, 1/2, 0, 0, 0],
            [0, 1/2, 1/2, 0, 0, 0],
            [1, 0, 0, 0, 0, 0]
        ], dtype=np.float32)

        np.testing.assert_almost_equal(P, expected_P, decimal=4)

        # Check the degree matrix `D`
        expected_D = np.array([2, 1, 2, 2, 2, 1], dtype=np.float32)
        np.testing.assert_almost_equal(self.graph_rec.D, expected_D, decimal=4)

    def test_initial_hop(self):
        """Test that the initial hop (1-hop, 2-hop, 3-hop) probabilities are correctly computed."""
        self.graph_rec.performInitialHop()
        P3 = self.graph_rec.P_multi[3].toarray()
        self.assertEqual(P3.shape, (6, 6))

        # 3 hop
        expected_P3 = np.array([[0., 0., 0., 0.5, 0.125, 0.375],
                                [0., 0., 0., 0.25, 0.75, 0.],
                                [0., 0., 0., 0.375, 0.5, 0.125],
                                [0.5, 0.125, 0.375, 0., 0., 0.],
                                [0.125, 0.375, 0.5, 0., 0., 0.],
                                [0.75, 0., 0.25, 0., 0., 0.]], dtype=np.float32)
        np.testing.assert_almost_equal(P3, expected_P3, decimal=4)

    def test_multihop(self):
        """Test that the multi-hop (arbitrary m-hop) probabilities are correctly computed."""
        # Perform up to 5-hop transition probabilities
        P5 = self.graph_rec.performMultiHop(5).toarray()

        self.assertEqual(P5.shape, (6, 6))

        expected_p5 = np.array([[0., 0., 0., 0.46875, 0.21875, 0.3125],
                                [0., 0., 0., 0.3125, 0.625, 0.0625],
                                [0., 0., 0., 0.375, 0.46875, 0.15625],
                                [0.46875, 0.15625, 0.375, 0., 0., 0.],
                                [0.21875, 0.3125, 0.46875, 0., 0., 0.],
                                [0.625, 0.0625, 0.3125, 0., 0., 0.]],
                               dtype=np.float32)
        np.testing.assert_almost_equal(P5, expected_p5,  decimal=4)

    def test_skip_hop(self):
        """Test skipping hops that have already been calculated."""
        # Perform 3-hop first
        self.graph_rec.performInitialHop()

        # Now call performMultiHop(3) again, which should return the cached result
        P3 = self.graph_rec.performMultiHop(3).toarray()

        np.testing.assert_equal(P3, self.graph_rec.P_multi[3].toarray())

    def test_edge_case_empty_matrix(self):
        """Test behavior with an empty interaction matrix."""
        empty_matrix = csr_matrix((0, 0), dtype=np.float32)
        graph_rec_empty = GraphRec(empty_matrix)

        # Check that the adjacency matrix shape is (0, 0)
        self.assertEqual(graph_rec_empty.A.shape, (0, 0))

        self.assertEqual(graph_rec_empty.D.shape, (0,))

        graph_rec_empty.performInitialHop()

    def test_edge_case_zero_interactions(self):
        """Test behavior with a matrix with no interactions (all zeros)."""
        zero_matrix = csr_matrix((3, 3), dtype=np.float32)
        graph_rec_zero = GraphRec(zero_matrix)
        self.assertEqual(graph_rec_zero.A.shape, (6, 6))
        expected_zero_result = np.array([[0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0.]],
                                        dtype=np.float32)

        np.testing.assert_equal(
            graph_rec_zero.A.toarray(), expected_zero_result)

        np.testing.assert_equal(
            graph_rec_zero.P.toarray(), expected_zero_result)
        # Degree matrix should have very small non-zero values to avoid division by zero
        self.assertTrue(np.all(graph_rec_zero.D > 0))

        graph_rec_zero.performInitialHop()


if __name__ == '__main__':
    unittest.main()
