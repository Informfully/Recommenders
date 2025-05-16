import unittest
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from cornac.utils.correlation import merge_user_diversity_files
from cornac.utils.correlation import calculate_correlation
from cornac.utils.correlation import scale_data
from cornac.utils.correlation import apply_pca
from cornac.utils.correlation import apply_agglomerative_clustering
from cornac.utils.correlation import profile_clusters
from cornac.utils.correlation import apply_tsne
from cornac.utils.correlation import find_elbow_point
from cornac.utils.correlation import apply_kmeans_clustering
from cornac.utils.correlation import apply_gmm
from cornac.utils.correlation import apply_dbscan
from cornac.utils.correlation import count_data_points_in_clusters


class TestMergeUserDiversityFiles(unittest.TestCase):
    def test_merge_user_diversity_files(self):
        test_directory = 'test_data'
        os.makedirs(test_directory, exist_ok=True)
        file1_path = os.path.join(test_directory, 'file1.csv')
        file2_path = os.path.join(test_directory, 'file2.csv')
        file3_path = os.path.join(test_directory, 'file3.csv')

        data1 = {'User_ID': [1, 2, 3], 'Feature1': [10, 20, 30]}
        data2 = {'User_ID': [1, 2, 3], 'Feature2': [40, 50, 60]}
        data3 = {'User_ID': [1, 2, 3], 'Feature3': [70, 80, 90]}

        pd.DataFrame(data1).to_csv(file1_path, index=False)
        pd.DataFrame(data2).to_csv(file2_path, index=False)
        pd.DataFrame(data3).to_csv(file3_path, index=False)

        merged_df = merge_user_diversity_files(test_directory)

        for file in os.listdir(test_directory):
            file_path = os.path.join(test_directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(test_directory)

        expected_columns = ['User_ID', 'Feature1', 'Feature2', 'Feature3']
        self.assertListEqual(list(merged_df.columns), expected_columns)

        expected_num_rows = 3
        self.assertEqual(len(merged_df), expected_num_rows)

        expected_data = {
            'User_ID': [1, 2, 3],
            'Feature1': [10, 20, 30],
            'Feature2': [40, 50, 60],
            'Feature3': [70, 80, 90]
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(merged_df, expected_df)

    def test_calculate_correlation(self):
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [4, 3, 2, 1],
            'C': [1, 3, 2, 4]
        })

        correlation = calculate_correlation(self.df, 'A', 'B')
        self.assertAlmostEqual(correlation, -1.0)

        correlation = calculate_correlation(self.df, 'A', 'C')
        self.assertAlmostEqual(correlation, 0.7999999999999999)

        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            calculate_correlation(empty_df, 'A', 'B')

        with self.assertRaises(ValueError):
            calculate_correlation(self.df, 'A', 'D')
        with self.assertRaises(ValueError):
            calculate_correlation(self.df, 'E', 'B')

    def test_scale_data(self):
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [2, 3, 4, 5, 6]
        })

        with self.assertRaises(ValueError):
            scale_data(pd.DataFrame())
        # Invalid input, not a DataFrame
        with self.assertRaises(ValueError):
            scale_data([])

        result = scale_data(self.data)
        expected = StandardScaler().fit_transform(self.data)
        np.testing.assert_array_almost_equal(result.values, expected)

        result = scale_data(self.data, columns=['A', 'B'])
        expected = StandardScaler().fit_transform(self.data[['A', 'B']])
        np.testing.assert_array_almost_equal(result.values, expected)
        self.assertListEqual(result.columns.tolist(), ['A', 'B'])

        scaler = MinMaxScaler()
        result = scale_data(self.data, scaler=scaler)
        expected = scaler.fit_transform(self.data)
        np.testing.assert_array_almost_equal(result.values, expected)

        scaler = MinMaxScaler()
        result = scale_data(self.data, columns=['A', 'B'], scaler=scaler)
        expected = scaler.fit_transform(self.data[['A', 'B']])
        np.testing.assert_array_almost_equal(result.values, expected)
        self.assertListEqual(result.columns.tolist(), ['A', 'B'])

    def test_apply_pca(self):
        self.data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
            'feature4': np.random.rand(100)
        })
        self.scaler = StandardScaler()
        self.scaled_data = pd.DataFrame(self.scaler.fit_transform(self.data), columns=self.data.columns)

        pca_df, loadings_df = apply_pca(self.scaled_data, n_components=3)
        self.assertEqual(pca_df.shape[1], 3)
        self.assertEqual(len(pca_df), len(self.scaled_data))
        self.assertEqual(loadings_df.shape, (3, self.scaled_data.shape[1]))
        self.assertListEqual(pca_df.columns.tolist(), ['PC1', 'PC2', 'PC3'])

        custom_columns = ['Comp1', 'Comp2']
        pca_df, _ = apply_pca(self.scaled_data, n_components=2, column_names=custom_columns)
        self.assertListEqual(pca_df.columns.tolist(), custom_columns)

        with self.assertRaises(ValueError):
            apply_pca(self.scaled_data, n_components=-1)

        with self.assertRaises(ValueError):
            apply_pca(self.scaled_data, n_components=10)

        with self.assertRaises(ValueError):
            apply_pca(self.scaled_data, n_components=3, column_names=['PC1'])

        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            apply_pca(empty_df, n_components=2)

    def test_apply_agglomerative_clustering(self):
        np.random.seed(0)
        self.data = np.random.rand(100, 2)
        self.n_clusters = 3

        with self.assertRaises(ValueError):
            apply_agglomerative_clustering(self.data, -1)

        with self.assertRaises(ValueError):
            apply_agglomerative_clustering(self.data, self.n_clusters, linkage='ward', metric='manhattan')

        clusters = apply_agglomerative_clustering(self.data, self.n_clusters)
        self.assertEqual(len(clusters), len(self.data))

        with self.assertRaises(ValueError):
            apply_agglomerative_clustering(None, self.n_clusters)

        clusters = apply_agglomerative_clustering(self.data, self.n_clusters, linkage='average')
        self.assertEqual(len(clusters), len(self.data))

    def test_profile_clusters(self):
        self.data = np.random.rand(100, 3)  # Random data with 3 features
        self.clusters = np.random.randint(0, 3, size=100)  # Random cluster labels

        result = profile_clusters(self.data, self.clusters)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(np.unique(self.clusters)))

        for key, value in result.items():
            self.assertTrue(key.startswith('Cluster '))
            self.assertIsInstance(value, pd.DataFrame)
            self.assertEqual(value.shape[1], self.data.shape[1])  # Number of columns should match data's features

        with self.assertRaises(ValueError):
            profile_clusters(self.data, self.clusters[:-1])

    def test_apply_tsne(self):
        self.data_array = np.random.rand(100, 4)
        self.data_df = pd.DataFrame(self.data_array, columns=['A', 'B', 'C', 'D'])
        self.data_list = self.data_array.tolist()

        tsne_df = apply_tsne(self.data_array, n_components=2)
        self.assertEqual(tsne_df.shape[1], 2)
        self.assertIsInstance(tsne_df, pd.DataFrame)

        tsne_df = apply_tsne(self.data_df, n_components=2)
        self.assertEqual(tsne_df.shape[1], 2)
        self.assertIsInstance(tsne_df, pd.DataFrame)

        tsne_df = apply_tsne(self.data_list, n_components=2)
        self.assertEqual(tsne_df.shape[1], 2)
        self.assertIsInstance(tsne_df, pd.DataFrame)

        with self.assertRaises(ValueError):
            apply_tsne("invalid_input", n_components=2)

        with self.assertRaises(ValueError):
            apply_tsne([], n_components=2)

        with self.assertRaises(ValueError):
            apply_tsne(self.data_array, n_components=0)

        with self.assertRaises(ValueError):
            apply_tsne(self.data_array, perplexity=-10)

        with self.assertRaises(ValueError):
            apply_tsne(self.data_array, learning_rate=0)

        with self.assertRaises(ValueError):
            apply_tsne(self.data_array, n_iter=-500)

    def test_find_elbow_point(self):
        distances = [1, 2, 3, 4, 8, 10, 12]
        elbow_index, elbow_distance = find_elbow_point(distances)
        self.assertEqual(elbow_index, 3)
        self.assertEqual(elbow_distance, 4)

        distances = [1, 2, 5, 6, 7, 9, 10]
        elbow_index, elbow_distance = find_elbow_point(distances)
        self.assertEqual(elbow_index, 2)
        self.assertEqual(elbow_distance, 5)

        distances = np.sort([3, 1, 4, 1, 5, 9, 2, 6])
        elbow_index, elbow_distance = find_elbow_point(distances)
        self.assertEqual(elbow_index, 6)
        self.assertEqual(elbow_distance, 6)

        with self.assertRaises(ValueError):
            find_elbow_point("not an array")

        with self.assertRaises(ValueError):
            find_elbow_point([1])

    def test_apply_kmeans_clustering(self):
        data = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            apply_kmeans_clustering(data, column_names='invalid')

        data = pd.DataFrame({'A': [1, 2, 3, 4]})
        with self.assertRaises(ValueError):
            apply_kmeans_clustering(data, column_names='A')

        data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        expected_clusters = np.array([1, 1, 0, 0])
        clusters = apply_kmeans_clustering(data,  n_clusters=2, column_names=['A', 'B'])
        np.testing.assert_array_equal(clusters, expected_clusters)

        data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        expected_clusters = np.array([1, 1, 0, 0])
        clusters = apply_kmeans_clustering(data, n_clusters=2)
        np.testing.assert_array_equal(clusters, expected_clusters)

    def test_apply_gmm(self):

        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [1.0, 2.0, 3.0, 4.0]
        })
        clusters = apply_gmm(data, n_components=2, random_state=0)
        self.assertEqual(len(clusters), len(data))

        data = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ])
        clusters = apply_gmm(data, n_components=2, random_state=0)
        self.assertEqual(len(clusters), len(data))

        data = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ]
        clusters = apply_gmm(data, n_components=2, random_state=0)
        self.assertEqual(len(clusters), len(data))

        data = pd.DataFrame()
        with self.assertRaises(ValueError):
            apply_gmm(data)

        data = np.array([])
        with self.assertRaises(ValueError):
            apply_gmm(data)

        data = "invalid data type"
        with self.assertRaises(ValueError):
            apply_gmm(data)

    def test_apply_dbscan(self):
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 2.1, 8.0, 8.1],
            'feature2': [1.0, 2.0, 2.1, 8.0, 8.1]
        })
        clusters = apply_dbscan(data, eps=1.0, min_samples=2)
        self.assertEqual(len(clusters), len(data))

        data = pd.DataFrame()
        with self.assertRaises(ValueError):
            apply_dbscan(data)

        data = np.array([])
        with self.assertRaises(ValueError):
            apply_dbscan(data)

        data = "invalid data type"
        with self.assertRaises(ValueError):
            apply_dbscan(data)

    def test_count_data_points_in_clusters(self):
        clusters = [0, 0, 0, 0]
        expected_output = {0: 4}
        self.assertEqual(count_data_points_in_clusters(clusters), expected_output)

        clusters = [0, 1, 0, 1, 1, 2]
        expected_output = {0: 2, 1: 3, 2: 1}
        self.assertEqual(count_data_points_in_clusters(clusters), expected_output)

        clusters = "invalid_input"
        with self.assertRaises(ValueError):
            count_data_points_in_clusters(clusters)

        clusters = [1] * 10000 + [2] * 5000 + [3] * 2500
        expected_output = {1: 10000, 2: 5000, 3: 2500}
        self.assertEqual(count_data_points_in_clusters(clusters), expected_output)

        clusters = []
        expected_output = {}
        self.assertEqual(count_data_points_in_clusters(clusters), expected_output)


if __name__ == '__main__':
    unittest.main()
