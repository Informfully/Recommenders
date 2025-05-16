import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import os


def merge_user_diversity_files(directory):
    """
    Merge multiple CSV files containing user diversity data into a single DataFrame.

    Parameters:
        directory (str): The directory path containing the CSV files.

    Returns:
        pandas DataFrame: Merged DataFrame containing data from all CSV files.
    """
    if not os.path.isdir(directory):
        raise ValueError("Invalid directory path.")

    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    if not file_paths:
        raise ValueError("No CSV files found in the directory.")

    dfs = [pd.read_csv(file) for file in file_paths]
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='User_ID', how='inner'), dfs)
    user_diversity_df = merged_df.dropna()

    return user_diversity_df


def plot_histogram(data, column, bins=10, color='skyblue', edgecolor='black', ax=None):
    """
    Plot a histogram for a specified column in a DataFrame.

    Parameters:
        data (pandas DataFrame): The DataFrame containing the data.
        column (str): The name of the column to plot.
        bins (int or array_like, optional): The number of bins to use. Default is 10.
        color (str or array_like, optional): The color of the bars. Default is 'skyblue'.
        edgecolor (str, optional): The color of the edges of the bars. Default is 'black'.
        ax (matplotlib axes, optional): Axes to plot on. If None, a new figure and axes will be created.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Input data must be a non-empty DataFrame.")

    if column not in data.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    title = f'Histogram of {column}'
    xlabel = column
    ylabel = 'Frequency'

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.hist(data[column], bins=bins, color=color, edgecolor=edgecolor)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    if ax is None:
        plt.show()


def plot_scatter_one(data, column, alpha=0.2, color='skyblue', ax=None):
    """
    Plot a scatter plot for a specified column in a DataFrame.

    Parameters:
        data (pandas DataFrame): The DataFrame containing the data.
        column (str): The name of the column to plot on the x-axis.
        alpha (float, optional): The transparency of the scatter points. Default is 0.2.
        color (str or array_like, optional): The color of the scatter points. Default is 'skyblue'.
        ax (matplotlib axes, optional): Axes to plot on. If None, a new figure and axes will be created.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.scatter(data[column], range(len(data[column])), alpha=alpha, color=color)

    title = f'Scatter Plot of {column}'
    xlabel = column
    ylabel = 'Index'
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def plot_scatterplot_matrix(df, hue=None):
    """
    Create a scatterplot matrix for a DataFrame.

    Parameters:
        df (pandas DataFrame): The DataFrame containing the data.
        hue (str, optional): The name of the column in the DataFrame to map plot aspects to different colors.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input data must be a non-empty DataFrame.")

    if hue is not None and hue not in df.columns:
        raise ValueError(f"The specified hue column '{hue}' does not exist in the DataFrame.")

    sns.pairplot(df, hue=hue)


def plot_correlation_heatmap(df, selected_columns=None, title='', figsize=(10, 8), cmap='coolwarm', annot=True, fmt=".2f"):
    """
    Create a correlation heatmap for selected columns of a DataFrame.

    Parameters:
        df (pandas DataFrame): The DataFrame containing the data.
        selected_columns (list of str, optional): The names of the columns of interest. If None, use all columns.
        title (str, optional): The title of the heatmap.
        figsize (tuple, optional): The size of the figure (width, height). Default is (10, 8).
        cmap (str or colormap, optional): The colormap to use for the heatmap. Default is 'coolwarm'.
        annot (bool, optional): Whether to annotate the heatmap with correlation values. Default is True.
        fmt (str, optional): String formatting code to use when annotating the heatmap. Default is ".2f".
    """

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input data must be a non-empty DataFrame.")

    if selected_columns is None:
        selected_columns = df.columns.tolist()
    else:
        missing_columns = [col for col in selected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following columns are not found in the DataFrame: {', '.join(missing_columns)}")

    correlation_matrix = df[selected_columns].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt=fmt, square=True)
    plt.title(title)
    plt.show()


def calculate_correlation(data, column1, column2):
    """
    Calculate the correlation coefficient between two columns in a DataFrame.

    Parameters:
        data (array-like): The data containing the columns.
        column1 (str): The name of the column of interest.
        column2 (str): The name of the column of interest.

    Returns:
        float: The correlation coefficient between the two columns.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        data = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    if isinstance(column1, str) and column1 not in data.columns:
        raise ValueError(f"Column '{column1}' is not found in the data.")
    if isinstance(column2, str) and column2 not in data.columns:
        raise ValueError(f"Column '{column2}' is not found in the data.")

    column1_data = data[column1]
    column2_data = data[column2]

    correlation = pd.Series(column1_data).corr(pd.Series(column2_data))
    return correlation


def plot_scatter_with_regression(data, x_data, y_data, figsize=(8, 6), title='', x_label='', y_label='', alpha=0.5, regression_color='red', regression_linewidth=2):
    """
    Create a scatter plot with regression from the provided data.

    Parameters:
        data (pandas DataFrame, numpy array, or list): The data containing the x and y columns.
        x_data (str, int, or array-like): The data for the x-axis.
        y_data (str, int, or array-like): The data for the y-axis.
        figsize (tuple, optional): The size of the figure (width, height). Default is (8, 6).
        title (str): Title of the plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        alpha (float): The transparency of the scatter points. Default is 0.5.
        regression_color (str): Color of the regression line. Default is 'red'.
        regression_linewidth (float): Width of the regression line. Default is 2.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        data = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    if isinstance(x_data, str) and x_data not in data.columns:
        raise ValueError(f"Column '{x_data}' is not found in the data.")
    if isinstance(y_data, str) and y_data not in data.columns:
        raise ValueError(f"Column '{y_data}' is not found in the data.")

    x = data[x_data] if isinstance(x_data, str) else x_data
    y = data[y_data] if isinstance(y_data, str) else y_data

    plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y, alpha=alpha)

    # Fit non-parametric regression line
    smooth = sm.nonparametric.lowess(y, x)
    plt.plot(smooth[:, 0], smooth[:, 1], color=regression_color, linewidth=regression_linewidth)

    # Set title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


def scale_data(data, columns=None, scaler=None):
    """
    Preprocess the data for clustering by extracting specified columns and scaling the data.

    Parameters:
        data (pandas DataFrame): The original DataFrame containing the data.
        columns (list or None): A list of column names to be extracted for clustering.
                                 If None, all columns will be used.
        scaler (scikit-learn scaler or None): Scaler object to scale the data.
                                              If None, StandardScaler will be used.

    Returns:
        scaled_data_df (DataFrame): Scaled data for clustering.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        data = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    if columns is None:
        columns = data.columns.tolist()
    data_subset = data[columns]

    if scaler is None:
        scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data_subset)
    scaled_data_df = pd.DataFrame(scaled_data, columns=columns)

    return scaled_data_df


def plot_cumulative_variance_ratio(scaled_data, ax=None):
    """
    Plot the cumulative explained variance ratio for PCA.

    Parameters:
        scaled_data (array-like): Scaled data for PCA.
        ax (matplotlib axis, optional): Axis to plot on. If None, a new figure and axis will be created.
    """
    if isinstance(scaled_data, pd.DataFrame):
        if scaled_data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(scaled_data, (np.ndarray, list)):
        if len(scaled_data) == 0:
            raise ValueError("Input data is empty.")
        scaled_data = pd.DataFrame(scaled_data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    pca = PCA().fit(scaled_data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)

    ax.plot(range(1, pca.n_components_ + 1), cumulative_var_ratio, marker='o', linestyle='-')
    ax.set_title('Cumulative Explained Variance Ratio')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance Ratio')
    if not ax:
        plt.show()


def plot_scree_plot(scaled_data, ax=None):
    """
    Plot the scree plot for PCA.

    Parameters:
        scaled_data (array-like): Scaled data for PCA.
        ax (matplotlib axis): Axis to plot on. If None, a new figure and axis will be created.

    Returns:
        None
    """
    if isinstance(scaled_data, pd.DataFrame):
        if scaled_data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(scaled_data, (np.ndarray, list)):
        if len(scaled_data) == 0:
            raise ValueError("Input data is empty.")
        scaled_data = pd.DataFrame(scaled_data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    pca = PCA().fit(scaled_data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    ax.plot(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, marker='o', linestyle='-')
    ax.set_title('Scree Plot')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Explained Variance Ratio')
    if not ax:
        plt.show()


def apply_pca(scaled_data, n_components=4, column_names=None):
    """
    Apply Principal Component Analysis (PCA) to the scaled data and create a DataFrame for the transformed data.

    Parameters:
        scaled_data (array-like)): Scaled data for PCA.
        n_components (int): Number of principal components to retain. Default is 4.
        column_names (list of str, optional): Column names for the DataFrame. If None, default names ['PC1', 'PC2', ...] will be used.

    Returns:
        pca_df (pandas DataFrame): DataFrame containing the PCA-transformed data.
        loadings_df (pandas DataFrame): DataFrame containing the loadings.
    """
    if isinstance(scaled_data, pd.DataFrame):
        if scaled_data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(scaled_data, (np.ndarray, list)):
        if len(scaled_data) == 0:
            raise ValueError("Input data is empty.")
        scaled_data = pd.DataFrame(scaled_data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer.")

    num_features = scaled_data.shape[1]
    if n_components > num_features:
        raise ValueError(f"n_components cannot be greater than the number of features ({num_features}).")

    if column_names is None:
        column_names = [f'PC{i+1}' for i in range(n_components)]
    elif len(column_names) != n_components:
        raise ValueError("The length of column_names must match n_components.")

    pca = PCA(n_components=n_components)

    pca_data = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=pca_data, columns=column_names)

    loadings = pca.components_
    loadings_df = pd.DataFrame(loadings, columns=scaled_data.columns, index=column_names)

    return pca_df, loadings_df


def plot_dendrogram(data, method='complete', metric='euclidean', ax=None):
    """
    Create a dendrogram for hierarchical clustering.

    Parameters:
        data (array-like): The data to be clustered.
        method (str, optional): The linkage method to use. Default is 'complete'.
        metric (str, optional): The distance metric to use. Default is 'euclidean'.
        ax (matplotlib Axes, optional): The axes on which to plot the dendrogram.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        data = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    Z = linkage(data, method=method, metric=metric)

    if ax is None:
        plt.figure(figsize=(10, 6))
        dendrogram(Z)
        plt.title(f'Cluster Dendrogram (Method: {method.capitalize()}, Metric: {metric.capitalize()})')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()
    else:
        dendrogram(Z, ax=ax)
        ax.set_title(f'Cluster Dendrogram (Method: {method.capitalize()}, Metric: {metric.capitalize()})')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Distance')


def plot_cluster_dendrograms(data, methods=('ward', 'complete', 'average'), metrics=('euclidean', 'cityblock', 'cosine'), figsize=(25, 8), main_title=None):
    """
    Plot multiple cluster dendrograms in subplots.

    Parameters:
        data (array-like): The data to be clustered.
        methods (tuple): A string or tuple of linkage methods to use. Default is ('ward', 'complete', 'average').
        metrics (str or tuple): A string or tuple of distance metrics to use. Default is ('euclidean', 'cityblock', 'cosine').
        figsize (tuple, optional): The size of the figure. Default is (25, 8).
        main_title (str, optional): The main title for the plot. If not provided, no title will be set.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        data = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    if isinstance(metrics, str):
        metrics = (metrics,)

    if isinstance(methods, str):
        methods = (methods,)

    fig, axes = plt.subplots(nrows=len(metrics), ncols=len(methods), figsize=figsize)

    for i, metric in enumerate(metrics):
        for j, method in enumerate(methods):
            if method == 'ward' and metric != 'euclidean':
                axes[i, j].set_visible(False)
                axes[i, j].axis('off')
                continue  # Skip this combination
            plot_dendrogram(data, method=method, metric=metric, ax=axes[i, j] if len(metrics) > 1 else axes[j])

    # Add a single title for the entire plot
    if main_title:
        fig.suptitle(main_title, fontsize=16)
    plt.tight_layout()
    plt.show()


def apply_agglomerative_clustering(data, n_clusters, linkage='ward', metric='euclidean'):
    """
    Apply Agglomerative Clustering to the data.

    Parameters:
        data (array-like): The data to be clustered.
        n_clusters (int): The number of clusters to form.
        linkage (str, optional): The linkage method to use. Default is 'ward'.
            Possible values: 'ward', 'complete', 'average', 'single'.
        metric (str, optional): The distance metric to use. Default is 'euclidean'.
            Possible values: 'euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed'.

    Returns:
        array-like: Cluster labels assigned to each data point.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        data = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    if len(data) < n_clusters:
        raise ValueError("Number of rows in data must be greater than or equal to the number of clusters.")

    if not isinstance(n_clusters, int) or n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")

    if linkage == 'ward' and metric != 'euclidean':
        raise ValueError("When linkage is 'ward', metric must be 'euclidean'.")

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)

    clusters = model.fit_predict(data)

    return clusters


def profile_clusters(data, clusters):
    """
    Profile each cluster by providing summary statistics.

    Parameters:
        data (array-like): The data used for clustering.
        clusters (array-like): Cluster labels assigned to each data point.

    Returns:
        dict: Dictionary containing summary statistics for each cluster.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
        cluster_data = data
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        cluster_data = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    if len(data) != len(clusters):
        raise ValueError("Length of 'data' and 'clusters' must be the same.")

    cluster_profiles = {}

    # Iterate over each cluster
    for i in np.unique(clusters):
        cluster_data_i = cluster_data[clusters == i]
        cluster_profile = pd.DataFrame(cluster_data_i).describe()
        cluster_profiles[f'Cluster {i} profile'] = cluster_profile

    return cluster_profiles


def plot_silhouette_plot(data, clusters, title=None, ax=None):
    """
    Create a silhouette plot to evaluate cluster quality.

    Parameters:
        data (array-like): The data used for clustering.
        clusters (array-like): The cluster labels assigned to each data point.
        title (str, optional): The title for the plot. If not provided, no title will be set.
        ax (matplotlib.axes.Axes, optional): The subplot to plot on. If not provided, a new plot will be created.
    """
    if not isinstance(data, (np.ndarray, pd.DataFrame, list)):
        raise ValueError("Input data must be a numpy array, pandas DataFrame, or list.")

    if not isinstance(clusters, (np.ndarray, list)):
        raise ValueError("Input clusters must be a numpy array or list.")

    data = np.array(data)
    clusters = np.array(clusters)

    if len(data) == 0 or len(clusters) == 0:
        raise ValueError("Input data and clusters must be non-empty.")

    if len(data) != len(clusters):
        raise ValueError("Input data and clusters must have the same length.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Compute silhouette scores
    silhouette_avg = silhouette_score(data, clusters)
    sample_silhouette_values = silhouette_samples(data, clusters)

    # Plot silhouette plot
    y_lower = 10
    for i in np.unique(clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(i) / len(np.unique(clusters)))
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Silhouette plot")

    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.text(silhouette_avg + 0.01, 0, f'Average Silhouette Score: {silhouette_avg:.5f}', color="red")
    ax.set_yticks([])  # Clear the yaxis labels / ticks

    if ax is None:
        plt.show()


def apply_tsne(data, n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=None):
    """
    Apply t-distributed Stochastic Neighbor Embedding (t-SNE) to the data.

    Parameters:
        data (array-like): The input data to be embedded.
        n_components (int, optional): The dimension of the embedded space. Default is 2.
        perplexity (float, optional): The perplexity parameter. Default is 30.
        learning_rate (float, optional): The learning rate. Default is 200.
        n_iter (int, optional): Maximum number of iterations for optimization. Default is 1000.
        random_state (int or RandomState, optional): Random seed for reproducibility. Default is None.

    Returns:
        tsne_df (pandas DataFrame): The embedded data.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        data = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer.")

    if not isinstance(perplexity, (int, float)) or perplexity <= 0:
        raise ValueError("perplexity must be a positive number.")

    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("learning_rate must be a positive number.")

    if not isinstance(n_iter, int) or n_iter <= 0:
        raise ValueError("n_iter must be a positive integer.")

    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter,
                random_state=random_state)

    tsne_data = tsne.fit_transform(data)

    tsne_df = pd.DataFrame(tsne_data, columns=[f'Component {i}' for i in range(1, n_components + 1)])

    return tsne_df


def find_elbow_point(sorted_distances):
    """
    Find the elbow point using the method of finding the point farthest away from a line segment connecting
    the first and last points of the curve.

    Parameters:
        sorted_distances (array-like): Sorted array of distances.

    Returns:
        elbow_index (int): Index of the elbow point.
        elbow_distance (float): Distance at the elbow point.
    """
    if not isinstance(sorted_distances, (np.ndarray, list)):
        raise ValueError("Input must be a numpy array or a list.")

    if len(sorted_distances) < 2:
        raise ValueError("Input array must contain at least two elements.")

    # Define the line segment A connecting the first and last points of the curve
    line_segment_A = [sorted_distances[0], sorted_distances[-1]]

    # Calculate the distance of each point to line segment A and find the maximum distance
    max_distance = 0
    elbow_index = 0
    for i, distance in enumerate(sorted_distances):
        # Calculate the perpendicular distance of point i to line segment A
        # Compute the numerator
        numerator = np.abs((len(sorted_distances) - 1) * (line_segment_A[0] - distance) - (0 - i) * (line_segment_A[1] - line_segment_A[0]))
        # Compute the denominator
        denominator = np.sqrt((len(sorted_distances) - 1) ** 2 + (line_segment_A[1] - line_segment_A[0]) ** 2)
        # Compute the perpendicular distance
        perpendicular_distance = numerator / denominator
        if perpendicular_distance > max_distance:
            max_distance = perpendicular_distance
            elbow_index = i
    elbow_distance = sorted_distances[elbow_index]

    return elbow_index, elbow_distance


def kmeans_optimal_clusters(data, max_clusters=15, title=None, ax=None):
    """
    Plot the Elbow Method to determine the optimal number of clusters using KMeans.

    Parameters:
        data (array-like): The data for clustering.
        max_clusters (int): The maximum number of clusters to consider. Default is 15.
        title (str, optional): Title for the plot.
        ax (matplotlib axes, optional): Axes to plot on. If None, a new figure and axes will be created.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        data = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    wcss = []

    # Calculate WCSS for each number of clusters
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)  # Inertia is another name for WCSS

    # Plot WCSS vs. Number of Clusters
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
    if title is None:
        ax.set_title('Elbow Method')
    else:
        ax.set_title(title)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax.set_xticks(range(1, max_clusters + 1))
    ax.grid(True)

    # Call find_elbow_point to get elbow index and distance
    elbow_index, elbow_distance = find_elbow_point(wcss)

    # Draw a horizontal line at the elbow distance
    ax.axhline(y=elbow_distance, color='red', linestyle='--', label=f'Elbow Point: {elbow_index + 1} clusters')
    ax.legend()


def apply_kmeans_clustering(data, n_clusters=5, random_state=15, column_names=None):
    """
    Perform K-means clustering on the given data.

    Parameters:
        data (DataFrame): The data to be clustered.
        n_clusters (int, optional): The number of clusters to form. Default is 5.
        random_state (int, optional): Determines random number generation for centroid initialization.
                                      Default is 15.
        column_names (list of str, optional): Column names use for clustering. Default is None.

    Returns:
        numpy array: An array of cluster labels for each data point.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Input data must be a non-empty DataFrame.")

    if len(data) < n_clusters:
        raise ValueError("Number of rows in data must be greater than or equal to the number of clusters.")

    if column_names is not None:
        if isinstance(column_names, str):
            if column_names not in data.columns:
                raise ValueError(f"Column '{column_names}' is not found in the data.")
            else:
                column_names = [column_names]
                column_data = data[column_names]
        else:
            for c in column_names:
                if c not in data.columns:
                    raise ValueError(f"Column '{c}' is not found in the data.")
            column_data = data[column_names]

        if len(column_names) == 1:
            column_data = column_data.values.reshape(-1, 1)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(column_data)
    else:
        # Perform K-means clustering using all features
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(data)

    return clusters


def plot_bic(data, max_components=10, covariance_type='full', random_state=None, ax=None, title='BIC vs. Number of Components'):
    """
    Plot the Bayesian Information Criterion (BIC) values for different numbers of clusters.

    Parameters:
        data (array-like): The data to fit the model.
        max_components (int, optional): The maximum number of components to consider. Default is 10.
        covariance_type (str, optional): Type of covariance parameters to use.
            Must be one of {'full', 'tied', 'diag', 'spherical'}. Default is 'full'.
        random_state (int, RandomState instance or None, optional): Controls the random number generation.
            Pass an int for reproducible results. Default is None.
        ax (matplotlib.axes.Axes, optional): The axes where the plot will be drawn. Default is None.
        title (str, optional): Title of the plot. Default is 'BIC vs. Number of Components'.

    Returns:
        None (plots the BIC values)
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        data = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    bic_values = []

    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
        gmm.fit(data)
        bic_values.append(gmm.bic(data))

    elbow_index, elbow_distance = find_elbow_point(bic_values)

    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

    ax.plot(range(1, max_components + 1), bic_values, marker='o', linestyle='-')
    ax.set_title(title)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('BIC Value')
    ax.set_xticks(range(1, max_components + 1))
    ax.grid(True)
    ax.axvline(x=elbow_index + 1, color='red', linestyle='--', label=f'Elbow Point: {elbow_index + 1} clusters')
    ax.legend()


def apply_gmm(data, n_components=1, covariance_type='full', random_state=None):
    """
    Apply Gaussian Mixture Models (GMM) to data.

    Parameters:
        data (array-like): The data to fit the model.
        n_components (int, optional): The number of mixture components. Default is 1.
        covariance_type (str, optional): Type of covariance parameters to use.
            Must be one of {'full', 'tied', 'diag', 'spherical'}. Default is 'full'.
        random_state (int, RandomState instance or None, optional): Controls the random number generation.
            Pass an int for reproducible results. Default is None.

    Returns:
        array-like: The cluster labels assigned to each data point.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        data = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    clusters = gmm.fit_predict(data)
    return clusters


def plot_knn_distance(data, k=5, ax=None, title='k-NN Distances'):
    """
    Calculate the k-nearest neighbors distance for each point, sort them in increasing order, and plot them.

    Parameters:
        data (array-like): The data for which to calculate k-NN distances.
        k (int): The number of nearest neighbors to consider. Default is 5.
        ax (matplotlib.axes.Axes, optional): The axes where the plot will be drawn. Default is None.
        title (str, optional): Title of the plot. Default is 'k-NN Distances'.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        data = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(data)
    distances, indices = nn_model.kneighbors(data)
    knn_distances = np.mean(distances, axis=1)
    sorted_distances = np.sort(knn_distances)

    elbow_index, elbow_distance = find_elbow_point(sorted_distances)

    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

    # Plot the k-NN distances
    ax.plot(range(len(sorted_distances)), sorted_distances, marker='o', linestyle='-')
    ax.set_title(title)
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel(f'{k}-NN Distance')
    ax.grid(True)

    # Add a horizontal line at the elbow point
    ax.axhline(y=elbow_distance, color='r', linestyle='--', label='Elbow Point')

    # Plot the intersection point
    ax.scatter(elbow_index, elbow_distance, color='k', label='Intersection', zorder=3)
    ax.legend()

    # Annotate the intersection point
    ax.annotate(f'({elbow_index}, {elbow_distance:.2f})', xy=(elbow_index, elbow_distance),
                xytext=(elbow_index + 8000, elbow_distance + 2),
                arrowprops=dict(facecolor='black', shrink=0.05))


def apply_dbscan(data, eps=0.5, min_samples=5):
    """
    Apply DBSCAN clustering to the given data.

    Parameters:
        data (array-like): The data to be clustered.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
        clusters (array): The cluster labels assigned to each data point.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
    elif isinstance(data, (np.ndarray, list)):
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        data = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame, numpy array, or list.")

    # Initialize DBSCAN clustering model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the model to the data and obtain cluster labels
    clusters = dbscan.fit_predict(data)

    return clusters


def count_data_points_in_clusters(clusters):
    """
    Count data points in each cluster.

    Parameters:
        clusters (array-like): The cluster labels assigned to each data point.

    Returns:
        dict: A dictionary where keys are cluster numbers and values are the count of data points in each cluster.
    """
    if not isinstance(clusters, (np.ndarray, list)):
        raise ValueError("Input clusters should be a numpy array or a list.")

    clusters = np.asarray(clusters)
    cluster_counts = {cluster_num: sum(clusters == cluster_num) for cluster_num in set(clusters)}
    return cluster_counts


def visualize_clusters(data, cluster_labels, method='PCA', title=None, ax=None):
    """
    Visualizes clusters using dimensionality reduction.

    Parameters:
    - data (array-like): The original data.
    - cluster_labels (array-like): Cluster labels assigned to each data point.
    - method (str): The dimensionality reduction method. Can be 'PCA' (default), 'KernelPCA', or 't-SNE'.
    - title (str, optional): Title for the plot.
    - ax (matplotlib Axes, optional): The Axes object on which to plot the clusters. If not provided, a new figure will be created.
    Returns:
    - None (displays a plot)
    """

    if not isinstance(data, (np.ndarray, pd.DataFrame)):
        raise ValueError("Input data must be a numpy array or pandas DataFrame.")

    if isinstance(cluster_labels, (list, np.ndarray)):
        cluster_labels = np.array(cluster_labels)
    else:
        raise ValueError("Cluster labels must be a list or numpy array.")

    if len(data) != len(cluster_labels):
        raise ValueError("The length of data and cluster_labels must be the same.")

    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 'KernelPCA':
        reducer = KernelPCA(n_components=2, kernel='rbf')  # You can specify other kernels if needed
    elif method == 't-SNE':
        reducer = TSNE(n_components=2)
    else:
        raise ValueError("Invalid method. Choose 'PCA', 'KernelPCA', or 't-SNE'.")

    reduced_data = reducer.fit_transform(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 6))
    for cluster_num in range(len(np.unique(cluster_labels))):
        ax.scatter(reduced_data[cluster_labels == cluster_num, 0],
                   reduced_data[cluster_labels == cluster_num, 1],
                   label=f'Cluster {cluster_num}')
    if title is None:
        ax.set_title(f'Clustering Visualization using {method}')
    else:
        ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend()
    ax.grid(True)
    if ax is None:
        plt.show()