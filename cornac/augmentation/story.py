from datetime import timedelta
import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_story(df):
    """
    Enhance the dataframe with a new 'story' column by identifying stories based on text similarity within categories
    over time (mainly from
    https://github.com/svrijenhoek/RADio/blob/f1ce0d8bb0d7235f0c48b1745a8a81060683846a/dart/preprocess/identify_stories.py).

    Parameters
    ----------
    df: DataFrame
        Input dataframe with columns 'id', 'text', 'date', 'category' (either a str or a list of str).

    Returns
    -------
    df : DataFrame
        Extended dataframe with a new 'story' column.
    """

    required_columns = ['id', 'text', 'date']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"The input DataFrame must contain a '{column}' column. "
                             f"Please provide the {column} information in the input DataFrame.")

    original_df = df.copy()  # Keep the original DataFrame to merge later
    df = df.dropna(subset=['text']).reset_index(drop=True)  # Drop rows with missing 'text'
    df = df.dropna(subset=['date']).reset_index(drop=True)


    if df.empty or df['date'].isna().all():
        return original_df.assign(story=pd.NA)

    # Ensure category column is a list of strings
    df['category_list'] = df['category'].apply(lambda cat: cat if isinstance(cat, list) else [cat])

    unique_categories = set(cat for sublist in df['category_list'] for cat in sublist)
    vectorizer = TfidfVectorizer(stop_words='english')
    threshold = 0.5
    cosines = []

    df['date'] = pd.to_datetime(df['date'])
    first_date = df['date'].min()
    last_date = df['date'].max()
    delta = last_date - first_date

    for i in range(delta.days + 1):
        today = first_date + timedelta(days=i)
        yesterday = today - timedelta(days=1)
        past_three_days = today - timedelta(days=3)

        documents_3_days = df[(df['date'] >= past_three_days) & (df['date'] <= today)]
        documents_1_day = df[(df['date'] >= yesterday) & (df['date'] <= today)]

        for category in unique_categories:
            subset_3 = documents_3_days[documents_3_days['category_list'].apply(lambda cats: category in cats)]
            subset_1 = documents_1_day[documents_1_day['category_list'].apply(lambda cats: category in cats)]

            if not subset_1.empty and not subset_3.empty:
                # Combine the texts from both subsets for consistent vectorization
                combined_texts = subset_1['text'].tolist() + subset_3['text'].tolist()
                # print(f"")
                combined_matrix = vectorizer.fit_transform(combined_texts)

                

                # Separate the matrices for subset_1 and subset_3
                subset_1_matrix = combined_matrix[:len(subset_1)]
                subset_3_matrix = combined_matrix[len(subset_1):]

                # Calculate cosine similarities between the two matrices
                cosine_similarities = cosine_similarity(subset_1_matrix, subset_3_matrix)

                for idx_1, similarity_row in enumerate(cosine_similarities):
                    for idx_2, cosine in enumerate(similarity_row):
                        if threshold <= cosine < 1:
                            x_id = subset_1.index[idx_1]
                            y_id = subset_3.index[idx_2]
                            cosines.append({'x': x_id, 'y': y_id, 'cosine': cosine})

    if not cosines:
        cosines.append({'x': 0, 'y': 0, 'cosine': 1})

    cosine_df = pd.DataFrame(cosines).drop_duplicates()

    graph = nx.from_pandas_edgelist(cosine_df, 'x', 'y', edge_attr=True)

    # Create partitions, or stories
    partition = community_louvain.best_partition(graph)
    df['story'] = df.index.map(partition).fillna(0).astype(int)

    # Merge the original DataFrame with the new story column, preserving all original rows
    df_merged = pd.merge(original_df, df[['id', 'story']], on='id', how='left')

    # Set the story column for rows that don't match to pd.NA
    df_merged['story'] = df_merged['story'].astype('Int64')  # Use 'Int64' type to support pd.NA

    return df_merged
