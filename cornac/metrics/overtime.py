import pandas as pd
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
import glob

from cornac.datasets import mind as mind
from cornac.metrics.user import UserActivation
from cornac.metrics.user import UserGiniCoeff
from cornac.metrics.user import create_score_df
from cornac.metrics.user import save_dataframe_to_csv
from cornac.metrics.user import UserAlternativeVoices
from cornac.metrics.user import UserRepresentation
from cornac.metrics.user import UserCalibration
from cornac.metrics.user import UserFragmentation
from cornac.metrics.user import UserILD

from cornac.metrics.diversity import Activation
from cornac.metrics.diversity import GiniCoeff
from cornac.metrics.diversity import AlternativeVoices
from cornac.metrics.diversity import Representation
from cornac.metrics.diversity import Calibration
from cornac.metrics.diversity import ILD
from cornac.metrics.diversity import Fragmentation


# This just consider 'History' and 'Impressions', no consider time
def extract_seen_items(history, impressions):
    """
    Extracts a set of "seen items" from a user's history and impressions data.

    Parameters:
    history (str): A string containing history items, where items are separated by whitespace.
    impressions (str): A string of impression items, where each item may end with '-1' to indicate it was seen.

    Returns:
    set: A set of unique seen items from both history and impressions.
    """
    seen_items = set(str(history).split())
    impressions_items = [item.split('-')[0] for item in impressions.split() if item.endswith('-1')]
    seen_items.update(impressions_items)
    return seen_items


def extract_exposed_items(history, impressions):
    """
    Extracts a set of "exposed items" from a user's history and impressions data.

    Parameters:
    history (str): A string containing history items, where items are separated by whitespace.
    impressions (str): A string of impression items, where each item is in the format 'ID-status'.

    Returns:
    set: A set of unique exposed items from both history and impressions.
    """
    exposed_items = set(str(history).split())
    impressions_items = [item.split('-')[0] for item in impressions.split()]
    exposed_items.update(impressions_items)
    return exposed_items


def extract_seen_and_exposed_items(df):
    """
    Adds 'Seen Items' and 'Exposed Items' columns to a DataFrame by extracting items
    from each row's 'History' and 'Impressions' columns.

    Parameters:
    df (DataFrame): A DataFrame containing 'History' and 'Impressions' columns,
                    where each row represents a user's data.
                    - 'History' is expected to contain whitespace-separated item IDs.
                    - 'Impressions' contains item IDs with status (e.g., 'ID-1' or 'ID-0').

    Returns:
    DataFrame: The updated DataFrame with two new columns:
               - 'Seen Items': A set of items the user has seen.
               - 'Exposed Items': A set of items the user has been exposed to.
    """

    df['Seen Items'] = df.apply(lambda x: extract_seen_items(x['History'], x['Impressions']), axis=1)
    df['Exposed Items'] = df.apply(lambda x: extract_exposed_items(x['History'], x['Impressions']), axis=1)

    return df


# This just consider the column 'Impressions'
def extract_seen_items_separate(impressions):
    """
        Extracts the items that the user has seen from a string of impressions.

        Parameters:
        impressions (str): A string of space-separated impressions, where each impression is in the format 'itemID-status'.

        Returns:
        list: A list of item IDs that the user has seen (status '-1').
    """
    impressions_items_seen = [item.split('-')[0] for item in impressions.split() if item.endswith('-1')]
    return impressions_items_seen


def extract_exposed_items_separate(impressions):
    """
        Extracts all exposed items from a string of impressions.

        Parameters:
        impressions (str): A string of space-separated impressions, where each impression is in the format 'itemID-status'.

        Returns:
        list: A list of item IDs, extracted from the input string.
    """
    impressions_items_exposed = [item.split('-')[0] for item in impressions.split()]
    return impressions_items_exposed


def extract_seen_and_exposed_items_separate(df):
    """
        Extracts seen and exposed items from the 'Impressions' column of a DataFrame.

        Parameters:
        df (pandas.DataFrame): A DataFrame containing at least the 'Impressions' column,
                                where each entry is a space-separated string of item-status pairs.

        Returns:
        pandas.DataFrame: The input DataFrame with two new columns:
                           - 'Seen Items - Separate': List of items seen by the user.
                           - 'Exposed Items - Separate': List of all exposed items.
    """
    df['Seen Items - Separate'] = df.apply(lambda x: extract_seen_items_separate(x['Impressions']), axis=1)
    df['Exposed Items - Separate'] = df.apply(lambda x: extract_exposed_items_separate(x['Impressions']), axis=1)

    return df


def load_mind_data(news_path, behaviors_path):
    """
    Load MIND news and behaviors data from given file paths and return as DataFrames.

    Parameters:
    news_path (str): The file path to the news data (e.g., news.tsv).
    behaviors_path (str): The file path to the behaviors data (e.g., behaviors.tsv).

    Returns:
    tuple: A tuple containing two DataFrames:
           - df_news: DataFrame with news data.
           - df_behaviors: DataFrame with behaviors data, and 'Time' column as datetime with seen and exposed items extracted.
    """
    news_column_names = ['News ID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title Entities', 'Abstract Entities']
    behaviors_column_names = ['Impression ID', 'User ID', 'Time', 'History', 'Impressions']

    try:
        df_news = pd.read_csv(news_path, sep='\t', encoding='utf-8', header=None, names=news_column_names)
    except FileNotFoundError:
        print(f"Error: The file {news_path} was not found.")
        return None, None
    except pd.errors.ParserError:
        print(f"Error: Failed to parse the news file {news_path}. Please check the file format.")
        return None, None
    except Exception as e:
        print(f"Unexpected error while loading news data: {e}")
        return None, None

    try:
        # Load the behaviors data and convert the 'Time' column to datetime
        df_behaviors = pd.read_csv(behaviors_path, sep='\t', encoding='utf-8', header=None,
                                   names=behaviors_column_names)
        df_behaviors['Time'] = pd.to_datetime(df_behaviors['Time'], errors='coerce')
        # Check if any datetime conversions failed
        if df_behaviors['Time'].isnull().any():
            print("Warning: Some 'Time' values could not be converted to datetime format.")

        # Extract seen and exposed items
        df_behaviors = extract_seen_and_exposed_items(df_behaviors)
        df_behaviors = extract_seen_and_exposed_items_separate(df_behaviors)

    except FileNotFoundError:
        print(f"Error: The file {behaviors_path} was not found.")
        return None, None
    except pd.errors.ParserError:
        print(f"Error: Failed to parse the behaviors file {behaviors_path}. Please check the file format.")
        return None, None
    except Exception as e:
        print(f"Unexpected error while loading behaviors data: {e}")
        return None, None

    return df_news, df_behaviors


def load_globo_data(news_path, behaviors_path):
    """
    Load Globo news and behaviors data from the specified file paths and return as DataFrames.

    Parameters:
    news_path (str): The file path to the news data (e.g., .parquet file).
    behaviors_path (str): The file path to the behaviors data (e.g., .parquet file).

    Returns:
    tuple: A tuple containing two DataFrames:
           - df_news: DataFrame with news data.
           - df_behaviors: DataFrame with behaviors data, with the 'Time' column as datetime.
    """
    try:
        df_news = pd.read_parquet(news_path)
    except Exception as e:
        print(f"Error loading news data: {e}")
        return None, None
    try:
        df_behaviors = pd.read_parquet(behaviors_path)
        df_behaviors['Time'] = pd.to_datetime(df_behaviors['impressionTimestamp'], unit='ms', errors='coerce')
        df_behaviors['clickedItem'] = df_behaviors['clickedItem'].apply(lambda x: np.array([x], dtype=object))

        df_behaviors['Seen Items'] = df_behaviors.apply(
            lambda row: set(row['history']) | set(row['clickedItem']), axis=1)
        df_behaviors['Exposed Items'] = df_behaviors.apply(
            lambda row: set(row['impressions']) | set(row['history']), axis=1)

        df_behaviors['Seen Items - Separate'] = df_behaviors['clickedItem'].apply(set)
        df_behaviors['Exposed Items - Separate'] = df_behaviors['impressions'].apply(set)
        df_behaviors.rename(columns={'userId': 'User ID'}, inplace=True)
    except Exception as e:
        print(f"Error loading behaviors data: {e}")
        return df_news, None

    return df_news, df_behaviors


def cumulative_seen_items(items):
    seen = set()
    cumulative_list = []
    for item_set in items:
        seen.update(item_set)
        cumulative_list.append(seen.copy())
    return cumulative_list


def load_adressa_dataset(folder_path, adressa_uir):
    file_paths = glob.glob(folder_path + "2017010[1-7]")
    dataframes = []

    for file_path in file_paths:
        df = pd.read_json(file_path, lines=True)
        dataframes.append(df)

    df_adressa_behaviors = pd.concat(dataframes, ignore_index=True)
    df_adressa_behaviors = df_adressa_behaviors[['userId', 'time', 'id']].rename(
        columns={'userId': 'User ID', 'time': 'Time', 'id': 'Seen Items - Separate'})

    registered_users = adressa_uir['uid'].unique()
    df_adressa_behaviors = df_adressa_behaviors[df_adressa_behaviors['User ID'].isin(registered_users)]

    df_adressa_behaviors = df_adressa_behaviors.dropna(subset=['User ID', 'Time', 'Seen Items - Separate'])
    df_adressa_behaviors['Time'] = pd.to_datetime(df_adressa_behaviors['Time'], unit='s')

    df_adressa_behaviors = df_adressa_behaviors.groupby(['User ID', 'Time'], as_index=False).agg({
        'Seen Items - Separate': lambda x: set(x)})

    df_adressa_behaviors = df_adressa_behaviors.sort_values(by=['User ID', 'Time'])
    df_adressa_behaviors['Seen Items'] = (df_adressa_behaviors.groupby('User ID')['Seen Items - Separate']
                                          .apply(cumulative_seen_items)).explode().reset_index(drop=True)
    return df_adressa_behaviors


def update_seen_items(df, time_column='Time'):
    """
    Updates the 'Seen Items' column in the DataFrame so that later times include items from previous times.

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'User ID', time, and 'Seen Items' columns.
    time_column (str): The name of the time column. Default is 'Time'.
    seen_column (str): The name of the seen item column. Default is 'Seen Items'.

    Returns:
    pd.DataFrame: A DataFrame with updated 'Seen Items' columns.
    """
    df = df.sort_values(by=['User ID', time_column]).reset_index(drop=True)

    for user_id in df['User ID'].unique():
        user_df = df[df['User ID'] == user_id]
        accumulated_seen_items = set()
        for idx in user_df.index:
            accumulated_seen_items.update(df.at[idx, 'Seen Items'])
            df.at[idx, 'Seen Items'] = accumulated_seen_items.copy()

    return df


def update_exposed_items(df, time_column='Time'):
    """
    Updates the 'Exposed Items' column in the DataFrame to contain accumulated exposure data over time for each user.
    Each row's 'Exposed Items' will include all previous exposures for that user, based on the sorted time order.

    Parameters:
    df (DataFrame): A DataFrame with at least 'User ID', 'Exposed Items', and a timestamp column (default: 'Time').
                    - 'User ID': Identifies individual users.
                    - 'Exposed Items': A set of items the user has been exposed to in that row's context.
                    - time_column (str): The name of the column representing time, used for sorting exposure accumulation.

    Returns:
    DataFrame: The DataFrame with updated 'Exposed Items' containing cumulative exposures for each user.
    """
    df = df.sort_values(by=['User ID', time_column]).reset_index(drop=True)

    for user_id in df['User ID'].unique():
        user_df = df[df['User ID'] == user_id]
        accumulated_exposed_items = set()
        for idx in user_df.index:
            accumulated_exposed_items.update(df.at[idx, 'Exposed Items'])
            df.at[idx, 'Exposed Items'] = accumulated_exposed_items.copy()

    return df


def update_seen_and_exposed_items(df, time_column='Time'):
    """
    Updates both 'Seen Items' and 'Exposed Items' columns in the DataFrame so that later times include items from previous times.

    Parameters:
    df (pd.DataFrame): The input DataFrame with 'User ID', time, 'Seen Items', and 'Exposed Items' columns.
    time_column (str): The name of the time column. Default is 'Time'.

    Returns:
    pd.DataFrame: A DataFrame with updated 'Seen Items' and 'Exposed Items' columns.
    """
    # Update seen items
    df = update_seen_items(df, time_column)
    # Update exposed items
    df = update_exposed_items(df, time_column)

    return df


def get_top_n_user_ids(df, user_id_column='User ID', top_n=10):
    """
    Count the occurrences of each 'User ID' and return the top N most frequent user IDs.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a column for 'User ID'.
    user_id_column (str): The name of the 'User ID' column. Default is 'User ID'.
    top_n (int): The number of top frequent 'User ID's to return. Default is 10.

    Returns:
    pd.Series: A series containing the top N most frequent 'User ID's and their counts.
    """
    user_id_counts = df[user_id_column].value_counts()
    return user_id_counts.head(top_n)


def plot_metric_over_time_single(df, metric, recom_score=None):
    """
    Plots the given metric over time using Plotly and optionally adds a reference line for recommended activation score.

    Parameters:
    df (DataFrame): DataFrame containing 'Time' and the metric columns.
    metric (str): The column name of the metric to plot (e.g., 'activation score', 'gini score').
    recom_score (str or None): The column name of the recommended score to add as a reference line. Default is None.

    Returns:
    None: Displays the interactive plot for the selected metric over time.
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df[metric],
        mode='lines+markers',
        name=f"{df['user id'].iloc[0]} {metric.replace('_', ' ').capitalize()}",
        line=dict(color='blue'),
        marker=dict(color='blue'),
        showlegend=True
    ))

    if recom_score is not None:
        fig.add_trace(go.Scatter(
            x=df['Time'],
            y=df[recom_score],
            mode='lines+markers',
            name=f"{recom_score.replace('_', ' ').capitalize()}",
            line=dict(color='red', dash='dash'),
            marker=dict(color='red')
        ))

    fig.update_layout(
        title=f'{metric.replace("_", " ").capitalize()} Over Time',
        xaxis_title='Time',
        yaxis_title=metric.replace('_', ' ').capitalize(),
        legend_title="Legend",
        xaxis=dict(tickangle=-45),
        hovermode="x",
        template='plotly_white',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Display the plot
    fig.show()


def plot_user_diversity_during_time(df):
    """
    This function creates an interactive plot to visualize user diversity metrics over time.
    Users can search for specific user IDs, select a user and metric from dropdowns,
    and view time series plots for each metric, including optional recommendation metrics.

    Parameters:
    df (DataFrame): A DataFrame containing columns for user IDs, time, and various metrics
                    for diversity (e.g., activation score, gini score, etc.).

    Returns:
    None: Displays an interactive plot and search widget for exploring user diversity metrics.
    """
    search_text = widgets.Text(
        description='Search:',
        placeholder='Type to search user...',
    )

    user_dropdown = widgets.Dropdown(
        options=[],
        description='User:',
    )
    score_columns = [col for col in df.columns if 'score' in col]

    metric_dropdown = widgets.Dropdown(
        #options=['activation score', 'gini score', 'alternative voice score',
                 #'representation score', 'cali_complexity score',
                 #'cali_category score', 'ild score'],
        options=score_columns,
        description='Metric:',
    )

    def update_dropdown_options(change):
        search_value = search_text.value
        if search_value:
            filtered_users = [user for user in all_users if search_value.lower() in user.lower()]
        else:
            filtered_users = all_users
        user_dropdown.options = filtered_users

    def update_plot(user_id, metric):
        filtered_df = df[df['user id'] == user_id]

        fig = go.Figure()
        recom_metric = f"recom {metric}"
        fig.add_trace(go.Scatter(
            x=filtered_df['Time'],
            y=filtered_df[metric],
            mode='lines+markers',
            name=f'{user_id} {metric.capitalize()}'
        ))

        if recom_metric in filtered_df.columns:
            fig.add_trace(go.Scatter(
                x=filtered_df['Time'],
                y=filtered_df[recom_metric],
                mode='lines+markers',
                name=f'Recommendation {metric.capitalize()}',
                line=dict(color='red', dash='dash')
            ))

        fig.update_layout(
            title=f'{metric.capitalize()} Over Time for {user_id}',
            xaxis_title='Time',
            yaxis_title=metric.capitalize()
        )

        fig.show()

    all_users = df['user id'].unique()
    search_text.observe(update_dropdown_options, names='value')
    display(search_text)
    update_dropdown_options(None)

    interactive_plot = widgets.interactive(update_plot, user_id=user_dropdown, metric=metric_dropdown)
    display(interactive_plot)


def calculate_user_diversity_during_time(df, df_uir, user_ids, sentiment_path=None, genre_path=None,
                                         min_maj_path=None, entities_path=None, genre_multi_path=None, complexity_path=None,
                                         time='Time', minutes=0, hours=0, days=1, separate=False):
    """
    Calculate seen and exposed items during a specified time interval for one or more users,
    with an option to use separate or accumulative methods.

    Parameters:
    df (DataFrame): The behaviors dataframe including 'Seen Items' and 'Exposed Items' columns
                    (or 'Seen Items - Separate' and 'Exposed Items - Separate' if separate is True).
    df_uir (DataFrame): The user-item-rating dataframe.
    user_ids (list/str/int): The user ID(s) to filter the data. Can be a single user ID or a list of user IDs.
    sentiment_path: enhanced sentiment data file path (optional)
    genre_path: enhanced category data file path (optional)
    min_maj_path: enhanced major-minor data file path (optional)
    entities_path: enhanced party data file path (optional)
    genre_multi_path: enhanced category data file path (optional)
    complexity_path: enhanced complexity data file path (optional)
    time (str): The column name for time, default is 'Time'.
    minutes (int): Number of minutes for the time interval.
    hours (int): Number of hours for the time interval.
    days (int): Number of days for the time interval.
    separate (bool): Flag to indicate whether to use separate method (True) or accumulative method (False).

    Returns:
    DataFrame: A DataFrame containing user_id, timestamp, seen_items, exposed_items, and all metric scores.
    """
    if isinstance(user_ids, (str, int)):
        user_ids = [user_ids]

    all_results = []
    calculate_recom_scores = 'Exposed Items' in df.columns
    calculate_recom_scores_separate = 'Exposed Items - Separate' in df.columns

    for user_id in user_ids:
        if separate:
            user_df = df[df['User ID'] == user_id].sort_values(by=time)
            results = []

            for _, row in user_df.iterrows():
                seen_items = row['Seen Items - Separate']
                exposed_items = row['Exposed Items - Separate'] if calculate_recom_scores_separate else None
                timestamp = row[time]
                results.append((user_id, timestamp, list(seen_items), list(exposed_items) if exposed_items else []))

            result_df = pd.DataFrame(results, columns=['user id', 'Time', 'item seen', 'item exposed'])

        else:
            user_df = df[df['User ID'] == user_id].sort_values(by=time)
            seen_items = set()
            exposed_items = set() if calculate_recom_scores else None
            results = []
            start_time = user_df[time].min()
            end_time = user_df[time].max()
            time_interval = timedelta(days=days, hours=hours, minutes=minutes)
            next_time = start_time + time_interval

            while start_time <= end_time:
                interval_df = user_df[user_df[time] <= start_time]
                for _, row in interval_df.iterrows():
                    seen_items.update(row['Seen Items'])
                    if calculate_recom_scores:
                        exposed_items.update(row['Exposed Items'])
                results.append((user_id, start_time, list(seen_items.copy()),
                                list(exposed_items.copy()) if calculate_recom_scores else []))
                start_time = next_time
                next_time = start_time + time_interval

            if start_time > end_time:
                results.append((user_id, end_time, list(user_df[user_df[time] == end_time]['Seen Items'].iloc[0]),
                                list(user_df[user_df[time] == end_time]['Exposed Items'].iloc[
                                         0]) if calculate_recom_scores else []))

            result_df = pd.DataFrame(results, columns=['user id', 'Time', 'item seen', 'item exposed'])

        pool = df_uir['iid'].tolist()

        if sentiment_path:
            sentiment = mind.load_sentiment(fpath=sentiment_path)
            user_activation_metric = UserActivation(item_sentiments=sentiment,
                                                    user_seen_item=result_df[['user id', 'Time', 'item seen']].copy())
            activation = user_activation_metric.compute_user_during_time(
                user_exposed_df=result_df[
                    ['user id', 'Time', 'item exposed']].copy() if calculate_recom_scores else None)
            result_df['activation score'] = activation['activation score']

            if calculate_recom_scores or calculate_recom_scores_separate:
                recom_activation_metric = Activation(item_sentiment=sentiment)
                recom_activation = [recom_activation_metric.compute(pd_rank=row['item exposed'], pool=pool)
                                    for _, row in result_df.iterrows()]
                result_df['recom activation score'] = recom_activation

        if genre_path:
            genre = mind.load_category_multi(fpath=genre_path)
            user_gini_metric = UserGiniCoeff(item_genres=genre,
                                             user_seen_item=result_df[['user id', 'Time', 'item seen']].copy())
            gini = user_gini_metric.compute_user_during_time()
            result_df['gini score'] = gini['gini score']

            user_ild_metric = UserILD(item_features=genre,
                                      user_seen_item=result_df[['user id', 'Time', 'item seen']].copy())
            ild = user_ild_metric.compute_user_during_time()
            result_df['ild score'] = ild['ild score']

            if calculate_recom_scores or calculate_recom_scores_separate:
                recom_gini_metric = GiniCoeff(item_genre=genre)
                recom_gini = [recom_gini_metric.compute(pd_rank=row['item exposed'])
                              for _, row in result_df.iterrows()]
                result_df['recom gini score'] = recom_gini

                recom_ild_metric = ILD(item_feature=genre)
                recom_ild = [recom_ild_metric.compute(pd_rank=row['item exposed'])
                             for _, row in result_df.iterrows()]
                result_df['recom ild score'] = recom_ild

        if min_maj_path:
            min_maj = mind.load_min_maj(fpath=min_maj_path)
            user_min_maj_metric = UserAlternativeVoices(item_minor_major=min_maj,
                                                        user_seen_item=result_df[
                                                            ['user id', 'Time', 'item seen']].copy())
            alternative_voice = user_min_maj_metric.compute_user_during_time(
                user_exposed_df=result_df[
                    ['user id', 'Time', 'item exposed']].copy() if calculate_recom_scores else None)
            result_df['alternative voice score'] = alternative_voice['alternative voice score']

            if calculate_recom_scores or calculate_recom_scores_separate:
                recom_alternative_voice_metric = AlternativeVoices(item_minor_major=min_maj)
                recom_alternative_voice = [
                    recom_alternative_voice_metric.compute(pd_rank=row['item exposed'], pool=pool)
                    for _, row in result_df.iterrows()]
                result_df['recom alternative voice score'] = recom_alternative_voice

        if entities_path:
            entities = mind.load_entities(fpath=entities_path)
            user_representation_metric = UserRepresentation(item_entities=entities, user_seen_item=result_df[
                ['user id', 'Time', 'item seen']].copy())
            representation = user_representation_metric.compute_user_during_time(
                user_exposed_df=result_df[
                    ['user id', 'Time', 'item exposed']].copy() if calculate_recom_scores else None)
            result_df['representation score'] = representation['representation score']

            if calculate_recom_scores or calculate_recom_scores_separate:
                recom_representation_metric = Representation(item_entities=entities)
                recom_representation = [recom_representation_metric.compute(pd_rank=row['item exposed'], pool=pool)
                                        for _, row in result_df.iterrows()]
                result_df['recom representation score'] = recom_representation

        if complexity_path:
            complexity = mind.load_complexity(fpath=complexity_path)
            user_complexity_metric = UserCalibration(item_features=complexity,
                                                     user_seen_item=result_df[['user id', 'Time', 'item seen']].copy(),
                                                     data_type='complexity')
            cali_complexity = user_complexity_metric.compute_user_during_time(
                user_exposed_df=result_df[
                    ['user id', 'Time', 'item exposed']].copy() if calculate_recom_scores else None)
            result_df['cali_complexity score'] = cali_complexity['cali_complexity score']

            if calculate_recom_scores or calculate_recom_scores_separate:
                recom_complexity_metric = Calibration(item_feature=complexity, data_type='complexity')
                recom_complexity = [
                    recom_complexity_metric.compute(pd_rank=row['item exposed'], user_history=row['item seen'])
                    for _, row in result_df.iterrows()]
                result_df['recom cali_complexity score'] = recom_complexity

        if genre_multi_path:
            category_cali = mind.load_category(fpath=genre_multi_path)
            user_category_metric = UserCalibration(item_features=category_cali,
                                                   user_seen_item=result_df[['user id', 'Time', 'item seen']].copy(),
                                                   data_type='category')
            cali_category = user_category_metric.compute_user_during_time(
                user_exposed_df=result_df[
                    ['user id', 'Time', 'item exposed']].copy() if calculate_recom_scores else None)
            result_df['cali_category score'] = cali_category['cali_category score']

            if calculate_recom_scores or calculate_recom_scores_separate:
                recom_category_metric = Calibration(item_feature=category_cali, data_type='category')
                recom_category = [
                    recom_category_metric.compute(pd_rank=row['item exposed'], user_history=row['item seen'])
                    for _, row in result_df.iterrows()]
                result_df['recom cali_category score'] = recom_category

        all_results.append(result_df)

    combined_results = pd.concat(all_results, ignore_index=True)
    return combined_results


def compute_relative_time(df, time_unit='hours'):
    """
    Converts 'Time' to datetime format, sorts by user and time,
    and calculates relative time for each user in the specified time unit.

    Parameters:
        df (DataFrame): DataFrame containing user data with 'user id' and 'Time'.
        time_unit (str): Unit for relative time calculation, either 'hours' or 'days'.

    Returns:
        DataFrame: The input DataFrame with an additional 'relative_time' column.
    """

    df['Time'] = pd.to_datetime(df['Time'])
    df.sort_values(['user id', 'Time'], inplace=True)

    def relative_time(group):
        group = group.copy()
        if time_unit == 'hours':
            group['relative_time'] = (group['Time'] - group['Time'].iloc[0]).dt.total_seconds() / 3600
        elif time_unit == 'days':
            group['relative_time'] = (group['Time'] - group['Time'].iloc[0]).dt.total_seconds() / 86400
        else:
            raise ValueError("Invalid time_unit. Choose either 'hours' or 'days'.")
        return group

    df = df.groupby('user id').apply(relative_time)

    return df


def plot_metrics_histograms(df, num_users, time_col, metrics, bins):
    """
    Plots individual histograms for each metric with user counts along the x-axis and time on the y-axis.

    Parameters:
    - df (DataFrame): The input DataFrame containing the data.
    - num_user (int): The number of users to ensure consistent sample sizes across time points.
    - time_col (str): The column name representing timestamps.
    - metrics (list of str): List of metric columns to plot.
    - bins (dict): Dictionary specifying bins for each metric, e.g., {'metric_name': np.linspace(start, end, num_bins)}

    Returns:
    - None: Displays a plot for each metric with time on the y-axis.
    """

    df.sort_values(by=time_col, inplace=True)
    #times = df[time_col].unique()
    time_counts = df[time_col].value_counts()
    times = time_counts[time_counts == num_users]
    times = times.keys()
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, num_metrics * 4), sharex=True)
    fig.suptitle("User Metric Distribution Over Time")

    for i, metric in enumerate(metrics):
        hist_data = np.zeros((len(times), len(bins[metric]) - 1))

        for j, time in enumerate(times):
            df_time = df[df[time_col] == time]
            hist_data[j], _ = np.histogram(df_time[metric], bins=bins[metric])

        colors = plt.cm.viridis(np.linspace(0, 1, len(bins[metric]) - 1))
        legend_labels = [f"{bins[metric][k]} - {bins[metric][k + 1]}" for k in range(len(bins[metric]) - 1)]

        for j, time in enumerate(times):
            for k in range(len(hist_data[j])):
                axes[i].barh(y=time, width=hist_data[j][k], left=np.sum(hist_data[j][:k]),
                             height=0.4, edgecolor='black', align='center', color=colors[k])

        axes[i].set_title(f"{metric} Distribution Over Time")
        axes[i].set_ylabel("Time")
        axes[i].invert_yaxis()
        axes[i].set_xlabel("User Count")

        patches = [plt.Line2D([0], [0], color=colors[k], lw=4) for k in range(len(legend_labels))]
        axes[i].legend(patches, legend_labels, title="Value Ranges", loc='lower right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])


def compute_quantiles(df, metrics):
    """
    Computes the min, 25%, 50%, 75%, and max quantiles for the specified metrics in the DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data
    - metrics: list of column names (metrics) for which quantiles need to be computed

    Returns:
    - A dictionary where each metric is a key, and the values are arrays of quantiles (min, 25%, 50%, 75%, max)
    """
    quantiles = df.describe().loc[['min', '25%', '50%', '75%', 'max'], metrics]
    quantiles = quantiles.T
    quantiles_dict = quantiles.to_dict(orient='index')

    converted_quantiles = {metric: np.array([quantiles_dict[metric]['min'],
                                             quantiles_dict[metric]['25%'],
                                             quantiles_dict[metric]['50%'],
                                             quantiles_dict[metric]['75%'],
                                             quantiles_dict[metric]['max']])
                           for metric in quantiles_dict}

    return converted_quantiles

