from collections import Counter
import random
from cornac.eval_methods import RatioSplit
from cornac.experiment.experiment import Experiment
from cornac.metrics import NDCG, Recall

from cornac.metrics import Activation
from cornac.metrics import Calibration

from cornac.models import UserKNN, ItemKNN
from cornac.datasets import mind as mind
from cornac.rerankers import GreedyKLReranker, LeastPopReranker, PM2Reranker
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


news_files_dir = os.path.join(current_dir, 'example_news_files')

config_files_dir = os.path.join(current_dir, 'example_config_files')
sys.path.insert(0, news_files_dir)
sys.path.insert(0, config_files_dir)

input_path = news_files_dir
output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.', 'output_KNN_reranking_result')

uir_path = os.path.join(input_path, 'example_impression_all_uir.csv')

feedback = mind.load_feedback(fpath=uir_path)

sentiment_file_path  = os.path.join(input_path, 'example_sentiment.json')
sentiment = mind.load_sentiment(fpath=sentiment_file_path)
category_file_path  = os.path.join(input_path, 'example_category.json')
category = mind.load_category(fpath= category_file_path)
rs = RatioSplit(
    data=feedback,
    test_size=0.2,
    rating_threshold=1,
    seed=42,
    exclude_unknowns=True,
    verbose=True,
)
Item_sentiment = mind.build(
    data=sentiment, id_map=rs.global_iid_map)

Item_category = mind.build(
    data=category, id_map=rs.global_iid_map)

out_pd = pd.Series(Item_category).to_frame('category').join(
    pd.Series(Item_sentiment).to_frame('sentiment'), how='outer')
print(f"Input dataframe:{out_pd}")
# define metric
act = Activation(item_sentiment=Item_sentiment, k=10)
cal = Calibration(item_feature=Item_category,
                  data_type="category", k=10)

# 
    # sentiment_party_greedy_reranker = GreedyKLReranker(
    #     item_dataframe = article_feature_dataframe_keep_no_party,
    #     diversity_dimension = ["sentiment", "entities"],
    #     top_k=targetSize,
    #     target_distributions = Target_Mind_distribution,
    #     diversity_dimension_weight=[0.5, 0.5],
    #     user_item_history = user_item_history, 
    #     rerankers_item_pool = impression_iid_list 

    # )

category_100_reranker = GreedyKLReranker(name="greedyKL_cat",
                                         item_dataframe=out_pd, diversity_dimension=["category"],  top_k=10, pool_size=-1, # all items

                                         target_distributions={"category": {"type": "discrete", "distr": {
                                             "a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2,"e":0.2, "f":0}}},
                                         diversity_dimension_weight=[1])

category_80_sentiment_20_reranker = GreedyKLReranker(
    name="greedyKL_cat_senti",
    item_dataframe=out_pd,
    diversity_dimension=["category", "sentiment"],
    pool_size=-1,
    top_k=10,
    target_distributions={
        "category": {
            "type": "discrete",
            "distr": {
                "a": 0.2,
                "b": 0.2,
                "c": 0.2,
                "d": 0.2,
                "e": 0.2,
                "f":0
            }
        },
        "sentiment": {
            "type": "continuous",
            "distr": [
                    {"min": -1, "max": -0.5, "prob": 0},
                {"min": -0.5, "max": 0, "prob": 0},
                {"min": 0, "max": 0.5, "prob": 0.5},
                {"min": 0.5, "max": 1.01, "prob": 0.5}
            ]
        }
    },
    diversity_dimension_weight=[0.8, 0.2]
)

LeastPop_reranker = LeastPopReranker(top_k=10, pool_size=-1)
Experiment(eval_method=rs,
           models=[UserKNN(k=3, similarity="pearson",
                           name="UserKNN")],
           metrics=[NDCG(k=10), Recall(k=10), act, cal],
           rerankers={'static': [category_100_reranker,
                      category_80_sentiment_20_reranker, LeastPop_reranker]}
           ).run()


def calculate_distribution(df, item_ids, column_name, normalize=True, bins=None, bin_labels=None, plot=False, plot_file=None):

    # Filter the DataFrame to only include specified item IDs
    if len(item_ids) > 0:
        filtered_df = df.loc[item_ids]

        # Check if the column needs binning
        if bins is not None and pd.api.types.is_numeric_dtype(filtered_df[column_name]):
            # Bin the data if bins are provided and the data type is numeric
            if bin_labels is None or len(bin_labels) != len(bins) - 1:
                raise ValueError(
                    "Bin labels must be provided and match the number of bins - 1.")
            filtered_df['binned'] = pd.cut(
                filtered_df[column_name], bins=bins,  right=False, labels=bin_labels)
            # Calculate the distribution of the binned data
            distribution = filtered_df['binned'].value_counts(
                normalize=normalize)
        else:
            # Calculate the distribution of the categorical data
            distribution = filtered_df[column_name].value_counts(
                normalize=normalize)
        if plot:
            distribution.plot(kind='bar', color='skyblue')
            plt.xlabel(column_name)
            plt.ylabel('Proportion')
            plt.title(f'Distribution of {column_name}')
            plt.xticks(rotation=45)
            plt.tight_layout()

            if plot_file:
                plt.savefig(plot_file)
                print(f"Plot saved as {plot_file}")
                plt.close()
            # plt.show()

        return distribution
    else:
        return {}


save_dir = os.path.join(output_file_path, 'category_100_reranker_figures_random10user')
os.makedirs(save_dir, exist_ok=True)

random_users = random.sample(list(category_100_reranker.ranked_items.keys()), 10)


category_100_reranker_unique_items = set()
# # Iterate over the randomly selected users
for user_index, test_user in enumerate(random_users):
    # Get the list of items for the current user
    items = category_100_reranker.ranked_items[test_user]
    category_100_reranker_unique_items.update(items)
    # Create a unique filename for the plot, using the user index or a specific identifier
    plot_filename = os.path.join(
        save_dir, f"user_{test_user}_category_distribution.png")

    # Calculate and plot the distribution, saving the plot to a file
    q = calculate_distribution(out_pd, items, 'category', normalize=True,
                               plot=True, plot_file=plot_filename)

print(
    f"Number of unique items across the 10 users: {len(category_100_reranker_unique_items)}")


save_dir = os.path.join(output_file_path, 'category_80_sentiment_20_reranker_figures_random10user')
os.makedirs(save_dir, exist_ok=True)
category_80_sentiment_20_reranker_unique_items = set()

# # Iterate over all users in test_set
for user_index, test_user in enumerate(random_users):
    # Get the list of items for the current user
    items = category_80_sentiment_20_reranker.ranked_items[test_user]
    category_80_sentiment_20_reranker_unique_items.update(items)
    # Create a unique filename for the plot, using the user index or a specific identifier
    plot_filename = os.path.join(
        save_dir, f"user_{test_user}_category_distribution.png")

    # Calculate and plot the distribution, saving the plot to a file
    q = calculate_distribution(out_pd, items, 'category', normalize=True,
                               plot=True, plot_file=plot_filename)
print(
    f"Number of unique items across the 10 users: {len(category_80_sentiment_20_reranker_unique_items)}")

for user_index, test_user in enumerate(random_users):
    # Get the list of items for the current user
    items = category_80_sentiment_20_reranker.ranked_items[test_user]

    # Create a unique filename for the plot, using the user index or a specific identifier
    plot_filename = os.path.join(
        save_dir, f"user_{test_user}_sentiment_distribution.png")

    # Calculate and plot the distribution, saving the plot to a file
    bins = [-1, -0.5, 0, 0.5, 1]
    bin_labels = ['-1,-0.5', '-0.5,0', '0,0.5', '0.5,1']
    q = calculate_distribution(out_pd, items, 'sentiment', normalize=True,
                               bins=bins, bin_labels=bin_labels, plot=True, plot_file=plot_filename)

save_dir = os.path.join(output_file_path, 'leastPop_reranker_figures_random10user')
# # This will create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)
leastPop_reranker_unique_items = set()
for user_index, test_user in enumerate(random_users):
    items = LeastPop_reranker.ranked_items[test_user]
    leastPop_reranker_unique_items.update(items)
    values = LeastPop_reranker.popularityScores(items)

    frequency = Counter(values)
    # Separate the scores and their counts for plotting
    scores = list(frequency.keys())
    counts = list(frequency.values())

    # Create a bar plot for popularity scores
    plt.figure(figsize=(10, 6))  # Set figure size

    # Bar plot showing popularity score frequency
    plt.bar(scores, counts, color='skyblue', edgecolor='black', alpha=0.8)

    # Add gridlines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add titles and labels
    plt.title(
        f"Popularity Score Distribution for User {test_user}", fontsize=14)
    plt.xlabel("Popularity Score", fontsize=12)
    plt.ylabel("Count", fontsize=12)

    # Ensure the X-axis has only integer ticks (since popularity score is int)
    plt.xticks(scores, fontsize=10)
    plt.yticks(fontsize=10)

    # Add labels on top of each bar to show the exact count
    for i, count in enumerate(counts):
        plt.text(scores[i], count + 0.1, str(count), ha='center', fontsize=10)

    # Create a unique filename for the plot
    plot_filename = os.path.join(
        save_dir, f"user_{test_user}_popularity_distribution.png")

    # Save the plot to file
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory
print(
    f"Number of unique items across the 10 users: {len(leastPop_reranker_unique_items)}")
