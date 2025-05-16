#############################################################################################################
# VERSION:      1.0 (STANDALONE)                                                                            #
# DESCRIPTION:  PLD - Political Diversity Model for Cornac                                                  #
#############################################################################################################

import numpy as np
from tqdm import tqdm
import copy
# For political diversity
def calculatePoliticalScore(history_dict, party_dict_raw, party_list, num_users):
         
    user_score_matrix = np.full((num_users, len(party_list)), 0, dtype=float)
    party_dict = {}
    for k, v in party_dict_raw.items():
        if len(list(v)) == 0:
            party_dict[k] = -1
            # party_dict[k] = 0
        else:

            political_dict = {item: v[item] for item in party_list if item in v.keys()}

            if political_dict:
                max_party = max(political_dict, key=political_dict.get)
                party_dict[k] = party_list.index(max_party)
            else:
                party_dict[k] = -1
                # party_dict[k] = 0
        
    
    for user_idx, article_list in history_dict.items():

        # Update: for multi-party situation
        for i, article in enumerate(article_list):
            if article in party_dict.keys():
                if party_dict[article] == -1:
                    continue
                # print(party_dict[article])
                user_score_matrix[user_idx][party_dict[article]] += 1
    

    # user_score_matrix = roundColumnScore(user_score_matrix)
    user_score_matrix =  compute_political_leaning(user_score_matrix)

    return user_score_matrix


def roundColumnScore(scores_matrix):
    max = np.max(scores_matrix, axis=0)
    min = np.min(scores_matrix, axis=0)
    denominator = max - min
    if denominator.any() == 0:
        zero_i = np.where(denominator == 0)
        denominator[zero_i] = 1
        scores_matrix = 2*(scores_matrix - min)/denominator - 1
        scores_matrix[:,zero_i] = 0
        return scores_matrix
    else:
        return 2*(scores_matrix - min)/denominator - 1

def compute_political_leaning(counts_matrix):
    republican_count = counts_matrix[:, 0]
    democrat_count = counts_matrix[:, 1]
    total_mentions = republican_count + democrat_count

    leaning_score = np.zeros_like(total_mentions, dtype=float)  # default to neutral

    for i in range(len(total_mentions)):
        if total_mentions[i] != 0:
            leaning_score[i] = (republican_count[i] - democrat_count[i]) / total_mentions[i]
        # else: leaning_score[i] stays 0

    return leaning_score.reshape(-1, 1)

def calculateArticleScore(history_dict, userScores, num_users, num_items,  party_dict, party_list, article_pool, positive_score_party_name, negative_score_party_name):
    # article_mention_matrix = np.full((len(article_pool), len(party_list)), 0, dtype=float)
    article_mention_matrix = np.zeros((len(article_pool), len(party_list)), dtype=float)
    
    # for i in range(len(article_pool)):
    for i, article_id in enumerate(article_pool):
        
        parties = party_dict.get(article_id, {}) 

        positive_score_parties_count = parties.get(positive_score_party_name, 0)
        negative_score_parties_count = parties.get(negative_score_party_name, 0)
        
        article_mention_matrix[i, 0] = positive_score_parties_count  # First column for positive score party count (e.g., Republican count)
        article_mention_matrix[i, 1] = negative_score_parties_count    # Second column for negative score party count (e.g., Democrat count) 



    # Initialize article scores and counts
    articleScores = np.zeros(len(article_pool), dtype=float)
    articleCounts = np.zeros(len(article_pool), dtype=int)  # To count how many users read each article
    
    # Set to track processed articles
    processed_articles = set()

    # Step 1: Calculate article score based on user scores
    for u, uHistory in tqdm(history_dict.items(), total=len(history_dict.items()), desc="Processing History data"):
        for article_id in uHistory:
            # Record article if it's not processed
            
            if  article_id in article_pool:
                index = article_pool.index(article_id)

                articleScores[index] += userScores[u]
                articleCounts[index] += 1

                processed_articles.add(index)


    # Step 2: For articles with no readers, use proportional method
    # Calculate the overall leaning for all articles using party_mentions
    total_mentions = np.sum(article_mention_matrix, axis=1)  # Total mentions across both parties
    total_republican = article_mention_matrix[:, 0]  # Republican mentions for each article
    total_democrat = article_mention_matrix[:, 1]    # Democrat mentions for each article

    # Compute the proportional leaning score for each article
    proportional_scores = np.zeros_like(total_republican)
    for i in range(len(total_mentions)):
        if total_mentions[i] != 0:
            proportional_scores[i] = (total_republican[i] - total_democrat[i]) / total_mentions[i]
        else:
            proportional_scores[i] = 0  # If no mentions, set score to 0


    # Step 3: For articles that have been read (processed), calculate the average score
    for idx in processed_articles:
        if articleCounts[idx] > 0:
            # Calculate the average score for the article
            articleScores[idx] /= articleCounts[idx]

    # Step 4: For unread articles (those not processed), apply proportional score
    for idx in range(len(article_pool)):
        if idx not in processed_articles:
            articleScores[idx] = proportional_scores[idx]


    print("Finished calculating article score")

    return articleScores.reshape(-1, 1)  # Return as column vector






