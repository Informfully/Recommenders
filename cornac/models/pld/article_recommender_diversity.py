#############################################################################################################
# VERSION:      1.0 (STANDALONE)                                                                            #
# DESCRIPTION:  PLD - Political Diversity Model for Cornac                                                  #
#############################################################################################################

import math
import numpy as np
import random

# Calculate the difference in political score for a given user-article item
def CalculateDistance(user, article):

    distance = []

    for i in range(len(user)):
        distance.append(abs(user[i]-article[i]))

    # # Calculate absolute user-article distance for both of the available dimensions : 2D data

    distance = [pow(item, 2) for item in distance]

    # Tweak weight to adjust the contribution of individual dimensions
    overallDistance = math.sqrt(np.sum(distance))

    return overallDistance

# Function to round either user or article score to the level of granularity of score groups
def RoundScore(score, distribution, group_granularity):

    # Rounding is always done with the available group scores in mind
    for i in range(len(score)): 

        for group in range(0, len(distribution)):
            
            if (abs(distribution[group][0][0] - score[i]) <= 0.5 * group_granularity):
                score[i] = distribution[group][0][0]

    return score

# Predict score for a given user-item combination
def Predict(user, articles, distribution, group_granularity):

    # Empty recommendatino list that gets populated with the indices of recommended articles
    singleUserRecommendation = []

    # Round user score to assign it to user groups
    userScoreRounded = user
    # Round article scores for assigning them to score groups
    articleScoresRounded = articles
    # print(f"articleScoresRounded:{articleScoresRounded}")

    indexed_articles = list(enumerate(articleScoresRounded))  # [(0, score0), (1, score1), ...]
    random.shuffle(indexed_articles)

    distributionD = np.zeros((len(user), len(distribution[0][1])))

    distributionMerged = None


        # Find the distribution groups that fit the article score in each dimension
    for k in range(len(distributionD)):
        for group in range(0, len(distribution)):
            if (distribution[group][0] == userScoreRounded[k]):
                distributionD[k] = distribution[group][1]
    # print(f"distributionD:{distributionD}")

    # print(f"distributionD shape: {distributionD.shape}")
    if len(distributionD) > 4:
        # randomly choose two distribution to merge (due to memory restriction)
        X, Y, Z = random.choices(range(len(distributionD)), k=3)
        distributionD = distributionD[[X,Y,Z]]
    
    ##############add for 1-D user score
    if len(distributionD) == 1:
        distributionMerged = distributionD
    
    ##############
    if len(distributionD)>1:
        for i in range(len(distributionD)-1):
            if i == 0:
                distributionMerged = np.add.outer(distributionD[i], distributionD[i+1])
            else:
                distributionMerged = np.add.outer(distributionMerged, distributionD[i+1])

    # Ensure shape of distributionMerged aligns with userScoreRounded
    while distributionMerged.ndim < len(userScoreRounded):
        distributionMerged = np.expand_dims(distributionMerged, axis=-1)

    # print(f"distributionMerged:{distributionMerged}")
    articles_num = int(np.sum(distributionMerged))
    # print(f"articles_num:{articles_num}")
    # Go through all article groups recommend articles from groups with the highest count first 
    for _ in range(articles_num):

            # Look for the group with the most articles in it
            max_index = np.argmax(distributionMerged)
            max_coords  = np.unravel_index(max_index, distributionMerged.shape)


            targetScore = np.zeros(len(userScoreRounded))

            relevant_coords = max_coords[-len(userScoreRounded):]

            for i in range(len(targetScore)):
                targetScore[i] = -1 + relevant_coords[i] * group_granularity



            for original_index, currentArticleScore in indexed_articles:
               
                if currentArticleScore <= targetScore + abs(group_granularity) and currentArticleScore >= targetScore - abs(group_granularity):
                        # Only append the article to recommendations in case it is not yet listed
                    if (original_index not in singleUserRecommendation):
                        singleUserRecommendation.append(original_index)
                        ## issue: lack a break statement, just loop all availble items and add all
                        break

            # after recommending, the chosen max value in distribution decreases
            distributionMerged[max_coords] -= 1
    # print(f"singleUserRecommendation:{singleUserRecommendation}")
    return singleUserRecommendation