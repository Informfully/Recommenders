import logging
import time
import graph_recommender as graphRecommender

# Default parameter values for the recommender algorithm
PARAMETER_B = 2
PARAMETER_ITERATIONS = 2
PARAMETER_TYPE = "broadening"

# Global variabe for storing and the RWE ideology model of Bibek's recommender algorithm
MODEL_RWE = []

def getParameterB():
    return PARAMETER_B

def getParameterIterations():
    return PARAMETER_ITERATIONS

def getParameterType():
    return PARAMETER_TYPE

def setParameterB(b):
    PARAMETER_B = b

def setParameterIterations(iterations):
    PARAMETER_ITERATIONS = iterations

def setParameterType(type):
    PARAMETER_TYPE = type

# Detailed description of recommender parameters from meetings and e-mail conversations with Bibek:
#
# If you have multiple training datasets (e.g., if multiple train-test splits) pass a list of training 
# datasets to the recommender.
#
# Value "b" is the amount by which you want the algorithm to diversify recommendations:
#   b > 1   will add more diverse items, e.g., b = 2, 5, 10 etc.
#   b < 1   will add comparatively less (but still more than normal) items: e.g., b = 0.9, 0.1, etc.
# Important: Do not use negative values for "b"!
#
# The value of "iteration" determines for how long should the random walks on the graph continue:
#   1       value of the usual recommender where there is no diversification
#   > 1     starts to diversify
#   5-10    good value based on Bibek's experiene with Twitter and hashtags
#
# Vectors "user_positions" and "item_positions" containing user and article scores. And their respective 
# length needs to match the number of users and items in the training dataset UxI with the dimensions 
# len(user_positions) x len(item_position).
#
# Parameters "type" is one of either "moderating" or "broadening" offering the following functionality:
#   moderationg     Recommend only articles that have a score closer to 0 than the rating of the user. 
#                   E.g., if a user has a score of -0.5, it will not consider articles with a score
#                   < -0.5, i.e., only articles that have less of an "extreme" value (i.e., more to
#                   either -1.0 or 1.0 end of the spectrum) in the political dimension.
#   broadening      Recommend all articles no matter what their score is.
# "Moderating" is set by default if not type is specified.
#
# Return values of "graph_recommender.py":
#   rwe_predicted_indices:      A matrix with as many rows as training users, and for each user, it has
#                               the list of predicted indices (i.e., index of items).
#   rwe_evaluations:            Has for each test user values of precision, recall etc.
#   rwe_avg_evaluations:        Has the summary (avareged over all users) evaluations.
#
# It it important to mention that recommendations are based on positive-only feedback!
#
# General remark on the algorithm and file:
#
# The purpose of this file is that the provided recommender algorithm can be accessed the same way
# the other template-based soluitons are. (There is no additional functionality added.)

logger = logging.getLogger(__name__)

# Setup and execute article recommendation algorithm
def runArticleRecommender(userPositions, itemPositions):

    # Read out the parameters for running the recommender algorithm
    b = getParameterB()
    iterations = getParameterIterations()
    type = getParameterType()

    # Setup the timer to stop how long it takes to run the recommender algorithm
    t0 = time.time()

    # Retrieve trained model and pass on the input parameters to the recommender function
    articleRecommendations = MODEL_RWE.train_RWE_ideology(userPositions, itemPositions, b, iterations, type)

    # Stop and measure the time it takes to perform the article recommendation proces for benchmaking
    trainTime = time.time() - t0

    print("Total recommending time:\n" + str(trainTime))

    return articleRecommendations

# Set algrithm parameters according to the provided function arguments
def LoadModel(b, iterations, type):
    
    setParameterB(b)
    setParameterIterations(iterations)
    setParameterType(type)

# Given the list of user scores and article scores create recommendations using the previously trained model
def Predict(userPositions, itemPositions):

    articleRecommendations = runArticleRecommender(userPositions, itemPositions)
    return articleRecommendations

# Train the model using the binary UxI matrix (user x news articles) 
def Train(trainingDataset):

    # Setup the timer to stop how long it takes to run the recommender algorithm
    t0 = time.time()

    # Train the model and store it globally for it to be used in the recommendation part later on
    global MODEL_RWE
    MODEL_RWE = graphRecommender.GraphRec(trainingDataset)

    # Stop and measure the time it takes to perform the article recommendation proces for benchmaking
    trainTime = time.time() - t0
    print("Total training time:\n" + str(trainTime))
