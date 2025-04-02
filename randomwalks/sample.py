import datetime
import numpy
import pprint

import article_recommender_default as recommenderDefault
import positions_from_shares as positions

# Number of iteration the article scoring algorithm runs
SCORING_ITERATIONS = 100

# Number of iterations after which the article scoring algorithm prints the loss for benchmarking purposes
SCORING_LOSS = 10

# Create an instance of Bibek's algorithm for calculating scores of news articles
def calculateArticleScores(matrix, userPositions): 
    articleScores = positions.IdeologyFromTweets(matrix, userPositions, do_normalize_matrix = True)
    return articleScores

# Truncate scores for news article such that the minimum value is -1.0 and the maximum value is +1.0
def truncateArticleScores(itemScores):
    
    # According to Bibek it can happen that some articles get a score assigned that is not between the
    # desired range of -1.0 and +1.0. One suggested solution that he approved of is to simply truncate
    # values, i.e., rounding up and down if a score is larger than the min./max. value.
    for score in range(0, len(itemScores)):     
        if itemScores[score] > 1:
            itemScores[score] = 1           
        if itemScores[score] < -1:
            itemScores[score] = -1

    return itemScores

def main():
    
    # Step 0:   Define your user and item data (variable names still reflect news recommendations)
    #

    articleCollection = ["A", "B", "C"]       # List of articles (string, int etc.)
    userScores = [-10,10,0]                 # Will reflect in article scores
    userHistory = [["A"], ["B"], ["B", "C"]]	# Reading history of users, e.g., first user read article "A"

    # Step 1:   Create matrix UxI (user x items, i.e., news articles) with cell entries of either "1" or "0"
    #           ("1" -> user has read article, "0" -> user has not read article).

    # Initialize an empty matrix of dimension UxI to populate with reading metrics
    U = numpy.array(userScores, dtype = numpy.float32)
    I = numpy.zeros((len(U), len(articleCollection)))
    UxI = numpy.c_[I]  

    # Populate matrix with user-based reading data by iterating through all articles for all users
    for user in range(0, len(U)):
        for article in range (0, len(articleCollection)):

            # If a user has read an article insert "1" into the corresponding cell of the UxI matrix
            if (articleCollection[article] in userHistory[user]):
                UxI[user][article] = 1
                
    # Setp 2:   Calcualte matrix UxI*F, where F is a separate matrix storing a user-dependent multiplier for
    #           each news article they read. (I skip this part in the sample code here.)
    UxIF = UxI

    # Step 3:   Train the scoring algorithm based on the weihted UxI*F matrix and assign each article a 
    #           score. This is Bibek's original scoring algorithm for items.

    scoring = calculateArticleScores(numpy.abs(UxIF), U)
    scoring.initialize_variables()
    scoring.train(SCORING_ITERATIONS, SCORING_LOSS)
    itemScores = truncateArticleScores(scoring.getV().flatten())

    print(userScores)
    print(itemScores)
    print(UxIF)

    # Train the prediction model to provide more accurate recommendations
    recommenderDefault.Train(UxIF)

    # Calculate a prediction score for each of the given user-article pairs
    predictions = recommenderDefault.Predict(userScores, itemScores)

    # Row 1 shows predictions for user 1, listing the probabilities in column 1-3 for each article
    print(predictions)

if __name__ == '__main__':
    main()
