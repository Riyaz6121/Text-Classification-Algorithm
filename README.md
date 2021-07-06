# Text-Classification-Algorithm
This report will show the steps the Parasite team used to offer conclusions and provide a detailed explanation and justification of methods used, while at times summarizing, reflecting and drawing conclusions from the information presented in this report such as Precision, Recall and F1. 

The purpose of this report stems from the need to present a solution to the problem of recommending users find the most interesting articles according to their preferred topics. This project aims to train, tweak and perfect text classification machine learning models and use the model to predict the most relevant news articles for each of the 10 users, that are interested in one of 10 topics: ARTS CULTURE ENTERTAINMENT, BIOGRAPHIES PERSONALITIES PEOPLE, DEFENCE, DOMESTIC MARKETS, FOREX MARKETS, HEALTH, MONEY MARKETS, SCIENCE AND TECHNOLOGY, SHARE LISTINGS, and SPORTS. 

One main issue that needs to be addressed is the datasets feature size. The number of features is much bigger than the number of instances. Some pre-processing using the vectorizer of choice is needed to be done in order to reduce the dimensionality of input and not worsen the accuracy. Another issue that needs to be addressed is filtering out only the relevant words to be the input features and finding a way to remove the misspelled words from the feature space. One technique that can be used is to set some hyperparameters that remove words from the feature space if they don’t at least appear a certain number of times.


#Refer to the REPORT for Analysis

#Go to CODE directory if interested in the code
