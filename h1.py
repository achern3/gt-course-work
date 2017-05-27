import pandas as pd
import numpy as np
import re, csv, nltk, random
nltk.download("stopwords") # download the nltk "stopwords" corpora
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix
from copy import deepcopy

from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

# function to convert a raw review to a string of words
def review_to_words(raw_review):
    # input is a string (a raw movie review)
    # output is a string (a preprocessed movie review)

    # remove html markup and non-letters
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # convert to lower case and split into individual words
    words = letters_only.lower().split()

    # convert stop words from list to set and remove stop words
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]

    # join words back into one string separated by space, and return the result
    return (" ".join(meaningful_words))

# read in training data as dataframe
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=csv.QUOTE_NONE)

# get total number of reviews based on training dataframe column size
num_reviews = train["review"].size

# send each review to 'review_to_words' function to convert to string of words
clean_train_reviews = []
for i in range(0, num_reviews):
    clean_train_reviews.append(review_to_words(train["review"][i]))

vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

# fit reviews to model/learn the vocabulary, and transform training data into feature vectors (bag of words)
X_counts = vectorizer.fit_transform(clean_train_reviews).toarray()

# modify X_counts by converting all elements to either 0 or 1
X_binary = (X_counts >= 1).astype(int)

# modify X_counts with sklearn tfidf vectorizer
tfid_transformer = TfidfTransformer(smooth_idf=False)
X_tfidf = tfid_transformer.fit_transform(X_counts)

# modify X_binary by deleting 75% of rows corresponding to sentiment=1
random.seed(0)
sentiment_index = [i for i, v in enumerate(train["sentiment"]) if v] # find index of all elements with sentiment=1
delete_list = random.sample(sentiment_index, int(0.75 * len(sentiment_index))) # randomly select 75% of the list
X_binary_imbalance = np.delete(X_binary, delete_list, axis=0) # delete the rows that were selected

# function to calculate distance between two rows in a design matrix
def dist(X, i, j, distance_function="Euclidean"):
    # X is a design matrix, i, j are rows in X, and distance_function specifies the type of distance
    # returns the calculated distance between rows i and j of X

    # use norm function from numpy.linalg module to calculate and return the euclidean distance
    return norm(X[i] - X[j])

# function to get indices of the k closest pairs of a design matrix and the corresponding distances
def topk(X, k):
    # X is a design matrix, k is the number of pairs to return
    # returns the indices of the k closest pairs and the distance values within X

    # use pairwise_distances function from sklearn.metrics.pairwise to calculate the pairwise distance
    dist_matrix = pairwise_distances(X, metric="euclidean", n_jobs=1)

    output = []
    while len(output) < k:
        m = dist_matrix.min()
        min_index = np.where(dist_matrix == m) # returns tuple of arrays with min element indices
        temp_min_list = list(zip(min_index[0], min_index[1])) # zip arrays to form list of element index tuples

        min_list = []
        for x, y in temp_min_list:
            # exclude diagonal elements and same elements in reverse order
            if (x != y) and (x, y) not in min_list and (y, x) not in min_list:
                min_list.append((x, y))

        # if there are too many new elements to add to output, randomly keep which ones to retain
        if (len(min_list) + len(output)) > k:
            min_list = random.sample(min_list, k - len(output))

        # append the distance value to each index pair and add to output list
        for x in range(len(min_list)):
            min_list[x] += (m,)
            output.append(min_list[x])

        # modify the min values so that they do not get picked again
        dist_matrix[min_index] = dist_matrix.max() + 1
    return tuple(output)

# function to print out info as required by assignment (indices, distance, reviews, and labels)
def print_info(reviewTuple, trainingData):
    for review in reviewTuple:
        i, j = review[0], review[1] # first and second elements of tuple correspond to indices i and j
        print("review i: index = {}, first 20 chars = {}, label = {}".format(i, trainingData["review"][i][:20], trainingData["sentiment"][i]))
        print("review j: index = {}, first 20 chars = {}, label = {}".format(j, trainingData["review"][j][:20], trainingData["sentiment"][j]))
        print("distance: {}\n".format(review[2])) # third element of tuple corresponds to distance value

X_counts_out = topk(csr_matrix(X_counts), 3)
print_info(X_counts_out, train) # print info for X_counts

X_binary_out = topk(csr_matrix(X_binary), 3)
print_info(X_binary_out, train) # print info for X_binary

X_tfidf_out = topk(csr_matrix(X_tfidf), 3)
print_info(X_tfidf_out, train) # print info for X_tfidf

train_mod = deepcopy(train) # make deep copy of training data to avoid modification

# drop the rows that were deleted for X_binary_imbalance to maintain consistency
train_mod.drop(train_mod.index[delete_list], inplace=True)
train_mod.reset_index(drop=True, inplace=True)

X_binary_imbalance_out = topk(csr_matrix(X_binary_imbalance), 3)
print_info(X_binary_imbalance_out, train_mod) # print info for X_binary_imbalance

# referenced from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

# function to generate a base pyplot figure with diagonal line drawn
def get_configured_roc_curve_plot():
    plt.figure()
    plt.title("ROC curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
    return plt

# function to plot the roc curve
def plot_roc_curves(data, label, plt, title, color):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0) # split data 80/20
    X_data = csr_matrix(X_train);

    # compute roc auc with X_train (sparse) and y_train
    score_list = []
    for i in range(30): # repeat 30 times
        C = random.uniform(1e-4, 1e4) # randomly pick C value from interval (1e-4, 1e4)

        # use sparse matrix for input data to obtain 5-fold cv roc auc scores
        scores = cross_val_score(LinearSVC(C=C), X_data, y_train, scoring="roc_auc", cv=5, n_jobs=-1)

        score_list.append((sum(scores) / 5, C)) # average scores and store into list as (score, C) tuple

    # find value of C which scored highest (randomly break ties)
    C_list = [(x, y) for x, y in score_list if x == max(score_list)[0]]
    if len(C_list) == 1:
        C = C_list[0][1]
    else:
        C = random.choice(C_list)[1]

    # re-train classifier with best C and same data/label, then run on X_test to calculate scores
    y_score = LinearSVC(C=C).fit(X_data, y_train).decision_function(X_test)

    # compute roc curve data and roc auc value
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # plot roc data on same figure
    plt.plot(fpr, tpr, color=color, lw=2, label="{} (AUC = {:0.2f})".format(title, roc_auc))
    return (plt, C)

# function to output and export classification result file
def output_classification_result(classifier, matrix_title):
    # read in testing data as dataframe
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=csv.QUOTE_NONE)

    # get total number of reviews based on testing dataframe column size
    num_reviews = test["review"].size

    # send each review to 'review_to_words' function to convert to string of words
    clean_test_reviews = []
    for i in range(0, num_reviews):
        clean_test_reviews.append(review_to_words(test["review"][i]))

    # convert test data to bag of words
    test_data_features = vectorizer.transform(clean_test_reviews).toarray()

    # use trained classifer to make sentiment label predictions
    result = classifier.predict(test_data_features)

    # copy results to pandas dataframe with "id" column and "sentiment" column
    output = pd.DataFrame(data={"id":test["id"], "sentiment":result})

    # write output to csv file
    output.to_csv("Bag_of_Words_model_{}.csv".format(matrix_title), index=False, quoting=csv.QUOTE_NONE)

# get a pyplot figure to plot with
plt = get_configured_roc_curve_plot()

# plot X_counts roc curve and output classification result file
plt, C = plot_roc_curves(X_counts, train["sentiment"], plt, "X_counts", "navy")
classifier = LinearSVC(C=C).fit(csr_matrix(X_counts), train["sentiment"])
output_classification_result(classifier, "X_counts")

# plot X_binary roc curve and output classification result file
plt, C = plot_roc_curves(X_binary, train["sentiment"], plt, "X_binary", "orange")
classifier = LinearSVC(C=C).fit(csr_matrix(X_binary), train["sentiment"])
output_classification_result(classifier, "X_binary")

# plot X_tfidf roc curve and output classification result file
plt, C = plot_roc_curves(X_tfidf, train["sentiment"], plt, "X_tfidf", "aqua")
classifier = LinearSVC(C=C).fit(csr_matrix(X_tfidf), train["sentiment"])
output_classification_result(classifier, "X_tfidf")

# plot X_binary_imbalance roc curve and output classification result file
plt, C = plot_roc_curves(X_binary_imbalance, train_mod["sentiment"], plt, "X_binary_imbalance", "deeppink")
classifier = LinearSVC(C=C).fit(csr_matrix(X_binary_imbalance), train_mod["sentiment"])
output_classification_result(classifier, "X_binary_imbalance")

# set legend and show plot figure
plt.legend(loc="lower right")
plt.show()

# referenced from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

classifier = LogisticRegression(random_state=0, n_jobs=1) # instantiate logistic regression classifier
train_sizes=[100, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000] # as specified by assignment

# get train and test scores using learning_curve function
train_sizes, train_scores, test_scores = learning_curve(classifier, csr_matrix(X_counts), train["sentiment"],
                                                        train_sizes=train_sizes, cv=5, n_jobs=1)

# calculate mean and standard deviation values
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# plot the learning curve
plt.figure()
plt.title("Logistic Regression Classifier Learning Curve")
plt.xlabel("Training size")
plt.ylabel("Score")
plt.xlim([min(train_sizes) - 500, max(train_sizes) + 500])
y_lower = min(min(train_scores_mean), min(test_scores_mean)); y_upper = max(max(train_scores_mean), max(test_scores_mean))
plt.ylim([y_lower - 0.1, y_upper + 0.1])
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="lower right")
plt.show()
