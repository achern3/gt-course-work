import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math, sys

# reference: http://online.cambridgecoding.com/notebooks/eWReNYcAfB/implementing-your-own-recommender-systems-in-python-2
header = ['user', 'movie', 'rating', 'time']
data = pd.read_csv('ml-100k/u.data', sep='\t', names=header) # each category is tab-separated
total_num_users = data.user.unique().shape[0] # total number of distinct users
total_num_movies = data.movie.unique().shape[0] # total number of distinct movies

users = data.as_matrix([header[0]]) # get array of all user ids
movies = data.as_matrix([header[1]]) # get array of all movie ids
_, user_counts = np.unique(users, return_counts=True) # get number of times each user rated a movie
_, movie_counts = np.unique(movies, return_counts=True) # get number of times each movie got rated by a user

# plot histograms of number of ratings (bins) vs. count of users/movies
plt.close()
bins = math.ceil(max(user_counts - min(user_counts)) / 50) # use bin sizes of 50 (number of ratings)
plt.hist(user_counts, bins)
plt.xlabel('# of Ratings')
plt.ylabel('User Count')
plt.title('User Ratings Histogram')
plt.grid(True)
plt.show()

bins = math.ceil(max(movie_counts - min(movie_counts)) / 50) # use bin sizes of 50 (number of ratings)
plt.hist(movie_counts, bins)
plt.xlabel('# of Ratings')
plt.ylabel('Movie Count')
plt.title('Movie Ratings Histogram')
plt.grid(True)
plt.show()

# reference: Assignment reference [2], Recommender Systems Handbook p.148-149
def compute_b_i(A, mu):
    lambda_2 = 25
    b_i = np.zeros(A.shape[1]) # b_i is vector with size equal to number of columns in A
    for i in range(A.shape[1]): # iterate over columns of A
        sum_top, count = 0, 0
        for rating in A[:, i]: # iterate over each item in the column
            if rating: # include the item in the calculation formula if it has a rating
                sum_top += (rating - mu)
                count += 1
        b_i[i] = sum_top / (lambda_2 + count)
    return b_i

def compute_b_u(A, mu):
    lambda_3 = 10
    b_i = compute_b_i(A, mu)
    b_u = np.zeros(A.shape[0]) # b_u is vector with size equal to number of rows in A
    for i in range(A.shape[0]): # iterate over rows of A
        sum_top, count = 0, 0
        for j in range(len(A[i, :])): # iterate over each item in the row
            rating = A[i, :][j]
            if rating: # include the item in the calculation formula if it has a rating
                sum_top += (rating - mu - b_i[j])
                count += 1
        b_u[i] = sum_top / (lambda_3 + count)
    return b_u

def compute_b_ui(A, mu, b_u, b_i):
    b_ui = np.zeros((A.shape[0], A.shape[1]))
    for i in range(b_ui.shape[0]):
        for j in range(b_ui.shape[1]):
            b_ui[i, j] = mu + b_u[i] + b_i[j]
    return b_ui

def compute_test_rmse(test_A, predicted_A):
    non_zero_index = test_A != 0 # only care about places in matrix where there is already a user rating
    size_S_test = len(test_A[non_zero_index])
    rmse = math.sqrt(np.sum((test_A[non_zero_index] - predicted_A[non_zero_index])**2) / size_S_test)
    return rmse

def get_A_matrix(data):
    A = np.zeros((total_num_users, total_num_movies), dtype=np.int) # create user x movie matrix
    for line in data.itertuples():
        A[line[1] - 1, line[2] - 1] = line[3]
    return A

def get_b_ui(train_A, test_A):
    mu = np.mean(train_A[train_A != 0]) # average of all available training set ratings
    b_u = compute_b_u(train_A, mu) # get training set b_u
    b_i = compute_b_i(train_A, mu) # get training set b_i
    b_ui = compute_b_ui(test_A, mu, b_u, b_i) # get predicted matrix for test set
    return b_ui

ua_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=header)
ua_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=header)

ua_base_A = get_A_matrix(ua_base)
ua_test_A = get_A_matrix(ua_test)

predicted_A = get_b_ui(ua_base_A, ua_test_A)
rmse = compute_test_rmse(ua_test_A, predicted_A)
print('Baseline predictor test RMSE = {:f}'.format(rmse))

def plot_rmse_histogram(bin_edges, rmse_bins, title): # plot histograms of average user ratings (5 bins) vs. test RMSE
    num_bins = len(rmse_bins)
    width = bin_edges[1] - bin_edges[0] # width of bins equals difference of any pair of edge values
    plt.close()
    plt.bar(bin_edges[:num_bins], rmse_bins, width)
    plt.xlabel('Average Rating')
    plt.ylabel('Test RMSE')
    plt.title('{:s} Test RMSE Histogram'.format(title))
    plt.show()

def get_avg_user_ratings(A):
    avg_user_ratings = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        row = A[i]
        avg_user_ratings[i] = np.mean(row[row != 0])
    return avg_user_ratings

NUM_BINS = 5 # as defined by assignment

avg_user_ratings = get_avg_user_ratings(ua_base_A)
_, bin_edges = np.histogram(avg_user_ratings, bins=NUM_BINS)
rmse_bins = []
for i in range(NUM_BINS):
    subset_index = np.logical_and(avg_user_ratings >= bin_edges[i], avg_user_ratings < bin_edges[i + 1])
    train_A_subset = ua_base_A[subset_index]
    test_A_subset = ua_test_A[subset_index]
    rmse = compute_test_rmse(test_A_subset, get_b_ui(train_A_subset, test_A_subset))
    rmse_bins.append(rmse)

plot_rmse_histogram(bin_edges, rmse_bins, 'Baseline Predictor')

# reference: Assignment references [2], [3]
def compute_similarity_matrix(A):
    # sparse_A = csc_matrix(A)
    similarity_matrix = np.zeros((A.shape[1], A.shape[1]))
    for i in range(A.shape[1]):
        col_i = A[:, i]
        for j in range(i + 1, A.shape[1]):
            col_j = A[:, j]
            common_index = np.where(np.logical_and(col_i != 0, col_j != 0))[0] # array of indices where both items have a rating
            sum_top, sum_bot_i, sum_bot_j = 0, 0, 0
            for index in common_index: # calculate item similarity based on all users that have rated both items
                row = A[index, :]
                avg = np.mean(row[row != 0]) # find average of all ratings for user
                sum_top += ((row[i] - avg) * (row[j] - avg))
                sum_bot_i += (row[i] - avg)**2
                sum_bot_j += (row[j] - avg)**2
            if (sum_bot_i != 0) and (sum_bot_j != 0):
                similarity_matrix[i, j] = sum_top / math.sqrt(sum_bot_i * sum_bot_j)
    return similarity_matrix

def compute_weighted_average_matrix(test_A, train_avg_rating, similarity_matrix, k):
    weighted_average_matrix = np.zeros((test_A.shape[0], test_A.shape[1]))
    for item in range(test_A.shape[1]):
        similarities = np.append(similarity_matrix[:item, item], similarity_matrix[item, item:]) # similarities for item
        similarity_index = np.argsort(-similarities) # get indices of sorted similarity values (descending order)
        similarities = -np.sort(-similarities) # get sorted similarity values (descending order)
        for user in range(test_A.shape[0]):
            if test_A[user, item]: # only consider entries in test matrix that have a rating associated
                count, sum_top, sum_bot = 0, 0, 0
                for i in range(len(similarities)): # loop through the similarity values until k reached
                    similar_item, similarity = similarity_index[i], similarities[i]
                    if (item != similar_item) and (similarity > 0) and test_A[user, similar_item]:
                        count += 1
                        sum_top += (similarity * test_A[user, similar_item])
                        sum_bot += abs(similarity)
                        if count == k: break
                # if no similar items, set value of entry as the global training average rating
                weighted_average_matrix[user, item] = np.clip(sum_top / sum_bot, 1, 5) if (sum_bot != 0) else train_avg_rating
    return weighted_average_matrix

# find global average rating of the training data for cases of two items having no similar users
count, global_sum = 0, 0
for row in ua_base_A:
    for rating in row:
        if rating:
            count += 1
            global_sum += rating
train_avg_rating = global_sum / count

similarity_matrix = compute_similarity_matrix(ua_base_A) # get item-item similarity matrix

print('k-Similarity test RMSE:')
k_list = [1, 2, 3, 5, 10] # as defined by assignment
rmse_list = []
for k in k_list: # loop through different values of k and keep track of the corresponding test rmse
    predicted_A = compute_weighted_average_matrix(ua_test_A, train_avg_rating, similarity_matrix, k)
    rmse = compute_test_rmse(ua_test_A, predicted_A)
    rmse_list.append(rmse)
    print('k = {:d}, RMSE = {:f}'.format(k, rmse))

print('\nMin RMSE = {:f} when k = {:d}'.format(min(rmse_list), k_list[rmse_list.index(min(rmse_list))]))

avg_user_ratings = get_avg_user_ratings(ua_base_A)
_, bin_edges = np.histogram(avg_user_ratings, bins=NUM_BINS)

for k in k_list:
    rmse_bins = []
    for i in range(NUM_BINS):
        subset_index = np.logical_and(avg_user_ratings > bin_edges[i], avg_user_ratings <= bin_edges[i + 1])
        test_A_subset = ua_test_A[subset_index]
        predicted_A = compute_weighted_average_matrix(test_A_subset, train_avg_rating, similarity_matrix, k)
        rmse = compute_test_rmse(test_A_subset, predicted_A)
        rmse_bins.append(rmse)
    plot_rmse_histogram(bin_edges, rmse_bins, 'k = {:d}'.format(k))
