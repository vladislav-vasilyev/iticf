from math import sqrt

import math
import datetime
import pandas as pd


def normalize_by_users(data):
    avg = data.mean(0)
    normalized = data.sub(other=avg, axis=1)
    return normalized


def normalize_by_films(data):
    avg = data.mean(1)
    normalized = data.sub(other=avg, axis=0)
    return normalized


def cosine_similarity(data):
    # calculate numerator
    data_transposed = data.transpose()
    numerator = data.dot(data_transposed)
    # calculate denominator
    # element-wise multiplication
    data_pow2 = data * data
    data_sum = data_pow2.sum(1)
    data_sqrt = data_sum.apply(sqrt).to_frame()
    data_denominator = data_sqrt.dot(data_sqrt.transpose())
    # calculate similarity table
    similarity = numerator / data_denominator
    return similarity


def pearson_similarity(data):
    # calculate numerator
    data_transposed = data.transpose()
    data_cov = data_transposed.cov()
    # calculate denominator    
    data_std = data.std(1).to_frame()
    denominator = data_std.dot(data_std.transpose())
    # element-wise division
    result = data_cov / denominator
    return result


def recommend(pv_data, similarity, pv_test):
    # rix = sum(sij * rjx) / sum(sij)
    data = pv_data.loc[pv_test.index]
    # create a binary matrix
    data_bin = data / data
    data = data.fillna(0)
    data_bin = data_bin.fillna(0)

    data_ibs = similarity.loc[pv_test.index].loc[:, pv_test.index].fillna(0)
    data_ibs[data_ibs < 0] = 0

    numerator = data_ibs.dot(data)
    denominator = data_ibs.dot(data_bin)
    result = numerator / denominator
    return result


def recommend_weighted_avg(pv_data, similarity, pv_test):
    # rix = bxi + sum(data_ibs * (rjx - bxj)) / sum(sij)
    # rix = baseline + sum(data_ibs * (rjx - bxj)) / sum(data_ibs)
    data = pv_data.loc[pv_test.index]
    # create a binary matrix
    data_bin = data / data
    # calculate film mean rating
    films_mean = data.mean(1)
    users_mean = data.mean(0)
    # create zero-initialized matrix for 
    zeros = pd.DataFrame(index=films_mean.index, columns=users_mean.index)
    zeros = zeros.fillna(0)
    # --- old approach (my own) ---
    # films_std = data.std(1)
    # users_std = data.std(0)
    # user_deviation = zeros.add(other=users_std, axis='columns') -\
    #     zeros.add(other=films_std, axis='index')
    # baseline = films_mean + films_std
    # baseline = user_deviation.add(other=baseline, axis='index')
    # --- new approach (BellKor) ---
    f_mean = films_mean.mean()
    user_deviation = zeros.add(other=users_mean, axis='columns') - f_mean
    films_deviation = zeros.add(other=films_mean, axis='index') - f_mean
    baseline = f_mean + user_deviation + films_deviation
    
    data = data - baseline
    data = data.fillna(0)
    data_bin = data_bin.fillna(0)

    data_ibs = similarity.loc[pv_test.index].loc[:, pv_test.index].fillna(0)
    data_ibs[data_ibs < 0] = 0

    numerator = data_ibs.dot(data)
    denominator = data_ibs.dot(data_bin)
    result = numerator / denominator
    result += baseline
    return result


def give_recommendations(test, recom):
    rmse = 0
    rmse_num = 0
    unpredicted = 0
    predicted = []
    print(datetime.datetime.now())
    # give recommendations to users
    for i, row in test.iterrows():
        user_id = row['user_id']
        # id of the film for which we want to get recommendations
        film_id = row['item_id']
        # real rating of the film
        real_rating = row['rating']
        
        if film_id not in recom.index:
            predicted.append(0)
            unpredicted += 1
            continue
        # predict rating for the specified film
        predicted_rating = recom.loc[film_id].loc[user_id]
        predicted.append(predicted_rating)
        if not math.isnan(predicted_rating):
            rmse += pow(real_rating - predicted_rating, 2)
            rmse_num += 1
        else:
            unpredicted += 1
        
    print(datetime.datetime.now())
    rmse = sqrt(rmse / rmse_num)
    print(unpredicted)
    print(rmse)
    return predicted


# --- Read Data --- #
data = pd.read_csv(filepath_or_buffer='u1.base', sep='\s+')  # , header=None u1.base

# --- Start Item Based Recommendations --- #
# pivot the data thus the film names will be the column names
pv_data = data.pivot('item_id', 'user_id', 'rating')
# pv_data.to_csv(path_or_buf='pv_data.csv', sep=';', decimal=',')

# normalize by users
# normalized = normalize_by_users(pv_data)
# normalize by films
normalized = normalize_by_films(pv_data)
normalized = normalized.fillna(0)
# normalized.to_csv(path_or_buf='normalized.csv', sep=';', decimal=',')

# calculate similarity table using cosine distance
data_ibs = cosine_similarity(normalized)
# data_ibs = pearson_similarity(normalized)
# data_ibs.to_csv(path_or_buf='data_ibs.csv', sep=';', decimal=',')

# --- Test Recommendation System --- #
# load test info
test = pd.read_csv(filepath_or_buffer='u1.test', sep='\s+')
pv_test = test.pivot('item_id', 'user_id', 'rating')

# recom = recommend(pv_data, data_ibs, pv_test)
# predicted = give_recommendations(test, recom)

recom = recommend_weighted_avg(pv_data, data_ibs, pv_test)
predicted = give_recommendations(test, recom)

# output results of the prediction to the specified file
test.insert(loc=3, column='predicted', value=predicted)
# test.to_csv(path_or_buf='res.csv', sep=';', decimal=',')
