import time

import pandas as pd
import numpy as np
import requests
import json
from collections import namedtuple

from datetime import datetime, timedelta

import timesArticle
import timesComment
import getDataFromNYTimesAPI
import config

# Read articles for day
# Plot # of comments
# Calculate percentiles over training period
# Display number of votes in each percentiles


def top_n_comments_by_recommendations_old(start_date, end_date):
    """
    Returns the top_n comments by recommendations... currently only works for the same start & end_date
    :param end_date:
    :param start_date:
    :return:
    """

    curr_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    while curr_dt <= end_dt:
        curr_day = datetime.strftime(curr_dt, '%Y-%m-%d')
        print(f"Curr day is: {curr_day}")

        filename_for_articles = f"{config.global_data_read_dir}/trainingArticles-{start_date}.json"
        filename_for_comments = f"{config.global_data_read_dir}/trainingComments-{start_date}.json"
        print(f"Filename for articles: {filename_for_articles} filename_for_comments: {filename_for_comments}")
        curr_dt = curr_dt + timedelta(days=1)

        arts = timesArticle.TimesArticle.read_from_json(filename_for_articles)
        timesComment.TimesComment.read_from_json(filename_for_comments)
        print(f"Read {len(arts)} articles for {curr_day}")
        print(arts)

        # Find the most popular comments
        for a in arts:
            total_recommendations = 0
            for c in a.comments:
                total_recommendations += c.recommendations
            print(f"Total recommendations: {total_recommendations}")
            sorted_comms = sorted(a.comments, key=lambda x: x.recommendations, reverse=True)
            # Top 10
            print(sorted_comms[0:10])
        return sorted_comms[0:10]


def top_n_comments_by_recommendations(arts):
    """
    Returns the top_n comments by recommendations...
    :param arts:
    :return:
    """
    top_n_comments = []

    # Find the most popular comments
    for a in arts:
        total_recommendations = 0
        for c in a.comments:
            total_recommendations += c.recommendations
        print(f"Total recommendations: {total_recommendations}")
        sorted_comms = sorted(a.comments, key=lambda x: x.recommendations, reverse=True)
        # Top 10
        top_n = sorted_comms[0:10]
        top_n_comments.extend(top_n)

    return top_n_comments


def compute_total_comments_by_article(start_date, end_date):
    """
    :param end_date:
    :param start_date:
    :return:  an array of all the recommendations by article
    """
    num_comments_list = []
    curr_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    arts_with_comments = 0

    while curr_dt <= end_dt:
        curr_day = datetime.strftime(curr_dt, '%Y-%m-%d')
        print(f"Curr day is: {curr_day}")

        filename_for_articles = f"{config.global_data_read_dir}/trainingArticles-{curr_day}.json"
        filename_for_comments = f"{config.global_data_read_dir}/trainingComments-{curr_day}.json"
        print(f"Filename for articles: {filename_for_articles} filename_for_comments: {filename_for_comments}")
        curr_dt = curr_dt + timedelta(days=1)

        arts = timesArticle.TimesArticle.read_from_json(filename_for_articles)
        timesComment.TimesComment.read_from_json(filename_for_comments)
        print(f"Read {len(arts)} articles for {curr_day}")
        # print(arts)

        for a in arts:
            if a.num_comments == 0:
                continue
            arts_with_comments += 1
            print(f"{a.web_url} Total comments: {a.num_comments}")
            num_comments_list.append(a.num_comments)

    return num_comments_list, arts_with_comments


def code_data_by_quantile(start_date, end_date, num_quantiles, save_files=False):
    """
     :param save_files:
     :param num_quantiles:
     :param end_date:
     :param start_date:
     :return:  add the quantile prediction to each article
     """
    """
        Code 4 values: 
            < Q1 == 4, >= Q1 & < Q2 == 3
            >= Q2 & < Q3 == 2
            >= Q3 == 1
        q1 = 190.0
        q2 = 405.0
        q3 = 725.0
    """
    curr_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    arts_to_save = []

    arts = getDataFromNYTimesAPI.read_data()
    getDataFromNYTimesAPI.clean_data(arts)
    sort_comments = [a.num_comments for a in arts]
    quantiles = pd.qcut(sort_comments, num_quantiles, np.arange(num_quantiles, 0, -1))

    # Find the most popular comments
    i = 0
    for a in arts:
        # Skip articles with zero comments
        if a.num_comments == 0:
            continue
        # if i != 51 and i != 72 and i != 106 and a.popularity_quantile != quantiles[i]:
        #    raise Exception("old and new quantile values are not equal")
        a.popularity_quantile = int(quantiles[i])
        print(f"{a.web_url} Total comments: {a.num_comments} popularity: {a.popularity_quantile}")

        print(a)
        arts_to_save.append(a)
        i += 1

    if save_files:
        print(f"Saving: {len(arts_to_save)} articles")
        getDataFromNYTimesAPI.save_articles(arts_to_save, "data3", config.global_start_date, config.global_end_date)


# compute_comment_score("2020-01-12", "2020-01-12")
# recs = compute_recommendations_by_article("2020-01-12", "2020-01-12")
# print(recs)

code_data_by_quantile(config.global_start_date, config.global_end_date, 20, save_files=True)



