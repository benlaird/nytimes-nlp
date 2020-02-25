import time
import requests
import json
from collections import namedtuple

from datetime import datetime, timedelta

import timesArticle
import timesComment
import getDataFromNYTimesAPI

# Read articles for day
# Plot # of comments
# Calculate percentiles over training period
# Display number of votes in each percentiles

global_start_date = "2020-01-12"
global_end_date = "2020-02-10"


def top_n_comments_by_recommendations(start_date, end_date):
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

        filename_for_articles = f"data/trainingArticles-{start_date}.json"
        filename_for_comments = f"data/trainingComments-{start_date}.json"
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

        filename_for_articles = f"data/trainingArticles-{curr_day}.json"
        filename_for_comments = f"data/trainingComments-{curr_day}.json"
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


def code_data_by_quantile(start_date, end_date):
    """
     :param end_date:
     :param start_date:
     :return:  add the quantile prediction to each article
     """
    """
        Code 4 values: 
            < Q1 == 4, >= Q1 & < Q2 == 3
            >= Q2 & < Q3 == 2
            >= Q3 == 1
        Q1 (25%) quantile of arr :  2908.0  -- 
        Q2 (median) quantile of arr :  8819.0
        Q3 (75%) quantile of arr :  18033.0
    """
    q1 = 190.0
    q2 = 405.0
    q3 = 725.0

    curr_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    arts_to_save = []

    while curr_dt <= end_dt:
        curr_day = datetime.strftime(curr_dt, '%Y-%m-%d')
        print(f"Curr day is: {curr_day}")

        filename_for_articles = f"data/trainingArticles-{curr_day}.json"
        filename_for_comments = f"data/trainingComments-{curr_day}.json"
        print(f"Filename for articles: {filename_for_articles} filename_for_comments: {filename_for_comments}")
        curr_dt = curr_dt + timedelta(days=1)

        arts = timesArticle.TimesArticle.read_from_json(filename_for_articles)
        timesComment.TimesComment.read_from_json(filename_for_comments)
        print(f"Read {len(arts)} articles for {curr_day}")

        # Find the most popular comments
        for a in arts:
            # Skip articles with zero comments
            if a.num_comments == 0:
                continue
            print(f"{a.web_url} Total comments: {a.num_comments}")
            if a.num_comments < q1:
                a.popularity_quantile = 4
            elif a.num_comments >= q1 and a.num_comments < q2:
                a.popularity_quantile = 3
            elif a.num_comments >= q2 and a.num_comments < q3:
                a.popularity_quantile = 2
            else:
                a.popularity_quantile = 1
            print(a)
            arts_to_save.append(a)

    print(f"Saving: {len(arts_to_save)} articles")
    getDataFromNYTimesAPI.save_articles(arts_to_save, "data2", global_start_date, global_end_date)


# compute_comment_score("2020-01-12", "2020-01-12")
# recs = compute_recommendations_by_article("2020-01-12", "2020-01-12")
# print(recs)

# code_data_by_quantile(global_start_date, global_end_date)

code_data_by_quantile(global_start_date, global_end_date)