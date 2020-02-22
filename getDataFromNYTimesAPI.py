# used this and the function clean urls
import time
import requests
import json
from collections import namedtuple

from datetime import datetime, timedelta

from urllib.parse import quote_plus

import timesArticle
import timesComment
import config

api_key = config.api_key


def get_url(q, begin_date, end_date, page):
    url = (
        f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q={q}&begin_date={begin_date}&end_date={end_date}&page={page}&api-key={api_key}")

    print(url)
    return url


def get_opinion_piece(begin_date, end_date, page):
    """
    :param begin_date: e.g. "2020-02-10"
    :param end_date:  e.g. "2020-02-11"
    :param page:
    :return:
    """
    time.sleep(6)
    url = (f'https://api.nytimes.com/svc/search/v2/articlesearch.json?fq=news_desk:("OpEd")&'
           f'begin_date={begin_date}&end_date={end_date}&'
           f'page={page}&'
           f'&api-key={api_key}')
    r = requests.get(url)
    json_data = json.loads(r.text)
    hits = json_data['response']['meta']['hits']
    return json_data, hits


def get_comments_for_article(article_url):
    """
    :param article_url:
    :param page_offset: must be zero or a multiple of 25

    """
    page_offset = 0
    comments_read = 0
    comments_total = None

    print(f"Getting comments for: {article_url}")
    while comments_total is None or comments_read < comments_total:
        time.sleep(6)
        url = (f'https://api.nytimes.com/svc/community/v3/user-content/url.json?'
               f'url={article_url}&'
               f'offset={page_offset}&'
               f'api-key={api_key}')

        r = requests.get(url)
        json_data = json.loads(r.text)
        comments_returned = json_data['results']['totalParentCommentsReturned']
        if not comments_total:
            comments_total = json_data['results']['totalParentCommentsFound']
            if not comments_total:
                # This should never happen
                comments_total = 0
            print(f"Comments_total: {comments_total}")

        # Create the TimesComment objects
        for c in json_data['results']['comments']:
            comm_obj = timesComment.TimesComment(article_url, c['commentID'], c['userID'], c['commentTitle'],
                                                 c['commentBody'], c['recommendations'], c['replyCount'])

        comments_read += comments_returned
        print(f"Page offset: {page_offset} comments_returned: {comments_returned}")
        print(f"Comments read: {comments_read}")
        if page_offset == 0:
            page_offset = 25
        else:
            page_offset += 25
    # comments_total & comments_read should match
    print(f"comments_total: {comments_total} comments_read: {comments_read}")


def get_op_eds_for_date(date_str):
    """
    Get all the op eds along with their comments for a particular date
    :param date_str:
    :return:
    """
    results_processed = 0
    page = 0
    hits = None

    while hits is None or results_processed < hits:

        # TODO loop over any pages to get all the results
        res, hits_returned = get_opinion_piece(date_str, date_str, page)
        if hits is None:
            hits = hits_returned
        articles = res['response']['docs']

        results_processed += len(articles)

        for art in articles:
            if timesArticle.TimesArticle.article_in_list(art['web_url']):
                print(f"Article with abstract: {art['abstract']} already in list")
                continue
            # Parse JSON into an object with attributes corresponding to dict keys.
            # x = json.loads(res, object_hook=lambda d: namedtuple('X', art.keys())(*art.values()))
            print(art['abstract'])
            art_obj = timesArticle.TimesArticle(art['_id'], art['web_url'], art['lead_paragraph'], art['headline'],
                                                art['pub_date'], art['news_desk'], art['keywords'])
            art_obj.set_article_text()
            # Get all the comments for the article
            get_comments_for_article(art['web_url'])

        page += 1


def iterate_thru_dates():
    start_date = "2020-02-10"
    end_date = "2020-02-14"

    curr_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    while curr_dt <= end_dt:
        curr_day = datetime.strftime(curr_dt, '%Y-%m-%d')
        print(f"Curr day is: {curr_day}")
        curr_dt = curr_dt + timedelta(days=1)


def save_training_data(start_date, end_date):
    """
    1) Get all the articles in the date range
    2) Get all the comments for the articles in the date range
    3) Save all the articles
    4) Save all the comments

    :param end_date:
    :param start_date:
    :return:
    """
    filename_for_articles = f"data/trainingArticles-{start_date}.json"
    filename_for_comments = f"data/trainingComments-{start_date}.json"

    curr_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    while curr_dt <= end_dt:
        curr_day = datetime.strftime(curr_dt, '%Y-%m-%d')
        print(f"Curr day is: {curr_day}")
        get_op_eds_for_date(curr_day)
        curr_dt = curr_dt + timedelta(days=1)

    # Save the articles to file
    timesArticle.TimesArticle.save_to_json(filename_for_articles)

    # Save the comments to file
    timesComment.TimesComment.save_to_json(filename_for_comments)


timesArticle.TimesArticle.read_from_json(f"data/trainingArticles-2020-02-10.json")
# get_comments_for_article('https://www.nytimes.com/2020/02/10/opinion/trillion-trees-trump-climate.html')

# timesComment.TimesComment.read_from_json('data/testComments.json')
# get_comments_for_article('https://www.nytimes.com/2020/02/10/opinion/clean-water-act-trump.html')

# get_op_eds_for_date("2020-02-18")

# get_training_data("2020-01-18", "2020-01-18")
print("Got here")
