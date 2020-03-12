# used this and the function clean urls
import time
import requests
import json
from collections import namedtuple

from datetime import datetime, timedelta
import re
from urllib.parse import quote_plus
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")

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
    new_articles = []

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
            new_articles.append(art_obj)

        page += 1
    return new_articles


def iterate_thru_dates():
    start_date = "2020-02-10"
    end_date = "2020-02-14"

    curr_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    while curr_dt <= end_dt:
        curr_day = datetime.strftime(curr_dt, '%Y-%m-%d')
        print(f"Curr day is: {curr_day}")
        curr_dt = curr_dt + timedelta(days=1)


def save_training_data(start_date):
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

    curr_day = datetime.strftime(curr_dt, '%Y-%m-%d')
    print(f"Curr day is: {curr_day}")
    arts = get_op_eds_for_date(curr_day)

    # Save the articles to file
    timesArticle.TimesArticle.save_to_json(filename_for_articles, filename_for_comments, arts)

    # Save the comments to file
    # timesComment.TimesComment.save_to_json(filename_for_comments)


# timesArticle.TimesArticle.read_from_json(f"data/trainingArticles-2020-02-10.json")
# get_comments_for_article('https://www.nytimes.com/2020/02/10/opinion/trillion-trees-trump-climate.html')

# timesComment.TimesComment.read_from_json('data/testComments.json')
# get_comments_for_article('https://www.nytimes.com/2020/02/10/opinion/clean-water-act-trump.html')

# get_op_eds_for_date("2020-02-18")

# save_training_data("2020-01-12")

"""
article = timesArticle.TimesArticle._article_map[new_comment.article_url]
if article:
    article.add_comment(new_comment)
else:
    raise Exception(f"No article for comment_id: {new_comment.comment_id}"
                    f" article_url: {new_comment.article_url}")
"""


def clean_up_files(start_date, end_date=None):
    """
    For each day, read in the file.. Extract just the data for today, both the comments
    and the article. And then resave them.
    :return:
    """
    if end_date is None:
        end_date = start_date

    curr_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    while curr_dt <= end_dt:
        curr_day = datetime.strftime(curr_dt, '%Y-%m-%d')
        url_day = datetime.strftime(curr_dt, '%Y/%m/%d')
        print(f"Curr day is: {curr_day}")

        old_dir = "data.old/"
        new_dir = "data/"
        filename_for_articles = f"trainingArticles-{curr_day}.json"
        filename_for_comments = f"trainingComments-{curr_day}.json"

        arts = timesArticle.TimesArticle.read_from_json(f"{old_dir}/{filename_for_articles}")
        comments = timesComment.TimesComment.read_from_json(f"{old_dir}{filename_for_comments}")

        clean_arts = []
        clear_arts_urls = []
        found = False
        for a in arts:
            pub_date = a.pub_date[0:10]
            if curr_day == pub_date:
                print(f"Found: {a.web_url}")
                clean_arts.append(a)
                clear_arts_urls.append(a.web_url)
                found = True
            else:
                print(f"skipping: {a.web_url}")

        clean_comments = []
        for c in comments:
            if c.article_url in clear_arts_urls:
                clean_comments.append(c)

        print(f"Total articles: {len(arts)} clean articles: {len(clean_arts)}")
        print(f"Total comments: {len(comments)} clean comments: {len(clean_comments)}")

        # Save the articles to file
        timesArticle.TimesArticle.save_to_json(f"{new_dir}/{filename_for_articles}", clean_arts)

        # Save the comments to file
        timesComment.TimesComment.save_to_json(f"{new_dir}{filename_for_comments}", clean_comments)


def save_articles(articles, new_dir, start_date, end_date=None):

    if end_date is None:
        end_date = start_date

    curr_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    while curr_dt <= end_dt:
        curr_day = datetime.strftime(curr_dt, '%Y-%m-%d')
        url_day = datetime.strftime(curr_dt, '%Y/%m/%d')
        print(f"Curr day is: {curr_day}")

        filename_for_articles = f"trainingArticles-{curr_day}.json"

        clean_arts = []
        for a in articles:
            pub_date = a.pub_date[0:10]
            if curr_day == pub_date:
                clean_arts.append(a)

        # Save the articles to file
        timesArticle.TimesArticle.save_to_json(f"{new_dir}/{filename_for_articles}", clean_arts)

        curr_dt = curr_dt + timedelta(days=1)


def read_articles(start_date, end_date=None):
    """
    For each day, read in the file.. Extract just the data for today, both the comments
    and the article. And then resave them.
    :return:
    """
    if end_date is None:
        end_date = start_date

    timesArticle.TimesArticle._articles = []
    curr_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    all_arts = []
    running_total = 0

    while curr_dt <= end_dt:
        curr_day = datetime.strftime(curr_dt, '%Y-%m-%d')
        url_day = datetime.strftime(curr_dt, '%Y/%m/%d')
        print(f"Curr day is: {curr_day}")

        filename_for_articles = f"trainingArticles-{curr_day}.json"
        filename_for_comments = f"trainingComments-{curr_day}.json"

        arts = timesArticle.TimesArticle.read_from_json(f"{config.global_data_read_dir}/{filename_for_articles}")
        comments = timesComment.TimesComment.read_from_json(f"{config.global_data_read_dir}/{filename_for_comments}")
        running_total += len(arts)
        print(f"Total articles: {len(arts)}")
        print(f"Running total: {running_total}")
        print(f"Total comments: {len(comments)}")

        curr_dt = curr_dt + timedelta(days=1)

    return timesArticle.TimesArticle._articles


def read_data():
    arts = read_articles("2020-01-12", "2020-02-10")
    return arts


def clean_data(arts):
    global_num_replaces = 0
    num_replaces = 0
    clean_res = []
    p = re.compile("^[\s]*Advertisement[\s]*Supported[\s]*by")
    # re.DOTALL means match newlines
    p2 = re.compile("If[\s]*you’re[\s]*interested[\s]*in[\s]*talking.*", re.DOTALL)
    p3 = re.compile("The[\s]*Times[\s]*is[\s]*committed.*", re.DOTALL)
    clean_res = [p, p2, p3]
    for a in arts:
        num_replaces = 0
        for clean_re in clean_res:
            new_ft = clean_re.sub("", a.full_text)
            if new_ft != a.full_text:
                a.full_text = new_ft
                num_replaces += 1
                global_num_replaces += 1
        print(f"article_id {a.article_id} num_replaces: {num_replaces}")

    print(f"Clean up, global num replacements: {global_num_replaces}")


def all_article_text(arts):
    docs = [a.full_text for a in arts]
    return docs


def read_and_tokenize(quantile_cut):
    """
    :param quantile_cut: how many quantiles from top to consider as "popular" i.e. as the "1" category
    :return:
    """
    arts = read_data()
    # getDataFromNYTimesAPI.clean_data(arts)
    quantile_cut
    dependent_var = [1 if (a.popularity_quantile <= quantile_cut) else 0 for a in arts]
    docs = all_article_text(arts)
    # docs = [a.lead_paragraph for a in arts]  #Top model is mult_nb F1 == 0.2838

    nlp.vocab["Mr."].is_stop = True
    nlp.vocab["Ms."].is_stop = True
    nlp.vocab["Mrs."].is_stop = True

    tokenized_docs = []
    for d in docs:
        doc = nlp(d)
        tokens = []
        for token in doc:
            if not (token.is_punct | token.is_space | token.is_stop):
                # print("{:<12}{:<10}{:<10}".format(token.text, token.pos, token.dep))
                tokens.append(token.text)
        tokenized_docs.append(tokens)

    X = tokenized_docs
    y = dependent_var
    return X, y, arts

# arts = read_articles(config.global_start_date, config.global_end_date)
# print(len(arts))

