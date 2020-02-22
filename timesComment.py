import json
import time
import requests
import textwrap
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus


class TimesComment:
    _comments = []

    def __init__(self, article_url, comment_id, user_id, comment_title, comment_body, recommendations, reply_count):
        self.article_url = article_url
        self.comment_id = comment_id
        self.user_id = user_id
        self.comment_title = comment_title
        self.comment_body = comment_body
        self.recommendations = recommendations
        self.reply_count = reply_count
        TimesComment._comments.append(self)


    @classmethod
    def save_to_json(cls, filename):
        """
        Save all articles to a file as JSON
        :param filename:
        :return:
        """
        results = [a.__dict__ for a in TimesComment._comments]
        with open(filename, 'w') as fp:
            json.dump({"comments": results}, fp, indent=4)

    @classmethod
    def read_from_json(cls, filename):
        """

        :param filename:
        :return:
        """
        with open(filename, "r") as read_file:
            data = json.load(read_file)
        print("Got here")
        for c in data['comments']:
            # headline, pub_date, news_desk, keywords):
            new_comment = TimesComment(c['article_url'], c['comment_id'], c['user_id'], c['comment_title'],
                                                 c['comment_body'], c['recommendations'], c['reply_count'])
