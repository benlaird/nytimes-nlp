import json
import re
import time
import requests
import textwrap
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus

# from timesArticle import TimesArticle
import timesArticle


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

        # Store the first sentence
        find = re.compile(r"^(([^\.]*\.){2})")
        search_res = re.search(find, self.comment_body)
        if search_res:
            self.first_sentence = search_res.group(1)
        else:
            self.first_sentence = ""

        TimesComment._comments.append(self)

        article = timesArticle.TimesArticle._article_map[self.article_url]
        if article:
            article.add_comment(self)
        else:
            raise Exception(f"No article for comment_id: {self.comment_id}"
                            f" article_url: {self.article_url}")


    @classmethod
    def save_to_json(cls, filename, comments):
        """
        Save all articles to a file as JSON
        :param comments:
        :param filename:
        :return:
        """
        results = [a.__dict__ for a in comments]
        with open(filename, 'w') as fp:
            json.dump({"comments": results}, fp, indent=4)

    @classmethod
    def read_from_json(cls, filename):
        """

        :param filename:
        :return:
        """
        comments = []
        with open(filename, "r") as read_file:
            data = json.load(read_file)
        print("Got here")
        for c in data['comments']:
            # headline, pub_date, news_desk, keywords):
            new_comment = TimesComment(c['article_url'], c['comment_id'], c['user_id'], c['comment_title'],
                                                 c['comment_body'], c['recommendations'], c['reply_count'])
            comments.append(new_comment)
        return comments

    def __repr__(self):
        # s = f"Comment: {self.article_url}, {self.comment_body[0:30]}," \
        s = f"Comment: {self.article_url}, {self.first_sentence}," \
                    f"Recommendations: {self.recommendations}, Replies:{self.reply_count}  {self.comment_id}, {self.user_id}\n"
        return s

    def __str__(self):
        return self.__repr__()