import json
import time
import requests
import textwrap
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus

import config

api_key = config.api_key


class TimesArticle:
    _articles = []

    def __init__(self, article_id, web_url, lead_paragraph, headline, pub_date, news_desk, keywords):
        """
        Note: article_text has to be added by the find_by_id call

        :param article_id:
        :param web_url:
        :param lead_paragraph:
        :param headline:
        :param pub_date:
        :param news_desk:
        """

        self.article_id = article_id
        self.web_url = web_url
        self.lead_paragraph = lead_paragraph
        self.headline = headline

        self.num_comments = 0
        self.pub_date = pub_date
        self.news_desk = news_desk
        # keywords is a list of dictionaries, each of the form:
        # [{'name': 'subject', 'value': 'Trees and Shrubs', 'rank': 1, 'major': 'N'}
        # So it's all the value keys we care about
        self.keywords = keywords
        # TODO add byline later
        # Full text is added later by set_article_text
        self.full_text = ""

        # Add to class list
        TimesArticle._articles.append(self)

    def search_by_id(self):
        """
        TODO Not currently used
        :return:
        """
        web_url_encode = quote_plus(self.web_url)

        url = (f'https://api.nytimes.com/svc/search/v2/articlesearch.json?fq=web_url%3A"{web_url_encode}"'
               f"&api-key={api_key}")

        r = requests.get(url)
        json_data = json.loads(r.text)
        return json_data

    def set_article_text(self):
        """
        Using the article's web_url look up the full text
        :return:
        """
        content = []
        time.sleep(6)
        page = requests.get(self.web_url)
        soup = bs(page.content, 'html.parser')
        everything = soup.find_all(['p', 'h1', 'h2', 'h3'])
        for p in everything:
            content.append(p.get_text())
        self.full_text = " ".join(content)
        self.full_text = textwrap.fill(self.full_text, width=80, drop_whitespace=False)
        return self.full_text

    @classmethod
    def article_in_list(cls, web_url):
        """
        Searches for the web_url in the list of the articles, if found then returns the TimesArticle
        :param web_url:
        :return: if found the TimesArticle, else None
        """
        for a in TimesArticle._articles:
            if a.web_url == web_url:
                return a
        return None

    @classmethod
    def save_to_json(cls, filename):
        """
        Save all articles to a file as JSON
        :param filename:
        :return:
        """
        results = [a.__dict__ for a in TimesArticle._articles]
        with open(filename, 'w') as fp:
            json.dump({"articles": results}, fp, indent=4)

    @classmethod
    def read_from_json(cls, filename):
        """

        :param filename:
        :return:
        """
        with open(filename, "r") as read_file:
            data = json.load(read_file)
        print("Got here")
        for a in data['articles']:
            # headline, pub_date, news_desk, keywords):
            new_art = TimesArticle(a['article_id'], a['web_url'], a['lead_paragraph'], a['headline'], a['pub_date'],
                                  a['news_desk'], a['keywords'])
            new_art.num_comments = a['num_comments']
            new_art.full_text = a['full_text']


