from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
import struct

import getDataFromNYTimesAPI
import visualizeLDA

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
        else:
            self.dim = 0

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
            print(self.dim)
        else:
            self.dim = 0

    def fit(self, X, y):
        # tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf = TfidfVectorizer(analyzer=lambda x: x, min_df=2, max_df=.95)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

"""
 np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X"""


arts = getDataFromNYTimesAPI.read_data()
# getDataFromNYTimesAPI.clean_data(arts)
quantile_cut = 6
dependent_var = [1 if (a.popularity_quantile <= quantile_cut) else 0 for a in arts]
docs = visualizeLDA.all_article_text(arts)
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

X, y = np.array(X), np.array(y)
print("total examples %s" % len(y))

GLOVE_6B_50D_PATH = "glove.6B/glove.6B.50d.txt"
GLOVE_840B_300D_PATH = "glove.6B/glove.6B.300d.txt"
encoding = "utf-8"

glove_small = {}
all_words = set(w for words in X for w in words)
with open(GLOVE_6B_50D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode(encoding)
        if word in all_words:
            nums = np.array(parts[1:], dtype=np.float32)
            glove_small[word] = nums

glove_big = {}
with open(GLOVE_840B_300D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode(encoding)
        if word in all_words:
            nums = np.array(parts[1:], dtype=np.float32)
            glove_big[word] = nums

# train word2vec on all the texts - both training and test set
# we're not using test labels, just texts so this is fine
model = Word2Vec(X, size=300, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

print(len(all_words))

# start with the classics - naive bayes of the multinomial and bernoulli varieties
# with either pure counts or tfidf features
mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
mult_nb_tfidf = Pipeline(
    [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
# SVM - which is supposed to be more or less state of the art
# http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline(
    [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])

# Extra Trees classifier is almost universally great, let's stack it with our embeddings
etree_glove_small = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_small)),
                              ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_small_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)),
                                    ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_big = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_big)),
                            ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_big_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_big)),
                                  ("extra trees", ExtraTreesClassifier(n_estimators=200))])

etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                      ("extra trees", ExtraTreesClassifier(n_estimators=200))])
# etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
#                     ])
etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                            ("extra trees", ExtraTreesClassifier(n_estimators=200))])
# etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
#                           ])

all_models = [
    ("glove_small_tfidf", etree_glove_small_tfidf),
    ("mult_nb", mult_nb),
    ("w2v", etree_w2v),
    ("w2v_tfidf", etree_w2v_tfidf),
("mult_nb_tfidf", mult_nb_tfidf),
("bern_nb", bern_nb),
("bern_nb_tfidf", bern_nb_tfidf),
("svc", svc),
("svc_tfidf", svc_tfidf),
("glove_small", etree_glove_small),
("glove_big", etree_glove_big),
("glove_big_tfidf", etree_glove_big_tfidf),]


from sklearn.metrics import classification_report, accuracy_score, make_scorer


def classification_report_with_f1_score(y_true, y_pred):
    print(classification_report(y_true, y_pred))  # print classification report
    return f1_score(y_true, y_pred, average="micro")


run_all_models = False
print(f"Popularity quantile defined as in the top {quantile_cut / 20 * 100}%")
if run_all_models:
    unsorted_scores = []

    for name, model in all_models:
        print(f"Running model: {name}...")
        score = cross_val_score(model, X, y, cv=5, scoring="f1_micro").mean()
        print(f"Ran model: {name}... score: {score}")
        unsorted_scores.append((name, score))
    scores = sorted(unsorted_scores, key=lambda x: -x[1])
    print(tabulate(scores, floatfmt=".4f", headers=("model", 'score')))
else:
    name = "glove_small_tfidf"
    model = etree_glove_small_tfidf
    # F1 score
    score = cross_val_score(model, X, y, cv=5, scoring=make_scorer(classification_report_with_f1_score))
    print(f"Ran model: {name}... mean f1 score: {score.mean()}")

exit(0)
# unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
name = "glove_small_tfidf"
model = etree_glove_small_tfidf
unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean())]
scores = sorted(unsorted_scores, key=lambda x: -x[1])
