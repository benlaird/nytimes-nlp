from enum import Enum

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
from sklearn.model_selection import cross_val_score, train_test_split

import numpy as np
import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter

import getDataFromNYTimesAPI

QUANTILE_CUT = 6


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


def save_tensor_vectors_old(vectors, metadata):
    writer = SummaryWriter("tensor_vectors")
    writer.add_embedding(vectors, metadata)
    writer.close()


def save_tensor_vectors(vectors, idx, arts):
    md = []

    df = pd.DataFrame(vectors)
    for i in idx:
        pop_quartile = ((arts[i].popularity_quantile - 1) // 5) + 1
        md.append(
            [f"{pop_quartile}: {arts[i].headline['main']}", i, pop_quartile, arts[i].lead_paragraph])
    print(md)

    meta_df = pd.DataFrame(md, columns=['label', 'index', 'popularity', 'lead para'])

    df.to_csv("tensor_vectors/tensors.tsv", header=False, index=False, sep="\t")
    meta_df.to_csv("tensor_vectors/metadata.tsv", sep="\t", index=False)


X, y, arts = getDataFromNYTimesAPI.read_and_tokenize(QUANTILE_CUT)

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
    ("glove_big_tfidf", etree_glove_big_tfidf), ]

from sklearn.metrics import classification_report, accuracy_score, make_scorer


def classification_report_with_f1_score(y_true, y_pred):
    print(classification_report(y_true, y_pred))  # print classification report
    return f1_score(y_true, y_pred, average="micro")


ActionEnum = Enum('Action', ['run_all_models', 'glove_small_tfidf_cv', 'glove_small_tfidf_vectors'])
action = ActionEnum.glove_small_tfidf_vectors

print(f"Popularity quantile defined as in the top {QUANTILE_CUT / 20 * 100}%")
if action == ActionEnum.run_all_models:
    unsorted_scores = []

    for name, model in all_models:
        print(f"Running model: {name}...")
        score = cross_val_score(model, X, y, cv=5, scoring="f1_micro").mean()
        print(f"Ran model: {name}... score: {score}")
        unsorted_scores.append((name, score))
    scores = sorted(unsorted_scores, key=lambda x: -x[1])
    print(tabulate(scores, floatfmt=".4f", headers=("model", 'score')))
elif action == ActionEnum.glove_small_tfidf_cv:
    name = "glove_small_tfidf"
    model = etree_glove_small_tfidf
    # F1 score
    score = cross_val_score(model, X, y, cv=5, scoring=make_scorer(classification_report_with_f1_score))
    print(f"Ran model: {name}... mean f1 score: {score.mean()}")
elif action == ActionEnum.glove_small_tfidf_vectors:
    indices = range(0, len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, random_state=42,
                                                                             test_size=0.2)

    vectorizer = TfidfEmbeddingVectorizer(glove_small)

    vectorizer.fit(X_train, y_train)
    dtm_tfidf_train = vectorizer.transform(X_train)

    dtm_tfidf_test = vectorizer.transform(X_test)

    # gnb = MultinomialNB()
    clf = ExtraTreesClassifier(n_estimators=200)
    clf.fit(dtm_tfidf_train, y_train)

    nb_train_preds = clf.predict(dtm_tfidf_train)
    y_pred = clf.predict(dtm_tfidf_test)

    print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
    print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='micro')))
    print(classification_report(y_test, y_pred))

    save_tensor_vectors(dtm_tfidf_train, idx_train, arts)

exit(0)
# unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
name = "glove_small_tfidf"
model = etree_glove_small_tfidf
unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean())]
scores = sorted(unsorted_scores, key=lambda x: -x[1])
