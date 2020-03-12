import re

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from yellowbrick.classifier import classification_report as yb_class_report, ClassificationReport
from yellowbrick.classifier import ConfusionMatrix

import config
import getDataFromNYTimesAPI
import visualizeLDA


def top_n_features(feature_names, response, top_n=3):
    sorted_nzs = np.argsort(response.data)[:-(top_n + 1):-1]
    print(feature_names[response.indices[sorted_nzs]])


def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    """ Top tfidf features in specific document (matrix row) """
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    """ Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. """
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def count_zero_categories(X_2_3_4_docs):
    count_vectorizer = CountVectorizer(analyzer='word',
                                       strip_accents='unicode',
                                       stop_words='english',
                                       lowercase=True,
                                       # token_pattern=r'\b[a-zA-Z]{3,}\b',
                                       # max_df= 1.0,
                                       # min_df = 0.2
                                       )
    X_cat_2_3_4_counts = count_vectorizer.fit_transform(X_2_3_4_docs)
    vectorizer = TfidfTransformer()
    dtm_tfidf_zero_cat = vectorizer.fit_transform(X_cat_2_3_4_counts)
    feature_names = count_vectorizer.get_feature_names()
    print("*** Top mean zero-cat features ***")
    top_mean_f = top_mean_feats(dtm_tfidf_zero_cat, feature_names)
    print(top_mean_f)
    return dtm_tfidf_zero_cat


def model_naive_bayes(arts):
    use_counts = False
    dependent_var = [a.popularity_quantile if a.popularity_quantile == 1 else 0 for a in arts]
    docs = visualizeLDA.all_article_text(arts)

    X = docs
    y = dependent_var

    indices = range(0, len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, random_state=1,
                                                                             test_size=0.25)

    X_cat_2_3_4_arts = []
    for i in range(0, len(X_train)):
        article = arts[idx_train[i]]
        if article.popularity_quantile != 1:
            X_cat_2_3_4_arts.append(article)
    X_2_3_4_docs = visualizeLDA.all_article_text(X_cat_2_3_4_arts)
    dtm_tfidf_zero_cat = count_zero_categories(X_2_3_4_docs)

    vectorizer = TfidfVectorizer(analyzer='word',
                                 strip_accents='unicode',
                                 stop_words='english',
                                 lowercase=True,
                                 # token_pattern=r'\b[a-zA-Z]{3,}\b',
                                 # max_df=0.6,
                                 # min_df=0.0
                                 )
    dtm_tfidf_train = vectorizer.fit_transform(X_train)

    feature_names = np.array(vectorizer.get_feature_names())

    if False:
        for i in range(0, dtm_tfidf_train.get_shape()[0]):
            response = dtm_tfidf_train[i]
            article = arts[idx_train[i]]
            print(f"Article: {idx_train[i]} popularity: {article.popularity_quantile} {article.lead_paragraph}")
            top_n_features(feature_names, response, 50)

    print(f"Train type: {type(dtm_tfidf_train)} Train shape: {dtm_tfidf_train.shape}")

    dtm_tfidf_test = vectorizer.transform(X_test)

    if False:
        for response in dtm_tfidf_test:
            top_n_features(feature_names, response)

    gnb = MultinomialNB()
    gnb.fit(dtm_tfidf_train, y_train)

    nb_train_preds = gnb.predict(dtm_tfidf_train)
    nb_test_preds = gnb.predict(dtm_tfidf_test)

    nb_train_score = accuracy_score(y_train, nb_train_preds)
    nb_test_score = accuracy_score(y_test, nb_test_preds)

    print("Multinomial Naive Bayes")
    print("Training Accuracy: {:.4} \t\t Testing Accuracy: {:.4}".format(nb_train_score, nb_test_score))
    print("")
    print('-' * 70)
    print("")

    c_m = confusion_matrix(y_test, nb_test_preds)
    print(c_m)

    print("*** Top mean features ***")
    top_mean_f = top_mean_feats(dtm_tfidf_train, feature_names)
    print(top_mean_f)

    if False:
        visualizer = ClassificationReport(gnb)  # classes=classes, support=True)
        visualizer.fit(dtm_tfidf_train, y_train)
        visualizer.score(dtm_tfidf_test, y_test)  # Evaluate the model on the test data
        visualizer.show()  # Finalize and show the figure




def main():
    arts = getDataFromNYTimesAPI.read_data()
    model_naive_bayes(arts)


main()
