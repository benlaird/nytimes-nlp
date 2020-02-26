from __future__ import print_function

import pyLDAvis
import pyLDAvis.sklearn
# TODO not sure what this is
# pyLDAvis.enable_notebook()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import config
import getDataFromNYTimesAPI


def segment_docs(arts):
    a1 = []
    a2 = []
    a3 = []
    a4 = []

    for a in arts:
        if a.popularity_quantile == 1:
            a1.append(a)
        elif a.popularity_quantile == 2:
            a2.append(a)
        elif a.popularity_quantile == 3:
            a3.append(a)
        elif a.popularity_quantile == 4:
            a4.append(a)

    print(f"{len(a1)}, {len(a2)}, {len(a3)}, {len(a4)}")

    # Docs for quantile 1
    docs1 = [a.full_text for a in a1]
    docs2 = [a.full_text for a in a2]
    docs3 = [a.full_text for a in a3]
    docs4 = [a.full_text for a in a4]
    return docs1, docs2, docs3, docs4


def lda_docs(docs, filename, do_tf=True):
    """
    :param docs:
    :param filename:
    :param do_tf: if True, do plain term frequency, else do tf-idf
    :return:   Saves the visualization file
    """
    tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                    stop_words = 'english',
                                    lowercase = True,
                                    token_pattern = r'\b[a-zA-Z]{3,}\b',
                                    #max_df= 1.0,
                                    #min_df = 0.2
                                   )
    dtm_tf = tf_vectorizer.fit_transform(docs)
    print(dtm_tf.shape)

    tfidf_vectorizer = TfidfVectorizer(strip_accents = 'unicode',
                                    stop_words = 'english',
                                    lowercase = True,
                                    token_pattern = r'\b[a-zA-Z]{3,}\b',
                                    # max_df= 1.0,
                                    # min_df = 0.2
                                    )
    dtm_tfidf = tfidf_vectorizer.fit_transform(docs)
    print(dtm_tfidf.shape)
    print(type(dtm_tfidf))

    # Displays the feature names, should probably lemmatize them
    tfidf_vectorizer.get_feature_names()

    # mds is one of (mmds, pcoa, tsne)
    mds = 'tsne'
    data_vis = None
    if do_tf:
        # for TF DTM
        lda_tf = LatentDirichletAllocation(random_state=0, n_components=5)
        lda_tf.fit(dtm_tf)
        data_vis = pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer, mds=mds)
    else:
        # for TFIDF DTM
        # TODO tf-idf only works with mmds
        lda_tfidf = LatentDirichletAllocation(random_state=0, n_components=5)
        lda_tfidf.fit(dtm_tfidf)
        data_vis = pyLDAvis.sklearn.prepare(lda_tfidf, dtm_tfidf, tfidf_vectorizer, mds=mds)

    # pyLDAvis.show(data_vis)
    pyLDAvis.save_html(data_vis, f"{config.viz_dir}/{filename}")


def main():
    arts = getDataFromNYTimesAPI.read_articles("2020-01-12", "2020-02-10")

    docs1, docs2, docs3, docs4 = segment_docs(arts)
    doc_fn = [(docs1, "docs1"), (docs2, "docs2"), (docs3, "docs3"), (docs4, "docs4")]

    do_tf = True
    for docs, filename in doc_fn:
        if do_tf:
            filename = filename + "_tf"
        else:
            filename = filename + "_tfidf"
        lda_docs(docs, filename, do_tf=True)


main()

