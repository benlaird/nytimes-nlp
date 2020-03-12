# How to Predict the popularity of NY Times Opinion pieces
## Ben Johnson-Laird

## Goal & Data
Goal: to help editors and writers know what articles the public might like.

Data -  Loaded one month’s Opinion pieces from Jan 12th - approx 170 articles.

For each article captured:
* Full text
* The number of user comments (popularity measure)
* Dependent variable is derived from # of user comments


Used number of user comments as a proxy for what counts as a popular article. 

## Initial Model - Bag of words (TF-IDF)
Defining popular as being in the Top N% (by comment count)]

Predictor top N%  | F1 micro score
------------ | -------------
N = 50% | 75%
N = 40% | 55% but everything classified as unpopular!


## Word2vec, GloVe & document classification

Word2vec (2013) created by Tomas Mikolov el al. at Google.  The output represents a word as  a multidimensional vector typically at least 50 dimensions. Each dimension encodes some semantic features. 


Words with similar meaning tend to cluster together in the N-dimensional space.


GloVe (Stanford) provides pre-trained word vectors. This work used 50 & 300 dimensional vectors trained on Wikipedia & Gigaword.


Document classification 

Weight of a document is the mean vector of the weighted word vectors in the document.  The weights are are derived from inverse document frequency (IDF)

##Word2vec results using GloVe (50d) & tf-idf
5-fold cross-validated results, training set: 138, hold out: 35

Predictor top N% popular Op Eds | F1 micro score | Popular Op Eds (% of hold out test)
------------ | ------------- | -------------
N = 40% | 68% | 37 - 40%
N = 30% | 79% | 29 - 31%
N = 25% | 79% | 23 - 26%

Predictive ability peaks where popularity is defined as the top 25% of articles by user comments

## Conclusion & Future Steps
It’s clear that the meaning of an article it critical to predicting reader popularity. Word2vec performs significantly better than a pure-syntactic bag of words approachFrom a rough analysis it’s clear that the reaction to an article can elicit strong reader reaction in the comments. For example comments like the following were highly liked:

Snippet from first two sentences of comment | Number of recommendations
----------------------------------- | ------------- 
This is the way dictators rise. | 8,349
..as if she couldn't formulate intelligent, informed questions; … he seemed to become angry that she didn't fit his stereotype of her. | 6,640
It's appalling that you would cite the "thinnest evidentiary record"... | 5.333


## Acknowledgements
Nadbor Drozd - Text Classification With Word2Vec

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation

Chris Manning. Natural Language Processing with Deep Learning. https://www.youtube.com/watch?v=OQQ-W_63UgQ

