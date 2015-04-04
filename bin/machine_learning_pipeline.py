'''
author: Jiaxuan Wang
descrition:
machine learning pipeline
'''
# models 
from sklearn.lda import LDA
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
ofrom sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.hmm import GMMHMM
import network
# feature selection
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
# model/parameter searching
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
# auxiliary
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import TransformerMixin
# essential
import pickle
import os
import os.path
import sys
import pandas as pd
import numpy as np
from time import time

class DenseTransformer(TransformerMixin):
    '''
    Credit to
    http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
    '''
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

def extract_label(filename):
    if filename.find('mix') != -1: return 'mix'
    elif filename.find('joke') != -1:  return 'joke'
    else: raise Exception('filename does not contain either joke or mix')

def nn_transform(y):
    return [[[1],[0]] if label==0 else [[0],[1]] for label in y]

def neural_net_transfrom(X, y):
    '''output (x, y) pairs'''
    return [(x.reshape((X.shape[1],1)), label) for x, label in zip(X, y)]

def train_val(clf, X, y):
    # stratified k-fold cross validation
    print "doing cross validation for %s" % clf
    cv = StratifiedKFold(y, n_folds=3)    
    print cross_val_score(clf,X,y,scoring='accuracy',cv=cv)
    return clf

if __name__ == '__main__':

    clf_SGD = Pipeline([
            ('vect', CountVectorizer(decode_error='ignore')),
            ('clf', SGDClassifier(loss='hinge', penalty='l1')), # lasso linear
            ])

    clf_NB = Pipeline([
            ('vect', CountVectorizer(decode_error='ignore')),
            ('clf', MultinomialNB(alpha=1.0)),
            ])

    clf_NB_tfidf = Pipeline([
            ('vect', TfidfVectorizer(decode_error='ignore')),
            ('clf', MultinomialNB(alpha=1.0)),
            ])

    clf_boost = Pipeline([
            ('vect', CountVectorizer(decode_error='ignore')),
            ('to_dense', DenseTransformer()), 
            ('clf', AdaBoostClassifier())
            ])

    clf_GNB = Pipeline([
            ('vect', CountVectorizer(decode_error='ignore')),
            ('to_dense', DenseTransformer()), 
            ('clf', GaussianNB())
            ])

    clf_BNB = Pipeline([
            ('vect', CountVectorizer(decode_error='ignore')),
            ('clf', BernoulliNB())
            ])

    clf_BNB_tfidf = Pipeline([
            ('vect', TfidfVectorizer(decode_error='ignore', stop_words='english')),
            ('clf', BernoulliNB())
            ])

    clf_GNB = Pipeline([
            ('vect', CountVectorizer(decode_error='ignore')),
            ('to_dense', DenseTransformer()), 
            ('clf', GaussianNB())
            ])

    clf_gboost = Pipeline([
            ('vect', CountVectorizer(decode_error='ignore')),
            ('to_dense', DenseTransformer()), 
            ('clf', GradientBoostingClassifier())
            ])

    clf_HMM = Pipeline([
            ('vect', CountVectorizer(decode_error='ignore')),
            ('clf', GMMHMM())
            ]) # mixture of Gaussian hidden markov chain, doesn't apply here

    clf_LDA = Pipeline([
            ('vect', CountVectorizer(decode_error='ignore')),
            ('to_dense', DenseTransformer()), 
            ('clf', LDA())
            ]) # lda, too much memory required abort!

    clf_RF = Pipeline([
            ('vect', CountVectorizer(decode_error='ignore')),
            ('to_dense', DenseTransformer()), 
            ('clf', RandomForestClassifier(n_estimators=10, max_features="auto"))
            ])
    RF_grid = {'clf__n_estimators': [10, 20, 30], 'clf__max_features': [50, 100, 200, 400],
               'clf__max_depth': [50, 100, 150]
               }

    clf_SVM = Pipeline([
            ('vect', CountVectorizer(decode_error='ignore')),
            ('clf', SVC(C=1.0, kernel='rbf'))
            ])
    SVM_grid = [
        {'clf__C': [0.1, 1, 10, 100, 1000], 'clf__kernel': ['linear']},
        {'clf__C': [0.1, 1, 10, 100, 1000], 'clf__gamma': [0.001, 0.0001, 0.01], 'clf__kernel': ['rbf']},
        ]
    
    clf_best_SVM = Pipeline([
            ('vect', CountVectorizer(decode_error='ignore')),
            ('clf', SVC(C=900, gamma=0.06, kernel='rbf'))
            ])  # grid search result C=100, gamma=0.06


    # training part
    if os.path.exists('training_X.npy') and os.path.exists('training_y.npy'):
        print "using existing training file"
        X = np.load('training_X.npy')
        y = np.load('training_y.npy')
    else:
        print "training start"
        training_folder = 'kaggle.training/'
        train_files = os.listdir(training_folder) 
        labels = map(extract_label, train_files)
        X = np.array([open(training_folder+tf).read() for tf in train_files])
        y = labels
        # cache the input
        np.save('training_X',X)
        np.save('training_y',y)

    # feature extraction
    # print "feature extraction start"
    c = CountVectorizer(decode_error='ignore')
    X = c.fit_transform(X)
    X = X.todense()

    # EDA using PCA
    # print "PCA start"
    # pca = PCA()
    # pca.fit(X)

    # use cached PCA for 1000 features
    # print "using cached pca"
    # pca = pickle.load(open('wrong_pca.txt'))
    # pca.transform(X)

    # or load the already saved one
    # print 'get pca transformed X'
    # X = np.load('pcatransformedX.npy')
    
    # For time sake, I just want to use kmeans
    # k = 8
    # if not os.path.exists('kmeans_out_%d.npy' % k):
    #     print "doing kmeans"
    #     km = KMeans(k, copy_x=False)
    #     km.fit_transform(X)
    #     np.save('kmeans_out_%d' % k, X)
    # else:
    #     # or load already saved kmeans
    #     print "using kmeans from cached file"
    #     X = np.load('kmeans_out_%d.npy' % k)
    
    # transform X for neural network
    print 'turn y into 0, 1s'
    y = np.where(y=='joke', 1, 0)
    # validate
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    print "transform X for neural network"
    training_data = neural_net_transfrom(X_train, nn_transform(y_train))
    validation_data = neural_net_transfrom(X_val, y_val)
    # train a neural net
    net = network.Network([X_train.shape[1],30,30, 2])
    net.SGD(training_data, 30, 3, 3.0, test_data=validation_data)

    # get the best clf
    # clf = train_val(clf_best_SVM, X, y) # [ 0.83225658  0.8280637   0.83256404]
    # clf = train_val(clf_BNB_tfidf, X, y) # [ 0.81045224  0.81779368  0.819063  ]
    # clf = train_val(clf_BNB, X, y) # [ 0.81045224  0.81779368  0.819063  ]
    # clf = train_val(clf_NB,  X, y) # [ 0.77814952  0.78559889  0.77867528] 
    # clf = train_val(clf_NB_tfidf,  X, y) # [ 0.77099677  0.77994461  0.77244403]
    # clf = train_val(clf_RF, X, y) # [ 0.76892017  0.77071313  0.7642511 ]
    # clf = train_val(clf_gboost, X, y) # [ 0.73488694  0.73886453  0.73378721]
    # clf = train_val(clf_boost, X, y) # [ 0.72865713  0.73447958  0.72582506]
    # clf = train_val(clf_GNB, X, y) # [ 0.68874019  0.6897069   0.68716824]
    # clf = train_val(clf_SVM, X, y) # [ 0.52434241  0.52411724  0.54731133]
    # clf = train_val(clf_SGD, X, y) # [ 0.50288417  0.50680822  0.50900069]


    # parameter search for SVM
    # clf = GridSearchCV(clf_SVM, SVM_grid, scoring='accuracy', cv=StratifiedKFold(y, n_folds=3), verbose=1)
    # clf.fit(X, y)
    # print("Best parameters set found on development set:")
    # print(clf.best_estimator_)
    # print("Grid scores on development set:")
    # for params, mean_score, scores in clf.grid_scores_:
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean_score, scores.std() / 2, params))
    # print("Detailed classification report:")
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))

    # parameter searching for RF
    # RF_clf = RandomForestClassifier(n_estimators=10, max_features="auto")
    # RF_grid = {'n_estimators': [10, 20, 30], 'max_features': [50, 100, 200, 400],
    #            'max_depth': [50, 100, 150]
    #            } # not searched
    # clf = GridSearchCV(RF_clf, RF_grid, scoring='accuracy', cv=StratifiedKFold(y, n_folds=3), verbose=1)
    # clf.fit(X, y)
    # print("Best parameters set found on development set:")
    # print(clf.best_estimator_)
    # print("Grid scores on development set:")
    # for params, mean_score, scores in clf.grid_scores_:
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean_score, scores.std() / 2, params))

    # parameter searching for gradient boosting
    # params = dict(vect__max_df=[0.5, 1.0],
    #           vect__max_features=[None, 10000, 200000],
    #           vect__ngram_range=[(1, 1), (1, 2)],
    #           tfidf__use_idf=[True, False],
    #           tfidf__norm=['l1', 'l2'],
    #           bernoulli__alpha=[0, .5, 1],
    #           bernoulli__binarize=[None, .1, .5],
    #           bernoulli__fit_prior=[True, False]
    #          )
    # n_iter_search = 100
    # random_search = grid_search.RandomizedSearchCV(pipe, param_distributions=params,
    #                                                n_iter=n_iter_search)

    # # fit it
    # clf.fit(X,y)

    # # testing part
    # test_folder = 'kaggle.test/'  if len(sys.argv) < 2 else sys.argv[1]
    # test_files = np.array(os.listdir(test_folder))
    # X = np.array([open(test_folder+tf).read() for tf in test_files])
    # prediction = clf.predict(X)
    # # output for kaggle
    # t = pd.DataFrame(np.transpose([test_files,prediction]), columns=['File', 'Class'])
    # t.to_csv('predictions.out', index=False)

