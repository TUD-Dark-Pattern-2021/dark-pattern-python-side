# ------------------------------------- ########################################## ------------------------------------
# ------------------------------------- Training Using Machine Learning Algorithms ------------------------------------
# ------------------------------------- ########################################## ------------------------------------

### --------------------------
### ------Import Packages
### --------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
# provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words,
# but also to encode new documents using that vocabulary.
from sklearn.feature_extraction.text import TfidfVectorizer
# Bernoulli Naive Bayes (Similar as  MultinomialNB), this classifier is suitable for discrete data.
# The difference between MultinomialNB and BernoulliNB is that while  MultinomialNB works with occurrence counts,
# BernoulliNB is designed for binary/boolen features, which means in the case of text classification, word occurrence vectores
# (rather than word count vectors) may be more suitable to be used to train and use this classifier.
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
# Evaluation metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
# joblib is a set of tools to provide lightweight pipelining in Python.
# It provides utilities for saving and loading Python objects that make use of NumPy data structures, efficiently.
import joblib

### --------------------------
### ------Dataset Import
### --------------------------

data = pd.read_csv('enriched_data.csv')
# Change the label into strings
data['classification'].replace({0:'Dark',1:'Not_Dark'}, inplace = True)
# Check the target distribution.
print('\nDistribution of the tags:\n{}'.format(data['classification'].value_counts()))
# For later training the model, we should remove the duplicate input to reduce overfitting.
data = data.drop_duplicates(subset="Pattern String")

### --------------------------
### ------Data Preparation
### --------------------------

# organise the predictive feature and the target value.
Y = data['classification']
X = data['Pattern String']

# encode the target values into integers ---- "classification"
encoder = LabelEncoder()
encoder.fit(Y)
y = encoder.transform(Y)
print(y.shape)

# Check the frequency distribution of the training pattern classification with pattern classification names.
(unique, counts) = np.unique(y, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

# get the word count vector of the pattern string to encode the pattern string.
tv = TfidfVectorizer()
x = tv.fit_transform(X)

# save the TfidfVectorizer to disk
joblib.dump(tv, 'presence_TfidfVectorizer.joblib')



### --------------------------------------------------------------------------------------------------
### ------Rough Idea about the performance of different classifiers
### --------------------------------------------------------------------------------------------------
# Four models are tested:
# -- Logistic Regression
# -- Linear Support Vector Machine
# -- Random Forest
# -- Bernoulli Naive Bayes

classifiers = [LogisticRegression(), LinearSVC(), RandomForestClassifier(), BernoulliNB()]
# Calculate the accuracies of different classifiers using default settings.
acc = []
pre = []
cm = []

for clf in classifiers:
    y_pred = cross_val_predict(clf, x, y, cv=5, n_jobs = -1)
    acc.append(metrics.accuracy_score(y, y_pred))
    pre.append(metrics.precision_score(y,y_pred, pos_label=0))
    cm.append(metrics.confusion_matrix(y, y_pred))

# List the accuracies of different classifiers.
for i in range(len(classifiers)):
    print("{} accuracy: {:.3f}".format(classifiers[i],acc[i]))
    print("{} precision: {:.3f}".format(classifiers[i],pre[i]))
    print("Confusion Matrix: {}".format(cm[i]))



### --------------------------------------------------------------------------------------------------
### ------------------------------------- Bernoulli Naive Bayes Classifier Training/ Parameter Tuning
### --------------------------------------------------------------------------------------------------

# setup the Bernoulli Naive Bayes classifier
clf_bnb = BernoulliNB()
# define the combination of parameters to be considered for parameter tuning
param_grid = {'alpha':[0,1], 'fit_prior':[True, False]}
# Run the Grid Search
gs = GridSearchCV(clf_bnb, param_grid, cv=5, verbose = 1, n_jobs = -1)
best_bnb = gs.fit(x,y)

# print the result
scores_df = pd.DataFrame(best_bnb.cv_results_)
scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
print(scores_df[['rank_test_score', 'mean_test_score', 'param_alpha', 'param_fit_prior']])

# print the parameters of the best model
print(best_bnb.best_params_)

# Use the best hyper-parameters to train on the whole dataset.
bnb = best_bnb.best_estimator_.fit(x,y)

# save the model to local disk
joblib.dump(bnb, 'bnb_presence_classifier.joblib')



### --------------------------------------------------------------------------------------------------
### ------------------------------------- Support Vector Machine Classifier Training/ Parameter Tuning
### --------------------------------------------------------------------------------------------------

# setup the Support Vector Machine classifier
clf_svm = LinearSVC()
# define the combination of parameters to be considered for parameter tuning
param_grid = {'C':[0.1,1,10,100],
              'penalty':['l1','l2']}
# Run the Grid Search
gs = GridSearchCV(clf_svm, param_grid, cv = 5, verbose = 1, n_jobs = -1)
best_svm = gs.fit(x,y)

# print the result
scores_df = pd.DataFrame(best_svm.cv_results_)
scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
print(scores_df[['rank_test_score', 'mean_test_score', 'param_penalty', 'param_C']])

# print the parameters of the best model
print(best_svm.best_params_)

# Use the best hyper-parameters to train on the whole dataset.
svm = best_svm.best_estimator_.fit(x,y)

# save the model to local disk
joblib.dump(svm, 'svm_presence_classifier.joblib')



### --------------------------------------------------------------------------------------------------
### ------------------------------------- Logistic Regression Classifier Training/ Parameter Tuning
### --------------------------------------------------------------------------------------------------

# setup the Logistic Regression classifier
clf_lr = LogisticRegression()
# define the combination of parameters to be considered for parameter tuning
param_grid = {'penalty':['l1','l2'],
              'solver':['lbfgs','newton-cg','sag']}
# Run the Grid Search
gs = GridSearchCV(clf_lr,param_grid, cv=5, verbose = 1, n_jobs = -1)
best_lr = gs.fit(x,y)

# print the result
scores_df = pd.DataFrame(best_lr.cv_results_)
scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
print(scores_df [['rank_test_score', 'mean_test_score', 'param_penalty', 'param_solver']])

# print the parameters of the best model
print(best_lr.best_params_)

# Use the best hyper-parameters to train on the whole dataset.
lr = best_lr.best_estimator_.fit(x,y)

# save the model to local disk
joblib.dump(lr, 'lr_presence_classifier.joblib')



### --------------------------------------------------------------------------------------------------
### ------------------------------------- Random Forest Classifier Training/ Parameter Tuning
### --------------------------------------------------------------------------------------------------

# setup the Random Forest classifier
clf_rf = RandomForestClassifier()
# define the combination of parameters to be considered for parameter tuning
param_grid = {'bootstrap':[True,False],
              'criterion':['gini','entropy'],
              'max_depth':[10,20,30,40,50,60,70,80,90,100, None],
              'min_samples_leaf':[1,2,4],
              'min_samples_split':[2,5,10],
              'n_estimators':[100,200,300,400,500,600]}
# Run the Grid Search
gs = GridSearchCV(clf_rf, param_grid, cv=5, verbose = 1, n_jobs = -1)
best_rf = gs.fit(x,y)

# print the result
scores_df = pd.DataFrame(best_rf.cv_results_)
scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
print(scores_df [['rank_test_score', 'mean_test_score', 'param_bootstrap', 'param_criterion',
            'param_max_depth','param_min_samples_leaf','param_min_samples_split','param_n_estimators']])

# print the parameters of the best model
print(best_rf.best_params_)

# Use the best hyper-parameters to train on the whole dataset.
rf = best_rf.best_estimator_.fit(x,y)

# save the model to local disk
joblib.dump(rf, 'rf_presence_classifier.joblib')



