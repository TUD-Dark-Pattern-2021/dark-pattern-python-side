import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

# provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.
from sklearn.feature_extraction.text import CountVectorizer
# systematically compute word counts using CountVectorizer and them compute the Inverse Document Frequency (IDF) values and only then compute the Tf-idf scores.
from sklearn.feature_extraction.text import TfidfTransformer

# Bernoulli Naive Bayes (Similar as  MultinomialNB), this classifier is suitable for discrete data. The difference between MultinomialNB and BernoulliNB is that while  MultinomialNB works with occurrence counts, BernoulliNB is designed for binary/boolen features, which means in the case of text classification, word occurrence vectores (rather than word count vectors) may be more suitable to be used to train and use this classifier.
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Evaluation metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# joblib is a set of tools to provide lightweight pipelining in Python. It provides utilities for saving and loading Python objects that make use of NumPy data structures, efficiently.
import joblib

# import matplotlib.pyplot as plt
# import seaborn as sns

# -----------------------------------------------------
normie = pd.read_csv('normie.csv')
princeton = pd.read_csv('dark_patterns.csv')


# check the distribution of the target value --- classification.

print(normie['classification'].value_counts())

 # remove the instances with NULL value of 'Pattern String' and 'classification', which will be the input of our model.

normie = normie[pd.notnull(normie["Pattern String"])]
normie = normie[pd.notnull(normie["classification"])]

# check the final distribution of the classification after removing the rows with NULL values.

print(normie['classification'].value_counts())

normie = normie[normie["classification"] == 0]


normie["classification"] = "Not Dark"

 # For later training the model, we should remove the duplicate input to reduce overfitting.

normie = normie.drop_duplicates(subset="Pattern String")

 # remove the rows where there are NULL values in 'Pattern String' or 'Pattern Category' columns.

princeton = princeton[pd.notnull(princeton["Pattern String"])]
princeton = princeton[pd.notnull(princeton["Pattern Category"])]

# create a column named 'classification' and give all the values to be 'Dark', to match with the normie dataset.

princeton["classification"] = "Dark"

 # For later training the model, we should remove the duplicate input to reduce overfitting.

princeton = princeton.drop_duplicates(subset="Pattern String")

 # Subset the princeton dataset for joining with normie dataset.

cols = ["Pattern String", "classification"]
princeton = princeton[cols]

df = pd.concat([normie, princeton])

print(df['classification'].value_counts())

# --------------------- Data Preparation ------------------

# split the dataset into train and test dataset as a ratio of 60%/40% (train/test).

string_train, string_test, dark_train, dark_test = train_test_split(
    df['Pattern String'], df["classification"], train_size = .6)

encoder = LabelEncoder()
encoder.fit(dark_train)
y_train = encoder.transform(dark_train)
y_test = encoder.transform(dark_test)

# check the mapping of encoding results (from 0 to 1 representing 'Dark', 'Not Dark')

print(list(encoder.classes_))

# Check the frequency distribution of the training pattern classification with pattern classification names.

(unique, counts) = np.unique(dark_train, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)


# Check the frequency distribution of the encoded training pattern classification with encoded integers.

(unique, counts) = np.unique(y_train, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)


# Check the frequency distribution of the encoded testing pattern classification with encoded integers.

(unique, counts) = np.unique(y_test, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)


 # First get the word count vector of the pattern string to encode the pattern string.

cv = CountVectorizer()
string_train_counts = cv.fit_transform(string_train)

# Then use the tf-idf score to transform the encoded word count pattern string vectors.

tfidf_tf = TfidfTransformer()
X_train = tfidf_tf.fit_transform(string_train_counts)

 # save the CountVectorizer to disk

joblib.dump(cv, 'presence_CountVectorizer.joblib')

# Five models are tested:
# -- Logistic Regression
# -- Linear Support Vector Machine
# -- Random Forest
# -- Multinomial Naive Bayes
# -- Bernoulli Naive Bayes

classifiers = [LogisticRegression(),LinearSVC(), RandomForestClassifier(), MultinomialNB(), BernoulliNB()]


# Calculate the accuracies of different classifiers using default settings.

acc = []
cm = []

for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(cv.transform(string_test))
    acc.append(metrics.accuracy_score(y_test, y_pred))
    cm.append(metrics.confusion_matrix(y_test, y_pred))

 # List the accuracies of different classifiers.

for i in range(len(classifiers)):
    print(f"{classifiers[i]} accuracy: {acc[i]}")
    # print(f"Confusion Matris: {cm[i]}")

# ---------------- Bernoulli Naive Bayes Classifier ------------------

clf_bnb = BernoulliNB().fit(X_train, y_train)

y_pred = clf_bnb.predict(cv.transform(string_test))

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))

(unique, counts) = np.unique(y_pred, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)


# Parameter tunning

param_grid = {'alpha': [0, 1],
              'fit_prior': [True, False]}

gs = GridSearchCV(clf_bnb,param_grid,cv=5,
                      verbose = 1, n_jobs = -1)

best_bnb = gs.fit(X_train, y_train)

scores_df = pd.DataFrame(best_bnb.cv_results_)
scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
print(scores_df [['rank_test_score', 'mean_test_score', 'param_alpha', 'param_fit_prior']])

print(best_bnb.best_params_)

y_pred_best = best_bnb.predict(cv.transform(string_test))

(unique, counts) = np.unique(y_pred_best, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

 # save the model to local disk

joblib.dump(best_bnb, 'bnb_presence_classifier.joblib')




# ------------------- Random Forest Classifier -------------
clf_rf = RandomForestClassifier().fit(X_train, y_train)

y_pred = clf_rf.predict(cv.transform(string_test))

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))

(unique, counts) = np.unique(y_pred, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

# Parameter tunning
param_grid = {'bootstrap':[True,False],
              'criterion':['gini','entropy'],
              'max_depth':[10,20,30,40,50,60,70,80,90,100, None],
              'min_samples_leaf':[1,2,4],
              'min_samples_split':[2,5,10],
              'n_estimators':[100,200,300,400,500,600]}

gs = GridSearchCV(clf_rf, param_grid, cv=5,
                  verbose=1, n_jobs=-1)

best_rf = gs.fit(X_train,y_train)

scores_df = pd.DataFrame(best_rf.cv_results_)
scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
print(scores_df [['rank_test_score', 'mean_test_score', 'param_bootstrap', 'param_criterion','param_max_depth','param_min_samples_leaf','param_min_samples_split','param_n_estimators']])

print(best_rf.best_params_)

y_pred_best = best_rf.predict(cv.transform(string_test))

print("Accuracy:", metrics.accuracy_score(y_test, y_pred_best))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred_best))

(unique, counts) = np.unique(y_pred_best, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

# save the model to local disk

joblib.dump(best_rf, 'rf_presence_classifier.joblib')




