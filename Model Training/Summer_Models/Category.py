import pandas as pd
import numpy as np

# to encode text, aka tokenize documents, to learn the vocabulary and inverse document frequency weightings.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split

# provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.
from sklearn.feature_extraction.text import CountVectorizer

# systematically compute word counts using CountVectorizer and them compute the Inverse Document Frequency (IDF) values and only then compute the Tf-idf scores.
from sklearn.feature_extraction.text import TfidfTransformer

# MultinomialNB (multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts, however, in practice, fractional counts such as tf-idf may also work.
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Evaluation metrics
from sklearn import metrics

# joblib is a set of tools to provide lightweight pipelining in Python. It provides utilities for saving and loading Python objects that make use of NumPy data structures, efficiently.
import joblib

# ----------------- Data Preparation ------------

# ---- import dataset from the Princeton Article
df = pd.read_csv('dark_patterns.csv')

# ---- select from the dataset when 'Pattern String' is not NaN values.
df = df[pd.notnull(df["Pattern String"])]

# ---- select only "Pattern String" and "Pattern Category" 2 columns to be the sub-dataset.
col = ["Pattern String", "Pattern Category"]
df = df[col]

print(df['Pattern Category'].value_counts())

# ---- encode the pattern category type into integers (7 types in total, encoded into integers from 0-6).

df["category_id"] = df['Pattern Category'].factorize()[0]

# ---- Get the mapping of the encoding integers and the pattern categories.
# ---- {'Social Proof': 0, 'Misdirection': 1, 'Urgency': 2, 'Forced Action': 3, 'Obstruction': 4, 'Sneaking': 5, 'Scarcity': 6}

category_id_df = df[['Pattern Category', 'category_id']
                    ].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(
    category_id_df[['category_id', 'Pattern Category']].values)


# ---- result of the mapping

print(category_to_id)

# ---- convert a collection of raw documents to a matrix of TF-IDF features; Equivalent to CountVectorizer followed by TfidfTransformer.
# 'sublinear_tf=True' is used to normalise bias of term frequency ("where a term that is more frequent shouldn't be X times as important"). It is set to True to use a logarithmic form for frequency.
# 'norm='l2'' is the default setting of 'norm', used to reduce document length bias, to ensure all our feature vectors have a enclidian norm of 1.
# 'min_df=5', means when building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold (which is 5 here), which is the minimum numbers of documents a word must be present in to be kept.
# 'ngram_range=(1,2)' means unigrams and bigrams will be extracted, means we want to consider both unigrams and bigrams.
# 'stop_words='english'', if a string, it is passed to _check_stop_list and the appropriate stop list is returned. To remove all common pronouns ("a", "the" ...), reducing the number of noisy features.

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df['Pattern String']).toarray()
labels = df.category_id

print(features.shape)

# The result means each of the 1512 pattern strings is represented by 303 features, representing the tf-idf score for different unigrams and bigrams.

N = 3   # every n-gram will give 3 examples

for Category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("Pattern Category Type: '{}':".format(Category))
  print("  . Most correlated unigrams:\n     . {}".format('\n     . '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n     . {}".format('\n     . '.join(bigrams[-N:])))


# ----------- Split the dataset into Model Training and testing ------
String_train, String_test, Category_train, Category_test = train_test_split(
    df['Pattern String'], df['Pattern Category'], train_size=.6)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(Category_train)
y_train = encoder.transform(Category_train)
y_test = encoder.transform(Category_test)

# check the mapping of encoding results (from 0 to 6 representing 'Forced Action', 'Misdirection'......)

print(list(encoder.classes_))

# Check the frequency distribution of the Model Training pattern category with pattern category names.

(unique, counts) = np.unique(Category_train, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)

# Check the frequency distribution of the encoded Model Training pattern category with encoded integers.

(unique, counts) = np.unique(y_train, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)

# Check the frequency distribution of the encoded testing pattern category with encoded integers.

(unique, counts) = np.unique(y_test, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)

# ----------- Text Encoding -----------

# First get the word count vector of the pattern string to encode the pattern string.

cv = CountVectorizer()
String_train_counts = cv.fit_transform(String_train)

# Then use the tf-idf score to transform the encoded word count pattern string vectors.

tfidf_tf = TfidfTransformer()
X_train = tfidf_tf.fit_transform(String_train_counts)

# save the CountVectorizer to disk

joblib.dump(cv, 'category_CountVectorizer.joblib')


# ------------ Model Model Training -------------

# Five models are tested:
# -- Logistic Regression
# -- Linear Support Vector Machine
# -- Random Forest
# -- Decision Tree
# -- Multinomial Naive Bayes

classifiers = [LogisticRegression(),LinearSVC(), RandomForestClassifier(), DecisionTreeClassifier(), MultinomialNB()]

# Calculate the accuracies of different classifiers using default settings.

acc = []
# cm = []

for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(cv.transform(String_test))
    acc.append(metrics.accuracy_score(y_test, y_pred))
    cm.append(metrics.confusion_matrix(y_test, y_pred))

# List the accuracies of different classifiers.

for i in range(len(classifiers)):
    print(f"{classifiers[i]} accuracy: {acc[i]}")
    # print(f"Confusion Matris: {cm[i]}")


# --------------------- Multinomial Naive Bayes Classifier -----------

clf_mnb = MultinomialNB().fit(X_train, y_train)
print(clf_mnb.get_params())

y_pred = clf_mnb.predict(cv.transform(String_test))

print("Accuracy:", metrics.accuracy_score(y_pred, y_test))

(unique, counts) = np.unique(y_pred, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

# Parameter tunning
param_grid = {'alpha':[0,1],
              'fit_prior':[True, False]}

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(clf_mnb,param_grid,cv=5,
                      verbose = 1, n_jobs = -1)

best_mnb = gs.fit(X_train,y_train)

scores_df = pd.DataFrame(best_mnb.cv_results_)
scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
scores_df [['rank_test_score', 'mean_test_score', 'param_alpha', 'param_fit_prior']]


y_pred_best = best_mnb.predict(cv.transform(String_test))

print("Accuracy:", metrics.accuracy_score(y_pred_best, y_test))

(unique, counts) = np.unique(y_pred_best, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

 # save the model to local disk

joblib.dump(best_mnb, 'mnb_category_classifier.joblib')


# ---------------- Linear Support Vector Machine ------------

clf_svm = LinearSVC().fit(X_train, y_train)

print(clf_svm.get_params())

y_pred = clf_svm.predict(cv.transform(String_test))

print("Accuracy:", metrics.accuracy_score(y_pred, y_test))

(unique, counts) = np.unique(y_pred, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

# Parameter tunning
param_grid = {'penalty': ['l1', 'l2'],
              'C': [0.1, 1, 5, 10]}

gs = GridSearchCV(clf_svm, param_grid, cv=5,
                  verbose=1, n_jobs=-1)

best_svm = gs.fit(X_train, y_train)

scores_df = pd.DataFrame(best_svm.cv_results_)
scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
print(scores_df [['rank_test_score', 'mean_test_score', 'param_penalty', 'param_C']])

print(best_svm.best_params_ )

y_pred_best = best_svm.predict(cv.transform(String_test))

print("Accuracy:", metrics.accuracy_score(y_pred_best, y_test))

(unique, counts) = np.unique(y_pred_best, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

# save the model to local disk

joblib.dump(best_svm, 'svm_category_classifier.joblib')
