# ------------------------------------- ########################################## ------------------------------------
# ------------------------------------- Training Using Machine Learning Algorithms ------------------------------------
# ------------------------------------- ########################################## ------------------------------------

### --------------------------
### ------Import Packages
### --------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
from sklearn.neighbors import KNeighborsClassifier
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

# split the dataset into train and test dataset as a ratio of 70%/30% (train/test).
string_train, string_test, dark_train, dark_test = train_test_split(
    data['Pattern String'], data["classification"], train_size = .7, stratify = data["classification"])

# encode the target values into integers ---- "classification"
encoder = LabelEncoder()
encoder.fit(dark_train)
y_train = encoder.transform(dark_train)
y_test = encoder.transform(dark_test)

# Check the frequency distribution of the training pattern classification with pattern classification names.
(unique, counts) = np.unique(dark_train, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

# get the word count vector of the pattern string to encode the pattern string.
tv = TfidfVectorizer()
tv.fit(string_train)
x_train = tv.transform(string_train)
x_test = tv.transform(string_test)

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
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc.append(metrics.accuracy_score(y_test, y_pred))
    pre.append(metrics.precision_score(y_test, y_pred, pos_label=0))
    cm.append(metrics.confusion_matrix(y_test, y_pred))

# List the accuracies of different classifiers.
for i in range(len(classifiers)):
    print("{} accuracy: {:.3f}".format(classifiers[i],acc[i]))
    print("{} precision: {:.3f}".format(classifiers[i],pre[i]))
    print("Confusion Matrix: {}".format(cm[i]))



### --------------------------------------------------------------------------------------------------
### ------------------------------------- Bernoulli Naive Bayes Classifier Training/ Parameter Tuning
### --------------------------------------------------------------------------------------------------

# setup the Bernoulli Naive Bayes classifier
clf_bnb = BernoulliNB().fit(x_train, y_train)
# define the combination of parameters to be considered for parameter tuning
param_grid = {'alpha':[0,1], 'fit_prior':[True, False]}
# Run the Grid Search
gs = GridSearchCV(clf_bnb,param_grid,cv=5, verbose = 1, n_jobs = -1)
best_bnb = gs.fit(x_train,y_train)

# print the result
scores_df = pd.DataFrame(best_bnb.cv_results_)
scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
print(scores_df[['rank_test_score', 'mean_test_score', 'param_alpha', 'param_fit_prior']])

# use the best model to predict the test dataset
y_pred_best = best_bnb.predict(x_test)

# metrics of evaluation
bnb_accuracy = metrics.accuracy_score(y_test, y_pred_best)
bnb_precision = metrics.precision_score(y_test,y_pred_best, pos_label=0)
bnb_cm = metrics.confusion_matrix(y_test, y_pred_best)
bnb_recall = cm[0][0]/(cm[0][0]+cm[0][1])
bnb_f1 = 2*bnb_precision*bnb_recall/(bnb_recall+bnb_precision)
print("Accuracy (BNB):", bnb_accuracy)
print("Precision (BNB):", bnb_precision)
print("Recall (BNB):", bnb_recall)
print("F1 Score (BNB):", bnb_f1)
print("Confusion Matrix (BNB):\n", bnb_cm)

# save the model to local disk
joblib.dump(best_bnb, 'bnb_presence_classifier.joblib')



### --------------------------------------------------------------------------------------------------
### ------------------------------------- Support Vector Machine Classifier Training/ Parameter Tuning
### --------------------------------------------------------------------------------------------------

# setup the Support Vector Machine classifier
clf_svm = LinearSVC().fit(x_train,y_train)
# define the combination of parameters to be considered for parameter tuning
param_grid = {'C':[0.1,1,10,100],
              'penalty':['l1','l2']}
# Run the Grid Search
gs = GridSearchCV(clf_svm,param_grid, cv = 5, verbose = 1, n_jobs = -1)
best_svm = gs.fit(x_train,y_train)

# print the result
scores_df = pd.DataFrame(best_svm.cv_results_)
scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
print(scores_df[['rank_test_score', 'mean_test_score', 'param_penalty', 'param_C']])

# use the best model to predict the test dataset
y_pred_best = best_svm.predict(x_test)

# metrics of evaluation
svm_accuracy = metrics.accuracy_score(y_test, y_pred_best)
svm_precision = metrics.precision_score(y_test, y_pred_best, pos_label=0)
svm_cm = metrics.confusion_matrix(y_test, y_pred_best)
svm_recall = cm[0][0]/(cm[0][0]+cm[0][1])
svm_f1 = 2*svm_precision*svm_recall/(svm_recall+svm_precision)
print("Accuracy (RF):", svm_accuracy)
print("Precision (RF):", svm_precision)
print("Recall (RF):", svm_recall)
print("F1 Score (RF):", svm_f1)
print("Confusion Matrix (RF):\n", svm_cm)

# save the model to local disk
joblib.dump(best_svm, 'svm_presence_classifier.joblib')



### --------------------------------------------------------------------------------------------------
### ------------------------------------- Logistic Regression Classifier Training/ Parameter Tuning
### --------------------------------------------------------------------------------------------------

# setup the Logistic Regression classifier
clf_lr = LogisticRegression().fit(x_train, y_train)
# define the combination of parameters to be considered for parameter tuning
param_grid = {'penalty':['l1','l2'],
              'solver':['lbfgs','newton-cg','sag']}
# Run the Grid Search
gs = GridSearchCV(clf_lr,param_grid, cv=5, verbose = 1, n_jobs = -1)
best_lr = gs.fit(x_train,y_train)

# print the result
scores_df = pd.DataFrame(best_lr.cv_results_)
scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
print(scores_df [['rank_test_score', 'mean_test_score', 'param_penalty', 'param_solver']])

# use the best model to predict the test dataset
y_pred_best = best_lr.predict(x_test)

# metrics of evaluation
lr_accuracy = metrics.accuracy_score(y_test, y_pred_best)
lr_precision = metrics.precision_score(y_test, y_pred_best, pos_label=0)
lr_cm = metrics.confusion_matrix(y_test, y_pred_best)
lr_recall = cm[0][0]/(cm[0][0]+cm[0][1])
lr_f1 = 2*lr_precision*lr_recall/(lr_recall+lr_precision)
print("Accuracy (RF):", lr_accuracy)
print("Precision (RF):", lr_precision)
print("Recall (RF):", lr_recall)
print("F1 Score (RF):", lr_f1)
print("Confusion Matrix (RF):\n", lr_cm)

# save the model to local disk
joblib.dump(best_lr, 'lr_presence_classifier.joblib')



### --------------------------------------------------------------------------------------------------
### ------------------------------------- Random Forest Classifier Training/ Parameter Tuning
### --------------------------------------------------------------------------------------------------

# setup the Random Forest classifier
clf_rf = RandomForestClassifier().fit(x_train, y_train)
# define the combination of parameters to be considered for parameter tuning
param_grid = {'bootstrap':[True,False],
              'criterion':['gini','entropy'],
              'max_depth':[10,20,30,40,50,60,70,80,90,100, None],
              'min_samples_leaf':[1,2,4],
              'min_samples_split':[2,5,10],
              'n_estimators':[100,200,300,400,500,600]}
# Run the Grid Search
gs = GridSearchCV(clf_rf,param_grid, cv=5, verbose = 1, n_jobs = -1)
best_rf = gs.fit(x_train,y_train)

# print the result
scores_df = pd.DataFrame(best_rf.cv_results_)
scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
print(scores_df [['rank_test_score', 'mean_test_score', 'param_bootstrap', 'param_criterion',
            'param_max_depth','param_min_samples_leaf','param_min_samples_split','param_n_estimators']])

# use the best model to predict the test dataset
y_pred_best = best_rf.predict(x_test)

# metrics of evaluation
rf_accuracy = metrics.accuracy_score(y_test, y_pred_best)
rf_precision = metrics.precision_score(y_test, y_pred_best, pos_label=0)
rf_cm = metrics.confusion_matrix(y_test, y_pred_best)
rf_recall = cm[0][0]/(cm[0][0]+cm[0][1])
rf_f1 = 2*rf_precision*rf_recall/(rf_recall+rf_precision)
print("Accuracy (RF):", rf_accuracy)
print("Precision (RF):", rf_precision)
print("Recall (RF):", rf_recall)
print("F1 Score (RF):", rf_f1)
print("Confusion Matrix (RF):\n", rf_cm)

# save the model to local disk
joblib.dump(best_rf, 'rf_presence_classifier.joblib')


### --------------------------------------------------------------------------------------------------
### ------------------------------------- Model Selection for Deploy
### --------------------------------------------------------------------------------------------------

# create the dictionary gathering the information of the models trained
# format as [Accuracy, Precision, Recall, F1 Score]
model_dic = {"BNB": {"Accuracy": bnb_accuracy, "Precision": bnb_precision, "Recall": bnb_recall, "F1": bnb_f1},
             "LR":{"Accuracy": lr_accuracy, "Precision": lr_precision, "Recall": lr_recall, "F1": lr_f1},
             "SVM":{"Accuracy": svm_accuracy, "Precision": svm_precision, "Recall": svm_recall, "F1": svm_f1},
             "RF":{"Accuracy": rf_accuracy, "Precision": rf_precision, "Recall": rf_recall, "F1": rf_f1}}

# find the model with the highest Precision
max_precision = max(values["Precision"] for key, values in model_dic.items())
print(max_precision)
model_best_precision = [model for model, precision in model_dic.items() if precision["Precision"] == max_precision]
print(model_best_precision)

# find the model with the highest Recall
max_recall = max(values["Recall"] for key, values in model_dic.items())
print(max_recall)
model_best_recall = [model for model, recall in model_dic.items() if recall["Recall"] == max_recall]
print(model_best_recall)

# find the model with the highest F1 Score
max_f1 = max(values["F1"] for key, values in model_dic.items())
print(max_f1)
model_best_f1 = [model for model, f1 in model_dic.items() if f1["Recall"] == max_f1]
print(model_best_f1)

# save the model having the best F1 Score
model_map = {"BNB":best_bnb, "LR": best_lr, "SVM": best_svm, "RF": best_rf}
# if there are more than 1 best model (have same highest F1 Score), then we use "precision" to decide:
if len(model_best_f1) == 1:
    joblib.dump(model_map[model_best_f1[0]], 'best_f1_presence_classifier.joblib')
else:
    # subset the model_dic to be the ones have the highest F1 Score
    precision_dic = {key: value for key, value in model_dic.items() if key in model_best_f1}
    # find the model with the highest Precision
    sub_max_precision = max(values["Precision"] for key, values in precision_dic.items())
    print(sub_max_precision)
    model_sub_best_precision = [model for model, precision in precision_dic.items()
                                if precision["Precision"] == sub_max_precision]
    print(model_sub_best_precision)
    if len(model_sub_best_precision) == 1:
        joblib.dump(model_map[model_sub_best_precision[0]], 'best_precision_presence_classifier.joblib')
# if there are more than 1 best model with same F1 Score and same Precision, then we use "Recall" to decide:
    else:
        recall_dic = {key: value for key, value in model_dic.items() if key in model_sub_best_precision}
        # find the model with the highest Recall
        sub_max_recall = max(values["Recall"] for key, values in recall_dic.items())
        print(sub_max_recall)
        model_sub_best_recall = [model for model, recall in model_dic.items() if recall["Recall"] == max_recall]
        print(model_sub_best_recall)
        if len(model_sub_best_recall) == 1:
            joblib.dump(model_map[model_sub_best_recall[0]], 'best_recall_presence_classifier.joblib')
        else:
# if the best models have the same F1 Score, sampe precision, and same recall, then save them all.
            for model_index, model in enumerate(model_best_f1):
                filename = 'best_presence_classifier_' + str(model_index + 1)
                joblib.dump(model_map[model], filename)



