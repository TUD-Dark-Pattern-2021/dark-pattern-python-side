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
# The difference between MultinomialNB and BernoulliNB is that while  MultinomialNB works with occurrence counts,
# BernoulliNB is designed for binary/boolen features, which means in the case of text classification, word occurrence vectores
# (rather than word count vectors) may be more suitable to be used to train and use this classifier.
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
# Evaluation metrics
from sklearn import metrics
# joblib is a set of tools to provide lightweight pipelining in Python.
# It provides utilities for saving and loading Python objects that make use of NumPy data structures, efficiently.
import joblib

### --------------------------
### ------Dataset Import
### --------------------------

data = pd.read_csv('enriched_data.csv')
# Check the target distribution.
print('\nDistribution of the tags:\n{}'.format(data['Pattern Type'].value_counts()))
# For later training the model, we should remove the duplicate input to reduce overfitting.
data = data.drop_duplicates(subset="Pattern String")

### --------------------------
### ------Data Preparation
### --------------------------

# split the dataset into train and test dataset as a ratio of 80%/20% (train/test).
String_train, String_test, Type_train, Type_test = train_test_split(
    data['Pattern String'], data['Pattern Type'], train_size=.8)

# encode the target values into integers ---- "classification"
encoder = LabelEncoder()
encoder.fit(Type_train)
y_train = encoder.transform(Type_train)
y_test = encoder.transform(Type_test)
# check the mapping of encoding results (from 0 to 4 representing .......)
print(list(encoder.classes_))

# Check the frequency distribution of the training pattern classification with pattern classification names.
(unique, counts) = np.unique(Type_train, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

# get the word count vector of the pattern string to encode the pattern string.
tv = TfidfVectorizer()
tv.fit(String_train)
x_train = tv.transform(String_train)
x_test = tv.transform(String_test)

# save the TfidfVectorizer to disk
joblib.dump(tv, 'type_TfidfVectorizer.joblib')



### --------------------------------------------------------------------------------------------------
### ------Rough Idea about the performance of different classifiers
### --------------------------------------------------------------------------------------------------
# Four models are tested:
# -- Logistic Regression
# -- Linear Support Vector Machine
# -- Random Forest
# -- Multinomial Naive Bayes

classifiers = [LogisticRegression(), LinearSVC(), RandomForestClassifier(), MultinomialNB()]
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
### ------------------------------------- Multinomial Naive Bayes Classifier Training/ Parameter Tuning
### --------------------------------------------------------------------------------------------------

# setup the Multinomial Naive Bayes classifier
clf_mnb = MultinomialNB().fit(x_train, y_train)
# define the combination of parameters to be considered for parameter tuning
param_grid = {'alpha':[0,1], 'fit_prior':[True, False]}
# Run the Grid Search
gs = GridSearchCV(clf_mnb, param_grid, cv=5, verbose = 1, n_jobs = -1)
best_mnb = gs.fit(x_train, y_train)

# print the result
scores_df = pd.DataFrame(best_mnb.cv_results_)
scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
print(scores_df[['rank_test_score', 'mean_test_score', 'param_alpha', 'param_fit_prior']])

# use the best model to predict the test dataset
y_pred_best = best_mnb.predict(x_test)

# metrics of evaluation
mnb_accuracy = metrics.accuracy_score(y_test, y_pred_best)
mnb_precision = metrics.precision_score(y_test,y_pred_best, pos_label=0)
mnb_cm = metrics.confusion_matrix(y_test, y_pred_best)
mnb_recall = cm[0][0]/(cm[0][0]+cm[0][1])
mnb_f1 = 2*mnb_precision*mnb_recall/(mnb_recall+mnb_precision)
print("Accuracy (MNB):", mnb_accuracy)
print("Precision (MNB):", mnb_precision)
print("Recall (MNB):", mnb_recall)
print("F1 Score (MNB):", mnb_f1)
print("Confusion Matrix (MNB):\n", mnb_cm)

# print the distribution of the prediction result on the test dataset
(unique, counts) = np.unique(y_pred_best, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print('The distribution of predicted result of the best model:{}'.format(frequencies))

# save the model to local disk
joblib.dump(best_mnb, 'mnb_type_classifier.joblib')



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

# print the distribution of the prediction result on the test dataset
(unique, counts) = np.unique(y_pred_best, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print('The distribution of predicted result of the best model:{}'.format(frequencies))

# save the model to local disk
joblib.dump(best_svm, 'svm_type_classifier.joblib')



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

# print the distribution of the prediction result on the test dataset
(unique, counts) = np.unique(y_pred_best, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print('The distribution of predicted result of the best model:{}'.format(frequencies))

# save the model to local disk
joblib.dump(best_lr, 'lr_type_classifier.joblib')



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

# print the distribution of the prediction result on the test dataset
(unique, counts) = np.unique(y_pred_best, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print('The distribution of predicted result of the best model:{}'.format(frequencies))

# save the model to local disk
joblib.dump(best_rf, 'rf_type_classifier.joblib')


### --------------------------------------------------------------------------------------------------
### ------------------------------------- Model Selection for Deploy
### --------------------------------------------------------------------------------------------------

# create the dictionary gathering the information of the models trained
# format as [Accuracy, Precision, Recall, F1 Score]
model_dic = {"MNB": {"Accuracy": mnb_accuracy, "Precision": mnb_precision, "Recall": mnb_recall, "F1": mnb_f1},
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
model_map = {"MNB":best_mnb, "LR": best_lr, "SVM": best_svm, "RF": best_rf}
# if there are more than 1 best model (have same highest F1 Score), then we use "precision" to decide:
if len(model_best_f1) == 1:
    joblib.dump(model_map[model_best_f1[0]], 'best_f1_type_classifier.joblib')
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
        joblib.dump(model_map[model_sub_best_precision[0]], 'best_precision_type_classifier.joblib')
# if there are more than 1 best model with same F1 Score and same Precision, then we use "Recall" to decide:
    else:
        recall_dic = {key: value for key, value in model_dic.items() if key in model_sub_best_precision}
        # find the model with the highest Recall
        sub_max_recall = max(values["Recall"] for key, values in recall_dic.items())
        print(sub_max_recall)
        model_sub_best_recall = [model for model, recall in model_dic.items() if recall["Recall"] == max_recall]
        print(model_sub_best_recall)
        if len(model_sub_best_recall) == 1:
            joblib.dump(model_map[model_sub_best_recall[0]], 'best_recall_type_classifier.joblib')
        else:
# if the best models have the same F1 Score, sampe precision, and same recall, then save them all.
            for model_index, model in enumerate(model_best_f1):
                filename = 'best_type_classifier_' + str(model_index + 1)
                joblib.dump(model_map[model], filename)



