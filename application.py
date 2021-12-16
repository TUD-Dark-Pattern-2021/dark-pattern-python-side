import pandas as pd
import numpy as np
import platform

import io

from smart_open import smart_open
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

import urllib.request
# joblib is a set of tools to provide lightweight pipelining in Python. It provides utilities for saving and loading Python objects that make use of NumPy data structures, efficiently.

import joblib
import datetime

from flask import Flask, request, Response
import json
import requests
import shortuuid
from PIL import Image
import pytesseract
import os
import cv2

application = Flask(__name__)

def say_hello(username = "Roger's World"):
   return '<p>Hello Lan %s!</p>\n' % username

application.add_url_rule('/', 'index', (lambda: say_hello()))

@application.route('/api/checkDP',methods = ['POST'])
def checkDP():
    data = request.get_data()
    j_data = json.loads(data)

    # ------ Check the 5 pattern types -------
    presence_model = joblib.load('rf_presence_classifier.joblib')
    presence_cv = joblib.load('presence_TfidfVectorizer.joblib')
    prediction = presence_model.predict(presence_cv.transform([j_data['content']]))
    # -------- Check the total detection result --------
    if prediction == [0]:
        return_result = {
            "isDarkPattern": 'Yes'
        }
    else:
        return_result = {
            "isDarkPattern": 'No'
        }
    return Response(json.dumps(return_result), mimetype='application/json')

@application.route('/api/checkOCR',methods = ['POST'])
def checkOCR():
    #if platform.system().lower() == 'windows':
        #pytesseract.pytesseract.tesseract_cmd = r'C:\Users\seanq\AppData\Local\Tesseract-OCR\tesseract.exe'
    data = request.get_data()
    j_data = json.loads(data)
    print(j_data)
    r = requests.request('get', j_data['content'])
    image_name = shortuuid.uuid() + '.jpg'
    image_path = "./images/" + image_name
    with open(image_path, 'wb') as f:
        f.write(r.content)
    f.close()
    str = pytesseract.image_to_string(Image.open(image_path))
    os.remove(image_path)
    return_result = {
        "status":200,
        "content": str
    }
    return Response(json.dumps(return_result), mimetype='application/json')



@application.route('/api/parse',methods = ['POST'])

def parse():

    def ocr():
        # for running local
        #if platform.system().lower() == 'windows':
            #pytesseract.pytesseract.tesseract_cmd = r'C:\Users\seanq\AppData\Local\Tesseract-OCR\tesseract.exe'
        #pytesseract.pytesseract.tesseract_cmd = r'C:\Users\seanq\AppData\Local\Tesseract-OCR\tesseract.exe'
        data = request.get_data()
        j_data = json.loads(data)
        # get urls with type = image
        full = pd.DataFrame(j_data)
        print(full)
        urlss = full.loc[full['type'] == 'image']

        #print(urlss)

        urlss.duplicated(['content'])
        urlss3 = urlss.drop_duplicates(['content'])
        urlss2 = urlss3.reset_index(drop=True)
        print(urlss2)
        print('working')
        urls = urlss2['content']
        #print(urls)

#        def get_as_base64(url):
#            return base64.b64encode(requests.get(url).content)

#        def get_grayscale(img):
#            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#        def thresholding(img):
#            return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        def image_text(prep):
            return pytesseract.image_to_string(prep)

        # create a empty dataframe
        # df_image = pd.DataFrame(columns=["content", "tag", "key", "type"])
        def texture_detect(all_urls):
            a = -1
            for line in all_urls:
                a = a + 1
                #(debug) if url has http domain rather than https
                if 'http' in line:
                    #line = "https:" + line

                    print(line)

                #try:
                    img_resp = urllib.request.urlopen(line)
                    img = np.asarray(bytearray(img_resp.read()), dtype='uint8')
                    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                    '''
                    r = requests.get(line)
                    image_name = 'image.jpg'
                    image_path = "./" + image_name
                    with open(image_path, "wb") as f:
                        f.write(r.content)
                    f.close

                    itext = image_text(image_path)

                    os.remove(image_path)
                    '''
                    itext = image_text(img)

                    urlss2['content'][a] = itext
                    print(itext)
                    print('itextwork')
                #except:
                    #continue

            urlss2["content"] = urlss2["content"].map(lambda x: x.split('\n'))
            urlsss = urlss2.explode("content")
            return urlsss
        texture_detect = texture_detect(urls)
        return texture_detect

    # ---------------------------  Check Confirmshaming DP --------------------------

    def confirm_shaming():
        # Get all the HTML data
        data = request.get_data()
        j_data = json.loads(data)
        # get text with type == link or type == button
        html = pd.DataFrame(j_data)
        print(html)
        link_text = html.loc[html['type'].isin(['link','button'])]

        if len(link_text) != 0:
            # Loading the saved model with joblib
            detection_model = joblib.load('confirm_rf_clf.joblib')
            detection_cv = joblib.load('confirm_tv.joblib')

            # apply the pre-trained confirmshaming detection model to the button / link text data
            pred_vec = detection_model.predict(detection_cv.transform(link_text['content'].str.lower()))
            link_text['presence'] = pred_vec.tolist()

            # dark pattern content are those where the predicted result equals to 0.
            confirm_shaming = link_text.loc[link_text['presence'] == 0]
            confirm_shaming = confirm_shaming.reset_index(drop=True)

            # get the number of presence of dark pattern
            confirm_shaming_count = confirm_shaming.shape[0]
            print('Number of confirmshaming DP: ', confirm_shaming_count)

        else:
            empty_df = pd.DataFrame()
            confirm_shaming = empty_df

        return confirm_shaming

    confirm_shaming = confirm_shaming()
    confirm_count = confirm_shaming.shape[0]

    # ----------------------------- Get Text data for 5 DP types detection ---------------------

    data = request.get_data()
    j_data = json.loads(data)
    # print("input data", j_data)

    # ----------------------------- Checking Presence ----------------------

     # Loading the saved model with joblib
    presence_model = joblib.load('rf_presence_classifier.joblib')
    presence_cv = joblib.load('presence_TfidfVectorizer.joblib')

    # New dataset to predict
    presence_pred = pd.DataFrame(j_data)

    # --------- Make OCR optional -----
    if presence_pred['is_ocr'][0] == 1:
        texture_detect = ocr()
        # filter type == text
        #textpp = presence_pred.loc[presence_pred['type'] == 'text']
        #combine = [textpp, texture_detect]
        combine = [presence_pred, texture_detect]
        presence_pred = pd.concat(combine)

    else:
        presence_pred = presence_pred

    # Remove the rows where the first letter starting with ignoring characters
    ignore_str = [',', '.', ';', '{', '}', '#', '/', '(', ')', '?']
    presence_pred = presence_pred[~presence_pred['content'].str[0].isin(ignore_str)]

    # Keep the rows where the word count is between 2 and 20 (inclusive)
    presence_pred = presence_pred[presence_pred['content'].str.split().str.len() > 1]
    presence_pred = presence_pred[presence_pred['content'].str.split().str.len() < 21]

    # Filter out the disturibing content to be removed.
#    str_list = ['low to high', 'high to low', 'high low', 'low high', '{', 'ships', 'ship', '®', 'details',
#                'limited edition', 'cart is currently empty', 'in cart', 'out of stock', 'believe in',
#                'today\'s deals', 'customer service', 'offer available', 'offers available', 'collect',
#                '% off', 'in stock soon', 'problem', 'UTC', 'javascript', 'cookie', 'cookies', 'disclaimer','https']

    str_list = ['{', 'UTC']
    pattern = '|'.join(str_list)
    presence_pred = presence_pred[~presence_pred.content.str.lower().str.contains(pattern)]

    # apply the pre-trained model to the new content data
    pre_pred_vec = presence_model.predict(presence_cv.transform(presence_pred['content']))
    presence_pred['presence'] = pre_pred_vec.tolist()

    # dark pattern content are those where the predicted result equals to 0.
    dark = presence_pred.loc[presence_pred['presence'] == 0]

    # get the number of presence of dark pattern
    pre_count = dark.shape[0]

    # ------------------------- Pattern Type Classification ------------------
    # when there is no dark patterns at all
    if pre_count == 0 and confirm_count == 0:
        return_result = {
            "total_counts": {},
            "items_counts": {},
            "details": []
        }

    # when there are dark patterns in any of the 5 Types DP
    elif pre_count != 0:
        # Loading the saved model with joblib
#        type_model = joblib.load('lr_type_classifier.joblib')
#        type_cv = joblib.load('type_CountVectorizer.joblib')

        # testing auto-training result
        type_model = joblib.load('lr_type_classifier.joblib')
        type_cv = joblib.load('type_CountVectorizer.joblib')

        # mapping of the encoded dark pattern types.
        type_dic = {0:'FakeActivity', 1:'FakeCountdown', 2:'FakeHighDemand', 3:'FakeLimitedTime', 4:'FakeLowStock'}

        type_slug = {0:'Fake Activity', 1:'Fake Countdown', 2:'Fake High-demand', 3:'Fake Limited-time', 4:'Fake Low-stock'}

        # apply the model and the countvectorizer to the detected dark pattern content data
        type_pred_vec = type_model.predict(type_cv.transform(dark['content']))                   # Problem

        dark['classification'] = type_pred_vec.tolist()
        type_list = dark['classification'].tolist()

        # get the mapping of the type name and encoded type integers
        dark['type_name'] = [type_dic[int(classification)] for classification in type_list]
        dark['type_name_slug'] = [type_slug[int(classification)] for classification in type_list]

        # reset the index of the detected dark pattern list on the webpage.
        dark = dark.reset_index(drop=True)
        return_result = {
            "total_counts": {},
            "items_counts": {},
            "details": []
        }
        # get the list of the dark patterns detected with the frequency count

        return_result['total_counts'] = pre_count + confirm_count
        counts = dark['type_name'].value_counts()
        for index, type_name in enumerate(counts.index.tolist()):
            return_result["items_counts"][type_name] = int(counts.values[index])
        for j in range(len(dark)):
            return_result["details"].append({
                "content": dark['content'][j],
                "tag":dark['tag'][j],
                "tag_type": dark['type'][j],
                "key": dark['key'][j],
                "type_name": dark['type_name'][j],
                "type_name_slug": dark['type_name_slug'][j]
            })

        # ----------- Add confirmshaming DP information if there is any. ---------
        if confirm_count != 0:
            return_result["items_counts"]["Confirmshaming"] = confirm_count
            for j in range(len(confirm_shaming)):
                return_result["details"].append({
                    "content": confirm_shaming['content'][j],
                    "tag": confirm_shaming['tag'][j],
                    "tag_type": confirm_shaming['type'][j],
                    "key": confirm_shaming['key'][j],
                    "type_name": "Confirmshaming",
                    "type_name_slug": "Confirmshaming"
                })
        print("return_result", return_result)

    # when there is no DP in the 5 types DP, only confirmshaming DP exists
    else:
        return_result = {
            "total_counts": {},
            "items_counts": {},
            "details": []
        }
        return_result["total_counts"] = confirm_count
        return_result["items_counts"]["Confirmshaming"] = confirm_count
        for j in range(len(confirm_shaming)):
            return_result["details"].append({
                "content": confirm_shaming['content'][j],
                "tag": confirm_shaming['tag'][j],
                "tag_type": confirm_shaming['type'][j],
                "key": confirm_shaming['key'][j],
                "type_name": "Confirmshaming",
                "type_name_slug": "Confirmshaming"
            })
        print("return_result", return_result)
        # ----------------
    return Response(json.dumps(return_result), mimetype='application/json')


# -------------------------------------------- Auto Training --------------------------------------------

@application.route('/api/autoTrain', methods=['POST'])
def autoTrain():
    ### --------------------------
    ### ------Request Receiving
    ### --------------------------

    data = request.get_data()
    j_data = json.loads(data)

    ### --------------------------
    ### ------Dataset Import
    ### --------------------------

    bucket_name = j_data['bucket']
    object_key = j_data['csv']
    path = 's3://{}/{}'.format(bucket_name, object_key)

    dataset = pd.read_csv(smart_open(path))
    # Check the target distribution.
    print('\nDistribution of the tags:\n{}'.format(dataset['Pattern_Type:'].value_counts()))
    # For later training the model, we should remove the duplicate input to reduce overfitting.
    dataset = dataset.drop_duplicates(subset="Pattern_String")

    ### --------------------------
    ### ------Data Preparation
    ### --------------------------

    # split the dataset into train and test dataset as a ratio of 80%/20% (train/test).
    String_train, String_test, Type_train, Type_test = train_test_split(
        dataset['Pattern_String'], dataset['Pattern_Type:'], train_size=.8, random_state=22)

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
    filename = 'Auto_Training/Models/type_TfidfVectorizer_' + str(datetime.datetime.now().date()) + '_' \
                   + str(datetime.datetime.now().time()).replace(':', '.') + '.joblib'
    joblib.dump(tv, filename)

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
    cm = []
    for clf in classifiers:
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc.append(metrics.accuracy_score(y_test, y_pred))
        cm.append(metrics.confusion_matrix(y_test, y_pred))

    # List the accuracies of different classifiers.
    for i in range(len(classifiers)):
        print("{} accuracy: {:.3f}".format(classifiers[i], acc[i]))
        print("Confusion Matrix: {}".format(cm[i]))

    ### --------------------------------------------------------------------------------------------------
    ### ------------------------------------- Multinomial Naive Bayes Classifier Training/ Parameter Tuning
    ### --------------------------------------------------------------------------------------------------

    # setup the Multinomial Naive Bayes classifier
    clf_mnb = MultinomialNB().fit(x_train, y_train)
    # define the combination of parameters to be considered for parameter tuning
    param_grid = {'alpha': [0, 1], 'fit_prior': [True, False]}
    # Run the Grid Search
    gs = GridSearchCV(clf_mnb, param_grid, cv=5, verbose=1, n_jobs=-1)
    best_mnb = gs.fit(x_train, y_train)

    # print the result
    scores_df = pd.DataFrame(best_mnb.cv_results_)
    scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
    print(scores_df[['rank_test_score', 'mean_test_score', 'param_alpha', 'param_fit_prior']])

    # use the best model to predict the test dataset
    y_pred_best = best_mnb.predict(x_test)

    # metrics of evaluation
    mnb_accuracy = metrics.accuracy_score(y_test, y_pred_best)
    mnb_cm = metrics.confusion_matrix(y_test, y_pred_best)
    print("Accuracy (MNB):", mnb_accuracy)
    print("Confusion Matrix (MNB):\n", mnb_cm)

    # print the distribution of the prediction result on the test dataset
    (unique, counts) = np.unique(y_pred_best, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print('The distribution of predicted result of the best model:{}'.format(frequencies))


    ### --------------------------------------------------------------------------------------------------
    ### ------------------------------------- Support Vector Machine Classifier Training/ Parameter Tuning
    ### --------------------------------------------------------------------------------------------------

    # setup the Support Vector Machine classifier
    clf_svm = LinearSVC().fit(x_train, y_train)
    # define the combination of parameters to be considered for parameter tuning
    param_grid = {'C': [0.1, 1, 10, 100],
                  'penalty': ['l1', 'l2']}
    # Run the Grid Search
    gs = GridSearchCV(clf_svm, param_grid, cv=5, verbose=1, n_jobs=-1)
    best_svm = gs.fit(x_train, y_train)

    # print the result
    scores_df = pd.DataFrame(best_svm.cv_results_)
    scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
    print(scores_df[['rank_test_score', 'mean_test_score', 'param_penalty', 'param_C']])

    # use the best model to predict the test dataset
    y_pred_best = best_svm.predict(x_test)

    # metrics of evaluation
    svm_accuracy = metrics.accuracy_score(y_test, y_pred_best)
    svm_cm = metrics.confusion_matrix(y_test, y_pred_best)
    print("Accuracy (RF):", svm_accuracy)
    print("Confusion Matrix (RF):\n", svm_cm)

    # print the distribution of the prediction result on the test dataset
    (unique, counts) = np.unique(y_pred_best, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print('The distribution of predicted result of the best model:{}'.format(frequencies))

    ### --------------------------------------------------------------------------------------------------
    ### ------------------------------------- Logistic Regression Classifier Training/ Parameter Tuning
    ### --------------------------------------------------------------------------------------------------

    # setup the Logistic Regression classifier
    clf_lr = LogisticRegression().fit(x_train, y_train)
    # define the combination of parameters to be considered for parameter tuning
    param_grid = {'penalty': ['l1', 'l2'],
                  'solver': ['lbfgs', 'newton-cg', 'sag']}
    # Run the Grid Search
    gs = GridSearchCV(clf_lr, param_grid, cv=5, verbose=1, n_jobs=-1)
    best_lr = gs.fit(x_train, y_train)

    # print the result
    scores_df = pd.DataFrame(best_lr.cv_results_)
    scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
    print(scores_df[['rank_test_score', 'mean_test_score', 'param_penalty', 'param_solver']])

    # use the best model to predict the test dataset
    y_pred_best = best_lr.predict(x_test)

    # metrics of evaluation
    lr_accuracy = metrics.accuracy_score(y_test, y_pred_best)
    lr_cm = metrics.confusion_matrix(y_test, y_pred_best)
    print("Accuracy (RF):", lr_accuracy)
    print("Confusion Matrix (RF):\n", lr_cm)

    # print the distribution of the prediction result on the test dataset
    (unique, counts) = np.unique(y_pred_best, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print('The distribution of predicted result of the best model:{}'.format(frequencies))

    ### --------------------------------------------------------------------------------------------------
    ### ------------------------------------- Random Forest Classifier Training/ Parameter Tuning
    ### --------------------------------------------------------------------------------------------------

    # setup the Random Forest classifier
    clf_rf = RandomForestClassifier().fit(x_train, y_train)
    # define the combination of parameters to be considered for parameter tuning
    param_grid = {'bootstrap': [True, False],
                  'criterion': ['gini', 'entropy'],
                  'max_depth': [10, None],
                  'min_samples_leaf': [1, 2],
                  'min_samples_split': [2, 5],
                  'n_estimators': [100, 200]}
    # Run the Grid Search
    gs = GridSearchCV(clf_rf, param_grid, cv=5, verbose=1, n_jobs=-1)
    best_rf = gs.fit(x_train, y_train)

    # print the result
    scores_df = pd.DataFrame(best_rf.cv_results_)
    scores_df = scores_df.sort_values(by=['rank_test_score']).reset_index(drop='index')
    print(scores_df[['rank_test_score', 'mean_test_score', 'param_bootstrap', 'param_criterion',
                     'param_max_depth', 'param_min_samples_leaf', 'param_min_samples_split', 'param_n_estimators']])

    # use the best model to predict the test dataset
    y_pred_best = best_rf.predict(x_test)

    # metrics of evaluation
    rf_accuracy = metrics.accuracy_score(y_test, y_pred_best)
    rf_cm = metrics.confusion_matrix(y_test, y_pred_best)
    print("Accuracy (RF):", rf_accuracy)
    print("Confusion Matrix (RF):\n", rf_cm)

    # print the distribution of the prediction result on the test dataset
    (unique, counts) = np.unique(y_pred_best, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print('The distribution of predicted result of the best model:{}'.format(frequencies))

    ### --------------------------------------------------------------------------------------------------
    ### ------------------------------------- Model Selection for Deploy
    ### --------------------------------------------------------------------------------------------------

    # create the dictionary gathering the information of the models trained
    # format as [Accuracy, Precision, Recall, F1 Score]
    model_dic = {"MNB": {"Accuracy": mnb_accuracy},
                 "LR": {"Accuracy": lr_accuracy},
                 "SVM": {"Accuracy": svm_accuracy},
                 "RF": {"Accuracy": rf_accuracy}}

    # find the model with the highest Accuracy
    max_accuracy = max(values["Accuracy"] for key, values in model_dic.items())
    print(max_accuracy)
    model_best_accuracy = [model for model, accuracy in model_dic.items() if accuracy["Accuracy"] == max_accuracy]
    print(model_best_accuracy)

    # save the model having the best F1 Score
    model_map = {"MNB": best_mnb, "LR": best_lr, "SVM": best_svm, "RF": best_rf}

    if len(model_best_accuracy) == 1:
        filename = 'Auto_Training/Models/best_type_classifier_' + str(datetime.datetime.now().date()) + '_' \
                   + str(datetime.datetime.now().time()).replace(':', '.') + '.joblib'
        joblib.dump(model_map[model_best_accuracy[0]], filename)
    else:
        # if the best models have the same F1 Score, sampe precision, and same recall, then save them all.
        for model_index, model in enumerate(model_best_accuracy):
            filename = 'Auto_Training/Models/best_type_classifier_' + str(model_index + 1) \
                       + str(datetime.datetime.now().date()) + '_' \
                       + str(datetime.datetime.now().time()).replace(':', '.') + '.joblib'
            joblib.dump(model_map[model], filename)

    return_result = {
        "status": 200
    }

    return Response(json.dumps(return_result), mimetype='application/json')


if __name__ == '__main__':
   application.run(debug = True)

