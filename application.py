# Web Scraper

import pandas as pd
import numpy as np

# web scraper

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import lxml.html
import re
import time

# write to csv file

import csv

# joblib is a set of tools to provide lightweight pipelining in Python. It provides utilities for saving and loading Python objects that make use of NumPy data structures, efficiently.
import joblib

from flask import Flask, request, Response
import json

application = Flask(__name__)

def say_hello(username = "World"):
    return '<p>Hello %s!</p>\n' % username

application.add_url_rule('/', 'index', (lambda: say_hello()))

@application.route('/api/parse',methods = ['POST'])
def parse():
    data = request.get_data()
    j_data = json.loads(data)
    print("input data", j_data)
    # -------------------------- Checking Presence ------------------

     # Loading the saved model with joblib
    presence_model = joblib.load('bnb_presence_classifier.joblib')
    presence_cv = joblib.load('presence_CountVectorizer.joblib')

    # New dataset to predict
    # presence_pred = pd.read_csv('testing01.csv')
    presence_pred = pd.DataFrame(j_data)

    # apply the pretrained model to the new content data
    pre_pred_vec = presence_model.predict(presence_cv.transform(presence_pred['content']))

    presence_pred['presence'] = pre_pred_vec.tolist()

    # dark pattern content are those where the predicted result equals to 0.
    dark = presence_pred.loc[presence_pred['presence']==0]

    # get the number of presence of dark pattern
    pre_count = dark.shape[0]

    # ------------------------- Category Classification ------------------

    # Loading the saved model with joblib
    cat_model = joblib.load('mnb_category_classifier.joblib')
    cat_cv = joblib.load('category_CountVectorizer.joblib')

    # mapping of the encoded dark pattern categories.
    cat_dic = {0:'Forced Action', 1:'Misdirection', 2:'Obstruction', 3:'Scarcity', 4:'Sneaking',
               5:'Social Proof', 6:'Urgency'}

    # apply the model and the countvectorizer to the detected dark pattern content data
    cat_pred_vec = cat_model.predict(cat_cv.transform(dark['content']))


    dark['category'] = cat_pred_vec.tolist()

    category_list = dark['category'].tolist()

    dark['category_name'] = [cat_dic[int(category)] for category in category_list]

    return_result = {
        "items_counts": {},
        "details": []
    }
    # get the list of the dark patterns detected with the frequency count

    counts = dark['category'].value_counts()
    for index,name in enumerate(counts.index.tolist()):
        return_result["items_counts"][int(name)] =int(counts.values[index])

    for index, value in enumerate(dark['category'], start=0):
        return_result["details"].append({
            "content": dark['content'][index],
            "key": dark['key'][index],
            "category": int(dark['category'][index]),
            "category_name": dark['category_name'][index]
        })
    print("return_result", return_result)
    return Response(json.dumps(return_result), mimetype='application/json')

if __name__ == '__main__':
   application.run(debug = True)
# -----------------------------------

# url = 'https://outfithustler.com/collections/women-fashion?gclid=EAIaIQobChMIx_r5nM_o8QIVKYBQBh3fGwWvEAAYAiAAEgJYEvD_BwE&page=1'
#
# # to avoid opening browser while using selenium
# option = webdriver.ChromeOptions()
# option.add_argument('headless')
# driver = webdriver.Chrome(ChromeDriverManager().install(),options=option)
#
# driver.get(url)
# time.sleep(1)
#
# # get source code -- type: str
# html_source = driver.page_source
#
# # key
# html = lxml.html.fromstring(html_source)
#
# # obtain all the text under the 'div' tags
# items = html.xpath("//div//text()")
#
# pattern = re.compile("^\s+|\s+$|\n")
#
# clause_text = ""
#
# for item in items:
#     line = re.sub(pattern, "", item)
#     if len(item) > 1:
#         clause_text += line +"\n"
#
# driver.quit()
#
# # -------------------------------------------
#
# raw_text = clause_text
#
#
# # the beginning character of the content, which is the sign we should ignore the content
# ignore_str = ',.;{}'
#
# # the content we are going to keep to send to models.
# content_list = []
#
# # only keep the content that has words count from 2 to 50 (includes).
# for line in raw_text.split('\n'):
#     if 1<len(line.split())<=50 and line[0] not in ignore_str:
#         content_list.append([line])
#
# header = ['content']

# create a csv file to save the filtered content for later model analysis.
# with open('testing01.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#
#     # write the header
#     writer.writerow(header)
#
#     # write the data
#     writer.writerows(content_list)



