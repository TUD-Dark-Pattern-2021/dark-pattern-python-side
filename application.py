import pandas as pd
import numpy as np

# joblib is a set of tools to provide lightweight pipelining in Python. It provides utilities for saving and loading Python objects that make use of NumPy data structures, efficiently.
import joblib

from flask import Flask, request, Response
import json

application = Flask(__name__)

def say_hello(username = "Roger's World"):
   return '<p>Hello Lan %s!</p>\n' % username

application.add_url_rule('/', 'index', (lambda: say_hello()))

@application.route('/api/checkDP',methods = ['POST'])
def checkDP():
    data = request.get_data()
    j_data = json.loads(data)

    presence_model = joblib.load('rf_presence_classifier.joblib')
    presence_cv = joblib.load('dark_CountVectorizer.joblib')

    pre_pred = presence_model.predict(presence_cv.transform([j_data['content']]))

    if pre_pred == [0]:
        return_result = {
            "isDarkPattern": 'Yes'
        }
    else:
        return_result = {
            "isDarkPattern": 'No'
        }
    return Response(json.dumps(return_result), mimetype='application/json')


@application.route('/api/parse',methods = ['POST'])
def parse():
    data = request.get_data()
    j_data = json.loads(data)
    # print("input data", j_data)
    # -------------------------- Checking Presence ------------------

     # Loading the saved model with joblib
    presence_model = joblib.load('rf_presence_classifier.joblib')
    presence_cv = joblib.load('dark_CountVectorizer.joblib')

    # New dataset to predict
    presence_pred = pd.DataFrame(j_data)

    # Remove the rows where the first letter starting with ignoring characters
    ignore_str = [',', '.', ';', '{', '}', '#', '/', '(', ')', '?']
    presence_pred = presence_pred[~presence_pred['content'].str[0].isin(ignore_str)]

    # Keep the rows where the word count is between 2 and 20 (inclusive)
    presence_pred = presence_pred[presence_pred['content'].str.split().str.len() > 1]
    presence_pred = presence_pred[presence_pred['content'].str.split().str.len() < 21]

    # Filter out the disturibing content to be removed.
    str_list = ['low to high', 'high to low', 'high low', 'low high', '{', 'ships', 'ship', '®', 'details',
                'limited edition', 'cart is currently empty', 'in cart', 'out of stock', 'believe in',
                'today\'s deals', 'customer service', 'offer available', 'offers available', 'collect',
                '% off', 'in stock soon', 'problem', 'UTC', 'javascript', 'cookie', 'cookies', 'disclaimer']
    pattern = '|'.join(str_list)

    presence_pred = presence_pred[~presence_pred.content.str.lower().str.contains(pattern)]

    # apply the pre-trained model to the new content data
    pre_pred_vec = presence_model.predict(presence_cv.transform(presence_pred['content']))

    presence_pred['presence'] = pre_pred_vec.tolist()

    # dark pattern content are those where the predicted result equals to 0.
    dark = presence_pred.loc[presence_pred['presence'] == 0]

    # get the number of presence of dark pattern
    pre_count = dark.shape[0]

    # ------------------------- Category Classification ------------------
    if pre_count == 0:
        return_result = {
            "total_counts": {},
            "items_counts": {},
            "details": []
        }
    else:
        # Loading the saved model with joblib
        cat_model = joblib.load('lr_category_classifier.joblib')
        cat_cv = joblib.load('type_CountVectorizer.joblib')

        # mapping of the encoded dark pattern categories.
        cat_dic = {0:'FakeActivity', 1:'FakeCountdown', 2:'FakeHighDemand', 3:'FakeLimitedTime', 4:'FakeLowStock'}

        cat_slug = {0:'Fake Activity', 1:'Fake Countdown', 2:'Fake High-demand', 3:'Fake Limited-time', 4:'Fake Low-stock'}

        # apply the model and the countvectorizer to the detected dark pattern content data
        cat_pred_vec = cat_model.predict(cat_cv.transform(dark['content']))                   # Problem


        dark['category'] = cat_pred_vec.tolist()

        category_list = dark['category'].tolist()

        # get the mapping of the category name and encoded category integers
        dark['category_name'] = [cat_dic[int(category)] for category in category_list]

        dark['category_name_slug'] = [cat_slug[int(category)] for category in category_list]

        # reset the index of the detected dark pattern list on the webpage.
        dark = dark.reset_index(drop=True)

        return_result = {
            "total_counts": {},
            "items_counts": {},
            "details": []
        }
        # get the list of the dark patterns detected with the frequency count

        return_result['total_counts'] = pre_count

        counts = dark['category_name'].value_counts()
        for index, category_name in enumerate(counts.index.tolist()):
            return_result["items_counts"][category_name] = int(counts.values[index])

        for j in range(len(dark)):
            return_result["details"].append({
                "content": dark['content'][j],
                "tag":dark['tag'][j],
                "key": dark['key'][j],
                "category_name": dark['category_name'][j],
                "category_name_slug": dark['category_name_slug'][j]
            })
        print("return_result", return_result)
    return Response(json.dumps(return_result), mimetype='application/json')

if __name__ == '__main__':
   application.run(debug = True)

