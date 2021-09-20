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


@application.route('/api/parse',methods = ['POST'])
def parse():
    data = request.get_data()
    j_data = json.loads(data)
    # print("input data", j_data)
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
    dark = presence_pred.loc[presence_pred['presence'] == 0]

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

    dark = dark.reset_index(drop=True)

    return_result = {
        "total_counts": {},
        "items_counts": {},
        "details": []
    }
    # get the list of the dark patterns detected with the frequency count

    return_result['total_counts'] = pre_count

    counts = dark['category'].value_counts()
    # for index,name in enumerate(counts.index.tolist()):
        # return_result["items_counts"][int(name)] =int(counts.values[index])

    category = counts.keys().tolist()
    number = counts.tolist()
    for i in range(len(category)):
        return_result["items_counts"][category[i]] = number[i]

    for j in range(len(dark)):
        return_result["details"].append({
            "content": dark['content'][j],
            "tag":dark['tag'][j],
            "key": dark['key'][j],
            "category": int(dark['category'][j]),
            "category_name": dark['category_name'][j]
        })
    print("return_result", return_result)
    return Response(json.dumps(return_result), mimetype='application/json')

if __name__ == '__main__':
   application.run(debug = True)

