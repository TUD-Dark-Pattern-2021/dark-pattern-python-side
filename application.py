import pandas as pd
import numpy as np
import platform

# joblib is a set of tools to provide lightweight pipelining in Python. It provides utilities for saving and loading Python objects that make use of NumPy data structures, efficiently.
import joblib

from flask import Flask, request, Response
import json
import requests
import shortuuid
from PIL import Image
import pytesseract
import os
#import cv2

application = Flask(__name__)

def say_hello(username = "Roger's World"):
   return '<p>Hello Lan %s!</p>\n' % username

application.add_url_rule('/', 'index', (lambda: say_hello()))

@application.route('/api/checkDP',methods = ['POST'])
def checkDP():
    data = request.get_data()
    j_data = json.loads(data)
    df = pd.DataFrame(j_data)
    # ------ Check the 5 pattern types -------
    presence_model = joblib.load('rf_presence_classifier.joblib')
    presence_cv = joblib.load('presence_TfidfVectorizer.joblib')
    prediction = presence_model.predict(presence_cv.transform(df['content']))
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
    if platform.system().lower() == 'windows':
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\seanq\AppData\Local\Tesseract-OCR\tesseract.exe'
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
        #print(full)
        urlss = full.loc[full['type'] == 'image']
        #print(urlss)
        urlss.duplicated(['content'])
        urlss3 = urlss.drop_duplicates(['content'])
        urlss2 = urlss3.reset_index(drop=True)
        #print(urlss2)
        #print('work')
        urls = urlss2['content']
        #print(urls)
        def get_grayscale(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        def thresholding(img):
            return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        def image_text(prep):
            return pytesseract.image_to_string(prep)

        # create a empty dataframe
        # df_image = pd.DataFrame(columns=["content", "tag", "key", "type"])
        def texture_detect(all_urls):
            a = -1
            for line in all_urls:
                a = a + 1
                if 'https:' not in line:
                    line = "https:" + line
                print(line)
                try:
                    # print(line)
                    r = requests.get(line)
                    image_name = 'image.jpg'
                    image_path = "./" + image_name
                    with open(image_path, 'wb') as f:
                        f.write(r.content)
                    f.close()

                    #img = cv2.resize(f, None, fx=2, fy=2)
                    #image = cv2.imread(image_path)

                    # grayscale the image
                    #gray = get_grayscale(image)
                    # threshold the processed image
                    #prep = thresholding(gray)

                    #filename = "{}.png".format(os.getpid())
                    #cv2.imwrite(filename, gray)

                    itext = image_text(image_path)
                    #itext = image_text(Image.open(filename))
                    os.remove(image_path)
                    #os.remove(filename)
                    # image detection
                    # print(itext)
                    urlss2['content'][a] = itext
                    print(itext)
                except:
                    continue

            urlss2["content"] = urlss2["content"].map(lambda x: x.split('\n'))
            urlsss = urlss2.explode("content")
            #print(urlsss)
            return urlsss
        texture_detect = texture_detect(urls)
        #print(texture_detect)
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

        # Loading the saved model with joblib
        detection_model = joblib.load('confirm_rf_clf.joblib')
        detection_cv = joblib.load('confirm_cv.joblib')

        # apply the pre-trained confirmshaming detection model to the button / link text data
        pred_vec = detection_model.predict(detection_cv.transform(link_text['content']))
        link_text['presence'] = pred_vec.tolist()

        # dark pattern content are those where the predicted result equals to 0.
        confirm_shaming = link_text.loc[link_text['presence'] == 0]
        confirm_shaming = confirm_shaming.reset_index(drop=True)

        # get the number of presence of dark pattern
        confirm_shaming_count = confirm_shaming.shape[0]
        print('Number of confirmshaming DP: ', confirm_shaming_count)
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
        textpp = presence_pred.loc[presence_pred['type'] == 'text']
        combine = [textpp, texture_detect]
        presence_pred = pd.concat(combine)

    else:
        presence_pred = presence_pred.loc[presence_pred['type'] == 'text']

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
    if pre_count == 0 and confirm_count == 0:
        return_result = {
            "total_counts": {},
            "items_counts": {},
            "details": []
        }
    else:
        # Loading the saved model with joblib
        type_model = joblib.load('lr_type_classifier.joblib')
        type_cv = joblib.load('type_CountVectorizer.joblib')

        # mapping of the encoded dark pattern types.
        type_dic = {0:'FakeActivity', 1:'FakeCountdown', 2:'FakeHighDemand', 3:'FakeLimitedTime', 4:'FakeLowStock'}

        type_slug = {0:'Fake Activity', 1:'Fake Countdown', 2:'Fake High-demand', 3:'Fake Limited-time', 4:'Fake Low-stock'}

        # apply the model and the countvectorizer to the detected dark pattern content data
        type_pred_vec = type_model.predict(type_cv.transform(dark['content']))                   # Problem

        dark['type'] = type_pred_vec.tolist()
        type_list = dark['type'].tolist()

        # get the mapping of the type name and encoded type integers
        dark['type_name'] = [type_dic[int(type)] for type in type_list]
        dark['type_name_slug'] = [type_slug[int(type)] for type in type_list]

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
                    "key": confirm_shaming['key'][j],
                    "type_name": "Confirmshaming",
                    "type_name_slug": "Confirmshaming"
                })
        print("return_result", return_result)
        # ----------------
    return Response(json.dumps(return_result), mimetype='application/json')

if __name__ == '__main__':
   application.run(debug = True)

