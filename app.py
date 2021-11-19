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

application = Flask(__name__)

def say_hello(username = "Roger's World"):
   return '<p>Hello Lan %s!</p>\n' % username

application.add_url_rule('/', 'index', (lambda: say_hello()))

@application.route('/api/checkDP',methods = ['POST'])
def checkDP():
    data = request.get_data()
    j_data = json.loads(data)

    presence_model = joblib.load('rf_presence_classifier.joblib')
    presence_cv = joblib.load('presence_TfidfVectorizer.joblib')

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
        #for running local
        #if platform.system().lower() == 'windows':
            #pytesseract.pytesseract.tesseract_cmd = r'C:\Users\seanq\AppData\Local\Tesseract-OCR\tesseract.exe'
        #pytesseract.pytesseract.tesseract_cmd = r'C:\Users\seanq\AppData\Local\Tesseract-OCR\tesseract.exe'
        data = request.get_data()
        j_data = json.loads(data)


        # get urls with type = image

        full = pd.DataFrame(j_data)
        print(full)
        urlss = full.loc[full['type'] == 'image']

        urlss = urlss.reset_index(drop=True)
        print(urlss)
        urls = urlss['content']

        print(urls)



        # def get_grayscale(img):
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # def thresholding(img):
        # return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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
                    #print(line)
                    r = requests.get(line)
                    image_name = '0.jpg'
                    image_path = "./" + image_name
                    with open(image_path, 'wb') as f:
                        f.write(r.content)
                    f.close()

                    # grayscale the image
                    # gray = get_grayscale(image_path)
                    # threshold the processed image
                    # prep = thresholding(gray)

                    itext = image_text(image_path)
                    os.remove(image_path)
                    # image detection
                    # print(itext)

                    urlss['content'][a] = itext


                except:
                    continue

            urlss["content"] = urlss["content"].map(lambda x: x.split('\n'))

            urlsss = urlss.explode("content")

            return urlsss

        texture_detect = texture_detect(urls)

        return texture_detect

    ocr = ocr()


    data = request.get_data()
    j_data = json.loads(data)
    # print("input data", j_data)
    # -------------------------- Checking Presence ------------------

     # Loading the saved model with joblib
    presence_model = joblib.load('rf_presence_classifier.joblib')
    presence_cv = joblib.load('presence_TfidfVectorizer.joblib')

    # New dataset to predict
    presence_pred = pd.DataFrame(j_data)

    #filter type == text
    #textpp = pp.loc[pp['type'] == 'text']

    #combine = [textpp, ocr]

    #presence_pred = pd.concat(combine)

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

    # ------------------------- Category Classification ------------------
    if pre_count == 0:
        return_result = {
            "total_counts": {},
            "items_counts": {},
            "details": []
        }
    else:
        # Loading the saved model with joblib
        cat_model = joblib.load('lr_type_classifier.joblib')
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


@application.route('/api/parse',methods = ['POST'])

def ocr():
    #for running local
    #if platform.system().lower() == 'windows':
        #pytesseract.pytesseract.tesseract_cmd = r'C:\Users\seanq\AppData\Local\Tesseract-OCR\tesseract.exe'
    #pytesseract.pytesseract.tesseract_cmd = r'C:\Users\seanq\AppData\Local\Tesseract-OCR\tesseract.exe'
    data = request.get_data()
    j_data = json.loads(data)


    # get urls with type = image

    full = pd.DataFrame(j_data)
    print(full)
    urlss = full.loc[full['type'] == 'image']

    urlss.duplicated(['content'])

    urlss = urlss.drop_duplicates(['content'])

    urlss = urlss.reset_index(drop=True)


    print(urlss)

    urls = urlss['content']


    print(urls)

    # def get_grayscale(img):
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # def thresholding(img):
    # return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def image_text(prep):
        return pytesseract.image_to_string(prep)

    # create a empty dataframe
    # df_image = pd.DataFrame(columns=["content", "tag", "key", "type"])

    def texture_detect(all_urls):
        #In case there will be some errors, count cannot start as 0
        a = -1
        for line in all_urls:
            a = a + 1
            #check if url has "https:" (only browser can automatically add it in front of the urls.)
            if 'https:' not in line:
                line = "https:" + line
            print(line)

            #skip if error
            try:
                ##debug
                #print(line)

                r = requests.get(line)
                image_name = '0.jpg'
                image_path = "./" + image_name
                with open(image_path, 'wb') as f:
                    f.write(r.content)
                f.close()

                # grayscale the image
                # gray = get_grayscale(image_path)
                # threshold the processed image
                # prep = thresholding(gray)

                itext = image_text(image_path)
                os.remove(image_path)
                # image detection
                # print(itext)

                urlss['content'][a] = itext

            except:
                continue

        #ocr will return several lines in a instance, need to split them line by line with the same tags
        urlss["content"] = urlss["content"].map(lambda x: x.split('\n'))

        urlsss = urlss.explode("content")

        return urlsss

    texture_detect = texture_detect(urls)

    #load joblib
    presence_model = joblib.load('rf_presence_classifier.joblib')
    presence_cv = joblib.load('presence_TfidfVectorizer.joblib')


    # Remove the rows where the first letter starting with ignoring characters
    ignore_str = [',', '.', ';', '{', '}', '#', '/', '(', ')', '?']
    texture_detect = texture_detect[~texture_detect['content'].str[0].isin(ignore_str)]

    # Keep the rows where the word count is between 2 and 20 (inclusive)
    texture_detect = texture_detect[texture_detect['content'].str.split().str.len() > 1]
    texture_detect = texture_detect[texture_detect['content'].str.split().str.len() < 21]

    str_list = ['{', 'UTC']
    pattern = '|'.join(str_list)

    texture_detect = texture_detect[~texture_detect.content.str.lower().str.contains(pattern)]

    # apply the pre-trained model to the new content data
    pre_pred_vec = presence_model.predict(presence_cv.transform(texture_detect['content']))

    texture_detect['presence'] = pre_pred_vec.tolist()

    # dark pattern content are those where the predicted result equals to 0.
    dark = texture_detect.loc[texture_detect['presence'] == 0]

    # get the number of presence of dark pattern
    pre_count = dark.shape[0]

    # ------------------------- Category Classification ------------------
    if pre_count == 0:
        return_imgresult = {
            "total_counts": {},
            "items_counts": {},
            "details": []
        }
    else:
        # Loading the saved model with joblib
        cat_model = joblib.load('lr_type_classifier.joblib')
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

        return_imgresult = {
            "total_counts": {},
            "items_counts": {},
            "details": []
        }
        # get the list of the dark patterns detected with the frequency count

        return_imgresult['total_counts'] = pre_count

        counts = dark['category_name'].value_counts()
        for index, category_name in enumerate(counts.index.tolist()):
            return_imgresult["items_counts"][category_name] = int(counts.values[index])

        for j in range(len(dark)):
            return_imgresult["details"].append({
                "content": dark['content'][j],
                "tag":dark['tag'][j],
                "key": dark['key'][j],
                "category_name": dark['category_name'][j],
                "category_name_slug": dark['category_name_slug'][j]
            })
        print("return_imgresult", return_imgresult)
    return Response(json.dumps(return_imgresult), mimetype='application/json')


