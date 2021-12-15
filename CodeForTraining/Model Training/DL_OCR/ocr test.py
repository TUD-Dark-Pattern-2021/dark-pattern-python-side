import pandas as pd
import numpy as np
import pytesseract
import cv2
import argparse
import os
from PIL import Image
import requests
import urllib.request as request

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\seanq\AppData\Local\Tesseract-OCR\tesseract.exe'
full = pd.DataFrame([['233', 'qqq', 'qwe', 'text'], [
    'https://img.ltwebstatic.com/images3_ach/2021/11/10/16365378007c7b2df3b913a3d430fe78fc28ff3472.jpg', 'www', 'sdd',
    'image'], ['https://m.media-amazon.com/images/I/41pBPGVc8zL._SX1500_.jpg', 'www', 'sdd', 'image']],
                    columns=["content", "tag", "key", "type"])
print(full)
# get urls with type = image
urlss = full.loc[full['type'] == 'image']

urlss = urlss.reset_index(drop=True)
print(urlss)
urls = urlss['content']
print(urls)


def get_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def thresholding(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def image_text(prep):
    return pytesseract.image_to_string(prep)

    # create a empty dataframe


df_image = pd.DataFrame(columns=["content", "tag", "key", "type"])


# print(df_image)


def texture_detect():
    a = 0
    for line in urls:
        print(line)
        r = requests.get(line)
        # response = request.urlopen('https://boohooamplience.a.bigcontent.io/v1/static/211111_Desktop_SinglesDay_70_ROW')

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
        print(itext)

        urlss['content'][a] = itext

        urlss["content"] = urlss["content"].map(lambda x:x.split('\n'))

        urlsss = urlss.explode("content")

        print(urlsss)
        a = a + 1

    print(urlss)


texture_detect()
