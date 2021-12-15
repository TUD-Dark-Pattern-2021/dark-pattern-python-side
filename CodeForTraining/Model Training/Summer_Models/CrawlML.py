# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 21:49:16 2021

@author: seanq
"""
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

url = 'https://www.lights.ie/umage'

# to avoid opening browser while using selenium
option = webdriver.ChromeOptions()
option.add_argument('headless')
driver = webdriver.Chrome(ChromeDriverManager().install(),options=option)

driver.get(url)
time.sleep(1)

# get source code -- type: str
html_source = driver.page_source

# key
html = lxml.html.fromstring(html_source)

# obtain all the text under the 'div' tags
items = html.xpath("//div//text()")

pattern = re.compile("^\s+|\s+$|\n")

clause_text = ""

for item in items:
    line = re.sub(pattern, "", item)
    if len(item) > 1:
        clause_text += line +"\n"

driver.quit()

raw_text = clause_text

# the beginning character of the content, which is the sign we should ignore the content
ignore_str = ',.;{}'

# the content we are going to keep to send to models.
content_list = []

# only keep the content that has words count from 2 to 50 (includes).
for line in raw_text.split('\n'):
    if 1<len(line.split())<=50 and line[0] not in ignore_str:
        content_list.append([line])

header = ['content']

# create a csv file to save the filtered content for later model analysis.
with open('../../../../Driver/newdata1.csv', 'a', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    writer.writerows(content_list)

# Loading the saved model with joblib
presence_model = joblib.load('../../../../Driver/bnb_presence_classifier.joblib')
presence_cv = joblib.load('../../../../Driver/presence_CountVectorizer.joblib')

# New dataset to predict
presence_pred = pd.read_csv('../../../../Driver/newdata1.csv')

# apply the pretrained model to the new content data
pre_pred_vec = presence_model.predict(presence_cv.transform(presence_pred['content']))

presence_pred['presence'] = pre_pred_vec.tolist()

# dark pattern content are those where the predicted result equals to 0.
dark = presence_pred.loc[presence_pred['presence']==0]

# get the number of presence of dark pattern
pre_count = dark.shape[0]

print(pre_count)

df = dark.drop_duplicates(subset="content")

df.to_csv('newpresence02.csv', index=False, header=True)

