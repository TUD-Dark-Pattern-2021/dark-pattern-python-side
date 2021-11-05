"""
Created on Tue Oct 31 22:19:10 2021

@author: seanq
"""
import pandas as pd
import numpy as np
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import lxml.html
import re
import time

# write to csv file
import csv


#update Crawl: make list for 20 websites' urls , add text in <span>
"""
url = ['https://www.lights.ie/umage',
'https://www.scan.co.uk/shop/music-and-pro-audio',
'https://www.smokecartel.com/collections/staff-picks',
'https://www.burton.co.uk/mens/sale',
'https://www.amazon.co.uk/s?k=graphic+card&ref=nb_sb_noss_2',
'https://www.bestbuy.com/site/misc/deal-of-the-day/pcmcat248000050016.c?id=pcmcat248000050016',
'https://outfithustler.com/collections/women-fashion?gclid=EAIaIQobChMIx_r5nM_o8QIVKYBQBh3fGwWvEAAYAiAAEgJYEvD_BwE&pf_pt_categories=Accessories&page=2'',
'https://www.logoup.com/Ladies-Core-Performance-Soft-Shell_p_989.html',
‘https://www.scan.co.uk/shop/music-and-pro-audio’
‘https://marleylilly.com/’  

]
"""


url = 'https://marleylilly.com/'

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

items2 = html.xpath("//span//text()")

pattern2 = re.compile("^\s+|\s+$|\n")

clause_text2 = ""

for item in items2:
    line = re.sub(pattern2, "", item)
    if len(item) > 1:
        clause_text2 += line +"\n"

driver.quit()

raw_text = clause_text + clause_text2



# the beginning character of the content, which is the sign we should ignore the content
ignore_str = ',.;{}/'

# the content we are going to keep to send to models.
content_list = []

# only keep the content that has words count from 2 to 50 (includes).
for line in raw_text.split('\n'):
    if 1<len(line.split())<=40 and line[0] not in ignore_str:
        content_list.append([line])

header = ['content']

# create a csv file. make sure the csv can be added new content
with open(r'C:\Users\seanq\Desktop\Evaluation\testingdata.csv', 'a', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    writer.writerows(content_list)
