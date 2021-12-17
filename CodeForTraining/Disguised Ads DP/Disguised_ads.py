"""
Created on Tue Nov 2 11:09:31 2021

@author: seanq
"""
import pandas as pd
import numpy as np
from urlparse import *
import requests

#connect ocr to api first(pending)
#Merge with the application.py(pending)

application = Flask(__name__)

def say_hello(username = "Roger's World"):
   return '<p>Hello Lan %s!</p>\n' % username

application.add_url_rule('/', 'index', (lambda: say_hello()))

@application.route('/api/htmlsource',methods = ['POST'])
def get_html():
    data = request.get_data()
    html = json.loads(data)


@application.route('/api/url',methods = ['POST'])
def get_url():
    data = request.get_data()
    url_origin = json.loads(data)


@application.route('/api/checkitext', methods=['POST'])

def checkitext():

    #read text data(image to text by ocr) from api   (contains: content(texture), (tag(element in html)))
    data = request.get_data()
    t_data = json.loads(data)


    df = pd.DataFrame(t_data)

    #lower for searching 'download'
    df['content'] = df['content'].str.lower()
    dl = df['content'].str.contains('download')

    #use the element(tag name) to find its href, and get the domain name from url
    save=[]
    url_type=[]
    for tag in dl[tag]:
        url = html.xpath('//div/%s%dl[tag]/@href')

        #identify if the url is a download link or not(by content type of http header)
        HttpMessage = response.info(url)
        ContentType = HttpMessage.gettype()
        #if the type is text/html then it might be a disguised ads
        if ContentType = "text/html":
            save = save + dl[tag]



    #check if it is misdetection by compare the domain name of url and origin html url
    for tag in save
        domain_url = urlparse(save)
        url_type = url_type + domain_url

        domain_origin = urlparse(url_origin)
        fake = url_type[~url_type.str.contains(domain_origin)] #filter by the origin domain name
    # fake will contain all diisguised ads' tags for backend to mark
    #return fake
    return Response(json.dumps(fake), mimetype='application/json')









