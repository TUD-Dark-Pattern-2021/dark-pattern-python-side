
import pandas as pd
import numpy as np

from flask import Flask, request, Response
import json
from urllib.request import urlretrieve #for saving images with url

application = Flask(__name__)

def say_hello(username = "Roger's World"):
   return '<p>Hello Lan %s!</p>\n' % username

application.add_url_rule('/', 'index', (lambda: say_hello()))

@application.route('/api/image',methods = ['POST'])
def imageprocess():
   data = request.get_data()
   url_data = json.loads(data)

   pred_url = pd.DataFrame(url_data)

   url = pd.index(pred_url)

   #store the url as images
   for index,url in urlindex:
      img_url = url[index]
      file_name = url[index] + ".jpg"
      urlretrieve(img_url, file_name)

if __name__ == '__main__':
   application.run(debug = True)








