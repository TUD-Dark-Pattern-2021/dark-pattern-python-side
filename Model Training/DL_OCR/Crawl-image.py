import requests


def get_pictures(url,path):
    headers={
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400'}
    re=requests.get(url,headers=headers)
    print(re.status_code)  #check the status of website -> 200
    with open(path, 'wb') as f:#wirte the image into local
                for chunk in re.iter_content(chunk_size=128):
                    f.write(chunk)
def get_pictures_urls(text):
    st='img src="'
    m=len(st)
    i=0
    n=len(text)
    urls=[]#store the url
    while i<n:
        if text[i:i+m]==st:
            url=''
            for j in range(i+m,n):
                if text[j]=='"':
                    i=j
                    urls.append(url)
                    break
                url+=text[j]
        i+=1
    return urls

headers={
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400'}
url=''
re=requests.get(url,headers=headers)
urls=get_pictures_urls(re.text)#get all urls from website
for i in range(len(urls)):#crawl all images
    url='https:'+urls[i]
    path='crawl'+str(i)+'.jpg'
    get_pictures(url,path)

