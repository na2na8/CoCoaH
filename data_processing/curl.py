from collections import defaultdict
import json
import requests
import re
from rouge import Rouge
from pprint import pprint
import pickle

rouge = Rouge()

with open('/home/nykim/2024_spring/99_get_urls/api_key.json') as f :
    api_key = json.load(f)

with open('/home/nykim/HateSpeech/00_data/KOLD/kold_v1.json') as f :
    kold = json.load(f)
    
with open('/home/nykim/2024_spring/00_data/korean-hate-speech/news_title/train.news_title.txt', 'r') as f :
    beep_title_lines = f.readlines()

kold_titles = defaultdict(int)
beep_titles = defaultdict(int)

for idx in range(len(kold)) :
    kold_titles[kold[idx]['title']] += 1

for line in beep_title_lines :
    beep_titles[line] += 1

kold_titles = dict(kold_titles)
beep_titles = dict(beep_titles)

def getRequestURL(query) :
    params = {
        'query' : query,
        'display' : 10,
        'sort' : 'sim'
    }
    
    headers = {
        'Accept' : 'application/json',
        'Content-Type' : 'application/json',
        'X-Naver-Client-Id' : api_key['clientID'],
        'X-Naver-Client-Secret' : api_key['clientSecret'],
        'User_Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
    }

    try :
        response = requests.get(url='https://openapi.naver.com/v1/search/news.json',
                                headers=headers,
                                params=params)
    except Exception as e :
        print(e)
    
    json_response = json.loads(response.text)
    items = json_response['items']
    return items

def getSamePost(query, items) :
    rouges = []
    for item in items :
        title = re.sub('<b>', '', item['title'])
        title = re.sub('</b>', '', title)
        title = re.sub('&quot', '', title)
        title = re.sub('\n', '', title)
        score = rouge.get_scores(title, query)
        rouges.append((item, score[0]['rouge-1']['f']))
    rouges.sort(key=lambda x:x[1], reverse=True)
    return rouges[0][0]
    
def getPostURL(dictionary) :
    urls = {}
    titles = list(dictionary.keys())
    for title in titles :
        try :
            result = getRequestURL(title)
            post = getSamePost(title, result)
            urls[title] = post['link']
        except Exception as e :
            print(f'title : {title} gets exception {e}')
    return urls

def getInfo(dictionary) :
    infos = {}
    titles = list(dictionary.keys())
    date_format = '%a, %d %b %Y %H:%M:%S %z'
    for title in titles :
        try :
            result = getRequestURL(title)
            post = getSamePost(title, result)
            infos[title] = (datetime.strptime(post['pubDate'], date_format), )
            # urls[title] = post['link']
        except Exception as e :
            print(f'title : {title} gets exception {e}')
    return urls

# print('kold_urls')
# kold_urls = getPostURL(kold_titles)
print('beep_urls')
beep_urls = getPostURL(beep_titles)

print('save to pickle')
# with open('/home/nykim/HateSpeech/08_Implicit/kold_urls.pkl', 'wb') as wp :
#     pickle.dump(kold_urls, wp)
with open('/home/nykim/HateSpeech/08_Implicit/beep_urls.pkl', 'wb') as wp :
    pickle.dump(beep_urls, wp)
# query = '페미니즘이 범죄가 되는 나라 [삶과 문화]'
# result = getRequestURL(query)
# post = getSamePost(query, result)

