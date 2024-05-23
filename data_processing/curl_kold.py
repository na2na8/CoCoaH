from collections import defaultdict
import json
import requests
import re
from rouge import Rouge
from pprint import pprint
import pickle
from datetime import datetime
rouge = Rouge()

with open('/home/nykim/2024_spring/03_data_processing/api_key.json') as f :
    api_key = json.load(f)
    
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

def getInfo(titles) :
    infos = []
    date_format = '%a, %d %b %Y %H:%M:%S %z'
    for title in titles :
        try :
            result = getRequestURL(title)
            post = getSamePost(title, result)
            infos.append((datetime.strptime(post['pubDate'], date_format), title, post['description'])) 
            # urls[title] = post['link']
        except Exception as e :
            print(f'title : {title} gets exception {e}')
    return infos

# KOLD Title gethering
with open('/home/nykim/HateSpeech/00_data/KOLD/kold_v1.json') as f :
    kold = json.load(f)

kold_titles = []
for idx in range(len(kold)) :
    kold_titles.append(kold[idx]['title'])

kold_title_set = list(set(kold_titles))

kold_info = getInfo(kold_title_set)

with open('/home/nykim/2024_spring/03_data_processing/kold_info.pkl', 'wb') as wp :
    pickle.dump(kold_info, wp)


    