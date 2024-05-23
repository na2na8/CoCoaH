import pickle
import argparse
from collections import defaultdict
import pandas as pd
import json

def split_titles(infos) :
    # sort by datetime
    infos.sort(key=lambda x:x[0], reverse=False)
    
    first = len(infos) * 8 // 10
    second = len(infos) // 10
    
    first_part = infos[ : first]
    second_part = infos[first : first + second]
    third_part = infos[first + second : ]
    
    print(len(infos))
    print(len(first_part), len(second_part), len(third_part))
    
    return first_part, second_part, third_part

def get_df(total_data, split, title_dict) :
    df = None
    for s_item in split :
        title = s_item[1]
        
        sub_df = total_data.iloc[title_dict[title]]
        sub_df['title'] = [title for idx in range(len(title_dict[title]))]
        if df is None :
            df = sub_df
        else :
            df = pd.concat([df, sub_df], ignore_index=True)
    return df
    
def get_split(total_data, first, second, third, title_dict) :
    train = get_df(total_data, first, title_dict)
    valid = get_df(total_data, second, title_dict)
    test = get_df(total_data, third, title_dict)
    
    train.to_csv('/home/nykim/2024_spring/00_data/processed_beep/beep_train.csv')
    valid.to_csv('/home/nykim/2024_spring/00_data/processed_beep/beep_valid.csv')
    test.to_csv('/home/nykim/2024_spring/00_data/processed_beep/beep_test.csv')

# KOLD processing     
def json2csv(kold) :
    dataframe = {
        'comment' : [],
        'title' : [],
        'OFF' : [],
        'TGT' : [],
        'GRP' : [],
        'OFF_span' : [],
        'TGT_span' : []
    }
    for item in kold :
        dataframe['comment'].append(item['comment']) 
        dataframe['title'].append(item['title'])
        dataframe['OFF'].append(item['OFF'])
        dataframe['TGT'].append(item['TGT'])
        dataframe['GRP'].append(item['GRP'])
        dataframe['OFF_span'].append(item['OFF_span'])
        dataframe['TGT_span'].append(item['TGT_span'])
    
    dataframe = pd.DataFrame(dataframe)
    return dataframe

def get_kold_df(total_data, split, title_dict) :
    df = None
    for s_item in split :
        title = s_item[1]
        
        sub_df = total_data.iloc[title_dict[title]]
        # sub_df['title'] = [title for idx in range(len(title_dict[title]))]
        if df is None :
            df = sub_df
        else :
            df = pd.concat([df, sub_df], ignore_index=True)
    return df

def get_kold_split(total_data, first, second, third, title_dict) :
    train = get_kold_df(total_data, first, title_dict)
    valid = get_kold_df(total_data, second, title_dict)
    test = get_kold_df(total_data, third, title_dict)
    
    train.to_csv('/home/nykim/2024_spring/00_data/processed_kold/kold_train.csv')
    valid.to_csv('/home/nykim/2024_spring/00_data/processed_kold/kold_valid.csv')
    test.to_csv('/home/nykim/2024_spring/00_data/processed_kold/kold_test.csv')
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_name', type=str, default='kold')
    args = parser.parse_args()
    
    title_dict = defaultdict(list)
    
    if args.d_name == 'beep' :
        with open('/home/nykim/2024_spring/03_data_processing/beep_info.pkl', 'rb') as p :
            infos = pickle.load(p)
        
        orig_train = pd.read_csv('/home/nykim/2024_spring/00_data/korean-hate-speech/labeled/train.tsv', sep='\t')
        with open('/home/nykim/2024_spring/00_data/korean-hate-speech/news_title/train.news_title.txt', 'r') as tt :
            title_t = tt.readlines()
        orig_dev = pd.read_csv('/home/nykim/2024_spring/00_data/korean-hate-speech/labeled/dev.tsv', sep='\t')
        with open('/home/nykim/2024_spring/00_data/korean-hate-speech/news_title/dev.news_title.txt', 'r') as vt :
            title_v = vt.readlines()
            
        orig_train['title'] = title_t
        orig_dev['title'] = title_v
        orig = pd.concat([orig_train, orig_dev], ignore_index=True)
        
        for idx in range(len(orig)) :
            title_dict[orig['title'].iloc[idx]] += [idx]
        
        first, second, third = split_titles(infos)
        get_split(orig, first, second, third, title_dict)
    
    elif args.d_name == 'kold' :
        with open('/home/nykim/2024_spring/03_data_processing/kold_info.pkl', 'rb') as p :
            infos = pickle.load(p)
        with open('/home/nykim/HateSpeech/00_data/KOLD/kold_v1.json') as f :
            kold = json.load(f)
        orig = json2csv(kold)
        
        for idx in range(len(orig)) :
            title_dict[orig['title'].iloc[idx]] += [idx]
            
        first, second, third = split_titles(infos)
        get_kold_split(orig, first, second, third, title_dict)
    
    