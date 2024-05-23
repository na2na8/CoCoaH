import pandas as pd
import argparse
from collections import defaultdict

def get_split(orig) :
    title_set = list(set(orig['title']))
    
    first = len(title_set) * 8 // 10
    second = len(title_set) // 10
    
    first_part = title_set[ : first]
    second_part = title_set[first : ]
    # third_part = title_set[first + second : ]
    
    print(len(title_set))
    print(len(first_part), len(second_part))
    return first_part, second_part

def get_subdf(orig, split, title_dict) :
    df = None
    for s in split :
        subdf = orig.iloc[title_dict[s]]
        if df is None :
            df = subdf
        else :
            df = pd.concat([df, subdf])
    return df

def get_dataframe(orig, first, second, title_dict, args) :
    train = get_subdf(orig, first, title_dict)
    valid = get_subdf(orig, second, title_dict)
    # test = get_subdf(orig, test, title_dict)
    
    train.to_csv(f'/home/nykim/2024_spring/00_data/processed_en/{args.d_name}_IMSyPP_train.csv')
    valid.to_csv(f'/home/nykim/2024_spring/00_data/processed_en/{args.d_name}_IMSyPP_valid.csv')
    # test.to_csv(f'/home/nykim/2024_spring/00_data/processed_en/{args.d_name}_IMSyPP_test.csv')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_name', type=str, default='en')
    args = parser.parse_args()
    
    train = pd.read_csv(f'/home/nykim/2024_spring/00_data/{args.d_name}_IMSyPP_train.csv')
    valid = pd.read_csv(f'/home/nykim/2024_spring/00_data/{args.d_name}_IMSyPP_valid.csv')

    orig = pd.concat([train, valid])
    orig.dropna(subset=['title', 'comment', 'hate'], inplace=True)
    
    title_dict = defaultdict(list)
    for idx in range(len(orig)) :
        title_dict[orig['title'].iloc[idx]] += [idx]
    
    first, second = get_split(orig)
    get_dataframe(orig, first, second, title_dict, args)
    
    
    