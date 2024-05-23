import os
import pandas as pd
import openai
from openai import OpenAI
import logging
import re
import argparse
import pickle


QUESTION_DICT = {
    'beep' : "다음 게시물이 모욕적인지 여부를 결정하고, 그 이유를 설명하십시오.",
    'kold' : "다음 게시물이 모욕적인지 여부를 결정하고, 그 이유를 설명하십시오.",
    'en_IMSyPP' : "Determine whether the following post is offensive, and explain why.",
    'en_IMSyPP_nocon' : "Determine whether the following post is offensive, and explain why.",
    'it_IMSyPP' : "Determina se il seguente post è offensivo e spiega il motivo."
}


LABEL_DICT = {
    'kold' : {True : 1, False : 0},
    'beep' : {'offensive' : 2, 'hate' : 1, 'none' : 0},
    'en_IMSyPP' : {
                '0. appropriate' : 0,
                '1. inappropriate' : 1,
                '2. offensive' : 2, 
                '3. violent' : 3
            },
    'en_IMSyPP_nocon' : {
                '0. appropriate' : 0,
                '1. inappropriate' : 1,
                '2. offensive' : 2, 
                '3. violent' : 3
            },
    'it_IMSyPP' : {
                '0. appropriato' : 0,
                '1. inappropriato' : 1,
                '2. offensivo' : 2,
                '3. violento' : 3
            }
}

def get_dataset(d_name, stage=None) :
    if d_name == 'beep' :
        dataset = pd.read_csv(f'/home/nykim/HateSpeech/09_TitlePrediction/00_data/beep_{stage}.csv')
        dataset.drop_duplicates(subset=['comments'], inplace=True)
        dataset.dropna(subset=['comments', 'hate', 'title'], inplace=True)
        return dataset
    
    elif d_name == 'kold' :
        dataset = pd.read_csv(f'/home/nykim/HateSpeech/09_TitlePrediction/00_data/kold_{stage}.csv')
        dataset.drop_duplicates(subset=['comment'], inplace=True)
        dataset.dropna(subset=['comment', 'OFF', 'title'], inplace=True)
        return dataset
    
    elif d_name == 'en_IMSyPP_nocon' :
        dataset = pd.read_csv('/home/nykim/HateSpeech/09_TitlePrediction/00_data/en_IMSyPP_nocon_test.csv')
        dataset.drop_duplicates(subset=['comment'], inplace=True)
        dataset = dataset[(dataset['hate']=='0. appropriate')|(dataset['hate']=='1. inappropriate')|
                              (dataset['hate']=='2. offensive')|(dataset['hate']=='3. violent')]
        dataset.dropna(subset=['comment', 'hate', 'title'], inplace=True)
        return dataset
    
    elif d_name == 'en_IMSyPP' :
        dataset = pd.read_csv(f'/home/nykim/HateSpeech/09_TitlePrediction/00_data/en_IMSyPP_{stage}.csv')
        dataset.drop_duplicates(subset=['comment'], inplace=True)
        dataset = dataset[(dataset['hate']=='0. appropriate')|(dataset['hate']=='1. inappropriate')|
                              (dataset['hate']=='2. offensive')|(dataset['hate']=='3. violent')]
        dataset.dropna(subset=['comment', 'hate', 'title'], inplace=True)
        return dataset
    
    elif d_name == 'it_IMSyPP' :
        dataset = pd.read_csv(f'/home/nykim/HateSpeech/09_TitlePrediction/00_data/it_IMSyPP_{stage}.csv')
        dataset.drop_duplicates(subset=['comment'], inplace=True)
        dataset = dataset[(dataset['hate']=='0. appropriato')|(dataset['hate']=='1. inappropriato')|
                              (dataset['hate']=='2. offensivo')|(dataset['hate']=='3. violento')]
        dataset.dropna(subset=['comment', 'hate', 'title'], inplace=True)
        return dataset
    
# dataset
en_imsypp_nocon = get_dataset('en_IMSyPP_nocon')

    
def get_prompt(d_name, comment) :
    question = QUESTION_DICT[d_name]
    if d_name == 'kold' or d_name == 'beep' : 
        prompt = f"{question}\nPost: {comment}\nAnswer: 한 단계씩 설명해보겠습니다."
    elif d_name == 'it_IMSyPP' :
        prompt = f"{question}\nPost: {comment}\nAnswer: Spieghiamo passo dopo passo."
    elif d_name == 'en_IMSyPP' or d_name == 'en_IMSyPP_nocon' :
        prompt = f"{question}\nPost: {comment}\nAnswer: Let's explain step by step"
    # prompt = f"{question}\nPost: {comment}\nOptions:\n{options}\nAnswer: "
    return prompt



def HARE(client, d_name, comment_cname, label_cname) :
    # label_dict = LABEL_DICT[d_name]
    # if d_name == 'en_IMSyPP' :
    #     en_imsypp_nocon = get_dataset('en_IMSyPP_nocon')
    #     nocon_hate = []
    # d_list = [get_dataset(d_name, 'test'), get_dataset(d_name, 'valid'), get_dataset(d_name, 'train')]
    d_list = [get_dataset(d_name, 'test')]
    # num_dict = {0 : 'test', 1 : 'valid', 2 : 'train'}
    savepoint = None
    if os.path.exists(f'/home/nykim/HateSpeech/09_TitlePrediction/07_sota/HARE/flan_t5/pickles/{d_name}_test.pkl') :
        with open(f'/home/nykim/HateSpeech/09_TitlePrediction/07_sota/HARE/flan_t5/pickles/{d_name}_test.pkl', 'rb') as rp :
            savepoint = pickle.load(rp)
            
    num_dict = { 0 : 'test'}
        
    for num, dataframe in enumerate(d_list) :
        comment = []
        hate = []
        frhare = []
        # nocon_hate = []
        if savepoint :
            start_idx = savepoint[0]
            comment = savepoint[1]
            hate = savepoint[2]
            frhare = savepoint[3]
        else :
            start_idx = 0    
            
        for idx in range(start_idx, len(dataframe)) :
            cmt = dataframe[comment_cname].iloc[idx]
            label = dataframe[label_cname].iloc[idx] # int label
            
            prompt = get_prompt(d_name, cmt)
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=360,
                
            )
            
            ans = stream.choices[0].message.content
            
            frhare.append(ans)
            hate.append(label)
            comment.append(cmt)
            # if num == 0 and d_name == 'en_IMSyPP' :
            #     nocon_label = en_imsypp_nocon[label_cname].iloc[idx]
            #     nocon_hate.append(nocon_label)
            
            

            logger.info(f'len_gold: {len(hate)} | len_answers: {len(frhare)} | len_comment: {len(comment)}')
            assert len(frhare) == len(hate) == len(comment)
            
            if idx % 100 == 0 :
                with open(f'/home/nykim/HateSpeech/09_TitlePrediction/07_sota/HARE/flan_t5/pickles/{d_name}_test.pkl', 'wb') as wp :
                    new_save = (idx+1, comment, hate, frhare)
                    pickle.dump(new_save, wp)
        
        if d_name == 'beep' :
            df = {
                'comments' : comment,
                'hate' : hate,
                'frhare' : frhare
            }
        
        elif d_name == 'kold' :
            df = {
                'comment' : comment,
                'OFF' : hate,
                'frhare' : frhare
            }
        
        elif d_name == 'en_IMSyPP' :
            df = {
                'comment' : comment,
                'hate' : hate,
                'frhare' : frhare
            }
        elif d_name == 'en_IMSyPP_nocon' :
            df = {
                'comment' : comment,
                'hate' : hate,
                'frhare' : frhare
            }
            
        
        elif d_name == 'it_IMSyPP' :
            df = {
                'comment' : comment,
                'hate' : hate,
                'frhare' : frhare
            }
        
        # csv 만들기
        df = pd.DataFrame(df)
        stage = num_dict[num]
        df.to_csv(f'/home/nykim/HateSpeech/09_TitlePrediction/07_sota/HARE/{d_name}_{stage}.csv')
        # if d_name == 'en_IMSyPP' and num == 0 :
        #     nocon_df = pd.DataFrame(nocon_df)
        #     df.to_csv(f'/home/nykim/HateSpeech/09_TitlePrediction/07_sota/HARE/en_IMSyPP_nocon_{stage}.csv')
    
parser = argparse.ArgumentParser()
parser.add_argument('--d_name', type=str, default='en_IMSyPP_nocon')
args = parser.parse_args()

if __name__ == '__main__' :
    logging.basicConfig(filename=f"{args.d_name}_test.log", level=logging.INFO)
    logger = logging.getLogger("postprocessor.skeleton")
    logger.setLevel(logging.INFO)
    
    if args.d_name == 'kold' :
        OPENAI_API_KEY = ''
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        
        logger.info("processing kold...")
        HARE(client, 'kold', 'comment', 'OFF')

    elif args.d_name == 'beep' :
        logger.info("processing beep")
        HARE('beep', 'comments', 'hate')

    elif args.d_name == 'en_IMSyPP' :
        OPENAI_API_KEY = ''
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        
        logger.info("processing en_IMSyPP")
        HARE(client, 'en_IMSyPP', 'comment', 'hate')
        
    elif args.d_name == 'en_IMSyPP_nocon' :
        OPENAI_API_KEY = ''
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        
        logger.info("processing en_IMSyPP")
        HARE(client, 'en_IMSyPP_nocon', 'comment', 'hate')

    elif args.d_name == 'it_IMSyPP' :
        OPENAI_API_KEY = ''
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        logger.info("processing it_IMSyPP")
        HARE(client, 'it_IMSyPP', 'comment', 'hate')

