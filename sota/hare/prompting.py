import os
import pandas as pd
import openai
from openai import OpenAI
import logging
import re

logging.basicConfig(filename="BEEP.log", level=logging.INFO)
logger = logging.getLogger("postprocessor.skeleton")
logger.setLevel(logging.INFO)

OPENAI_API_KEY = ''
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=OPENAI_API_KEY
)

QUESTION_DICT = {
    'beep' : "Determine whether the following Korean post is offensive or hate or none. You must choose one of the options with number.",
    'kold' : "Determine whether the following Korean post is offensive or not : True or False. You must choose one of the options with number.",
    'en_IMSyPP' : "Determine whether the following post is appropriate or inappropriate or offensive or violent. You must choose one of the options with number.",
    'en_IMSyPP_nocon' : "Determine whether the following post is appropriate or inappropriate or offensive or violent. You must choose one of the options with number.",
    'it_IMSyPP' : "Determine whether the following Italian post is appropriato or inappropriato or offensivo or violento. You must choose one of the options with number."
}

OPTION_DICT = {
    'beep' : '(0) None\n(1) Hate\n(2) Offensive',
    'kold' : '(0) False\n(1) True',
    'en_IMSyPP' : '(0) appropriate\n(1) inappropriate\n(2) offensive\n(3) violent',
    'en_IMSyPP_nocon' : '(0) appropriate\n(1) inappropriate\n(2) offensive\n(3) violent',
    'it_IMSyPP' : '(0) appropriato\n(1) inappropriato\n(2) offensivo\n(3) violento'
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

def get_dataset(d_name) :
    if d_name == 'beep' :
        dataset = pd.read_csv('/home/nykim/HateSpeech/09_TitlePrediction/00_data/beep_test.csv')
        dataset.drop_duplicates(subset=['comments'], inplace=True)
        dataset.dropna(subset=['comments', 'hate', 'title'], inplace=True)
        return dataset
    
    elif d_name == 'kold' :
        dataset = pd.read_csv('/home/nykim/HateSpeech/09_TitlePrediction/00_data/kold_test.csv')
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
        dataset = pd.read_csv('/home/nykim/HateSpeech/09_TitlePrediction/00_data/en_IMSyPP_test.csv')
        dataset.drop_duplicates(subset=['comment'], inplace=True)
        dataset = dataset[(dataset['hate']=='0. appropriate')|(dataset['hate']=='1. inappropriate')|
                              (dataset['hate']=='2. offensive')|(dataset['hate']=='3. violent')]
        dataset.dropna(subset=['comment', 'hate', 'title'], inplace=True)
        return dataset
    
    elif d_name == 'it_IMSyPP' :
        dataset = pd.read_csv('/home/nykim/HateSpeech/09_TitlePrediction/00_data/it_IMSyPP_test.csv')
        dataset.drop_duplicates(subset=['comment'], inplace=True)
        dataset = dataset[(dataset['hate']=='0. appropriato')|(dataset['hate']=='1. inappropriato')|
                              (dataset['hate']=='2. offensivo')|(dataset['hate']=='3. violento')]
        dataset.dropna(subset=['comment', 'hate', 'title'], inplace=True)
        return dataset
    
# dataset
beep = get_dataset('beep')
kold = get_dataset('kold')
en_imsypp = get_dataset('en_IMSyPP')
en_imsypp_nocon = get_dataset('en_IMSyPP_nocon')
it_imsypp = get_dataset('it_IMSyPP')
    
def get_prompt(d_name, comment) :
    question = QUESTION_DICT[d_name]
    options = OPTION_DICT[d_name]
    prompt = f"{question}\nPost: {comment}\nOptions:\n{options}\nAnswer: Let's explain step by step"
    # prompt = f"{question}\nPost: {comment}\nOptions:\n{options}\nAnswer: "
    return prompt

def HARE(dataframe, d_name, comment_cname, label_cname) :
    labels_LLM = OPTION_DICT[d_name].split('\n')
    
    gold = []
    pred = []
    answers = []
    prompts = []
    
    for idx in range(len(dataframe)) :
        comment = dataframe[comment_cname].iloc[idx]
        label = LABEL_DICT[d_name][dataframe[label_cname].iloc[idx]] # int label
        prompt = get_prompt(d_name, comment)
        stream = client.chat.completions.create(
            # model="gpt-3.5-turbo",
            model="gpt-3.5-turbo-0613",
            messages=[{"role": "user", "content": prompt}],
            # stream=True,
        )
        
        ans = stream.choices[0].message.content
        answers.append(ans)
        prompts.append(prompt)
        
        gold.append(label)
        label_LLM = re.findall(r'0|1|2|3|4',ans)
        try :
            label_LLM = label_LLM[0]
            pred.append(int(label_LLM))
        except Exception as e :
            pred.append(len(labels_LLM))
        # for num, label_LLM in enumerate(labels_LLM) :
        #     if label_LLM in ans :
        #         pred.append(num)
        #         break
        # if (len(answers) == len(prompts) == len(gold)) and len(gold) > len(pred) :
        #     pred.append(len(labels_LLM))
            
        logger.info(f'len_gold: {len(gold)} | len_pred: {len(pred)} | len_answers: {len(answers)} | len_prompts: {len(prompts)}')
        assert len(answers) == len(prompts) == len(gold) == len(pred)
    
    df = {
        'trgt' : gold,
        'pred' : pred,
        'answer' : answers,
        'prompt' : prompts
    }
    
    # csv 만들기
    df = pd.DataFrame(df)
    df.to_csv(f'/home/nykim/HateSpeech/09_TitlePrediction/07_sota/HARE/{d_name}.csv')
    

logger.info("processing kold...")
HARE(kold, 'kold', 'comment', 'OFF')

# logger.info("processing beep")
# HARE(beep, 'beep', 'comments', 'hate')

# logger.info("processing en_IMSyPP")
# HARE(en_imsypp, 'en_IMSyPP', 'comment', 'hate')

# logger.info("processing en_IMSyPP_nocon")
# HARE(en_imsypp_nocon, 'en_IMSyPP_nocon', 'comment', 'hate')

# logger.info("processing it_IMSyPP")
# HARE(it_imsypp, 'it_IMSyPP', 'comment', 'hate')

