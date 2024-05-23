import re
import emoji
from soynlp.normalizer import *
import pandas as pd
import random

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class TitlePredictionDataset(Dataset) :
    def __init__(self, tokenizer, stage, args) :
        self.tokenizer = tokenizer
        self.d_name = args.d_name
        self.mode = args.mode
        self.random_ratio = args.random_ratio
        self.max_length = args.max_length
        self.d_size = args.data_size
        
        self.dataset = self.get_dataframe(stage)
        self.label = self.get_label()
        self.title = self.get_title()
        
        
    def get_dataframe(self, stage) :
        if self.d_name == 'kold' :
            dataset = pd.read_csv(f'/home/nykim/2024_spring/00_data/processed_kold/kold_{stage}.csv')
            dataset.drop_duplicates(subset=['comment'], inplace=True)
            return dataset
        
        elif self.d_name == 'beep' :
            dataset = pd.read_csv(f'/home/nykim/2024_spring/00_data/processed_beep/beep_{stage}.csv')
            dataset.drop_duplicates(subset=['comments'], inplace=True)
            return dataset
        
        elif self.d_name == 'en_IMSyPP_nocon' and stage == 'test' :
            dataset = pd.read_csv(f'/home/nykim/2024_spring/00_data/processed_en/en_IMSyPP_nocon_test.csv')
            dataset.drop_duplicates(subset=['comment'], inplace=True)
            dataset.dropna(subset=['title', 'comment', 'hate'], inplace=True)
            dataset = dataset[(dataset['title'] != None)]
            dataset = dataset[(dataset['hate']=='0. appropriate')|(dataset['hate']=='1. inappropriate')|
                              (dataset['hate']=='2. offensive')|(dataset['hate']=='3. violent')]
            
            # dataset = dataset.iloc[range(int(len(dataset) * self.d_size))]
            return dataset
        
        elif self.d_name == 'en_IMSyPP' or (self.d_name == 'en_IMSyPP_nocon' and stage != 'test') :
            dataset = pd.read_csv(f'/home/nykim/2024_spring/00_data/processed_en/en_IMSyPP_{stage}.csv')
            dataset.drop_duplicates(subset=['comment'], inplace=True)
            dataset.dropna(subset=['title', 'comment', 'hate'], inplace=True)
            dataset = dataset[(dataset['title'] != None)]
            dataset = dataset[(dataset['hate']=='0. appropriate')|(dataset['hate']=='1. inappropriate')|
                              (dataset['hate']=='2. offensive')|(dataset['hate']=='3. violent')]
            if stage == 'train' :
                dataset = dataset.iloc[range(int(len(dataset) * self.d_size))]
            return dataset
        
        elif self.d_name == 'it_IMSyPP' :
            dataset = pd.read_csv(f'/home/nykim/2024_spring/00_data/processed_it/it_IMSyPP_{stage}.csv')
            dataset.drop_duplicates(subset=['comment'], inplace=True)
            dataset.dropna(subset=['title', 'comment', 'hate'], inplace=True)
            dataset = dataset[(dataset['title'] != None)]
            dataset = dataset[(dataset['hate']=='0. appropriato')|(dataset['hate']=='1. inappropriato')|
                              (dataset['hate']=='2. offensivo')|(dataset['hate']=='3. violento')]
            if stage == 'train' :
                dataset = dataset.iloc[range(int(len(dataset) * self.d_size))]
            return dataset
        
        elif self.d_name == 'cad' :
            dataset = pd.read_csv(f'/home/nykim/2024_spring/00_data/cad_{stage}.csv')
            dataset.drop_duplicates(subset=['comment'], inplace=True)
            dataset.dropna(subset=['title', 'comment', 'label'], inplace=True)
            dataset = dataset[(dataset['title'] != None)]
            
            return dataset
        
    def get_label(self) :
        if self.d_name == 'kold' :
            return {True : 1, False : 0}
        elif self.d_name == 'beep' : 
            return {'offensive' : 2, 'hate' : 1, 'none' : 0}
        
        elif 'en_IMSyPP' in self.d_name :
            return {
                '0. appropriate' : 0,
                '1. inappropriate' : 1,
                '2. offensive' : 2, 
                '3. violent' : 3
            }
            
        elif self.d_name == 'it_IMSyPP' :
            return {
                '0. appropriato' : 0,
                '1. inappropriato' : 1,
                '2. offensivo' : 2,
                '3. violento' : 3
            }
    
    def get_title(self) :
        if self.d_name in ['kold', 'beep'] or 'IMSyPP' in self.d_name:
            return set(list(self.dataset['title']))
        
    def cleanse(self, comment, title) :
        comment = emoticon_normalize(comment, num_repeats=2)
        comment = repeat_normalize(comment, num_repeats=2)
        if self.d_name == 'beep' or self.d_name =='kold' :
            comment = emoji.demojize(comment, language='ko')
        elif self.d_name == 'en_IMSyPP' :
            comment = emoji.demojize(comment, language='en')
        elif self.d_name == 'it_IMSyPP' :
            comment = emoji.demojize(comment, language='it')
        comment = re.sub(r"[:_]", ' ', comment)
        comment = re.sub(r"\s+", " ", comment)
        comment = re.sub('-*-*-*-', '', comment)
        comment = re.sub(r'@\w+', '', comment)
        
        title = re.sub(r'\[[\w\W]+\]', '', title)
        title = re.sub(r'\([\w\W]+\)', '', title)
        return comment, title
    
    def __getitem__(self, idx) :
        # hate speech detection
        if self.d_name == 'kold' :
            comment, title = self.dataset['comment'].iloc[idx], self.dataset['title'].iloc[idx]
        elif self.d_name == 'beep' :
            comment, title = self.dataset['comments'].iloc[idx], self.dataset['title'].iloc[idx]
        elif 'IMSyPP' in self.d_name :
            comment, title = self.dataset['comment'].iloc[idx], self.dataset['title'].iloc[idx]
        comment, title = self.cleanse(comment, title)
        
        if self.d_name == 'beep' :
            hs_label = self.label[self.dataset['hate'].iloc[idx]]
        elif self.d_name == 'kold' :
            hs_label = self.label[self.dataset['OFF'].iloc[idx]]
        elif 'IMSyPP' in self.d_name :
            hs_label = self.label[self.dataset['hate'].iloc[idx]]
        
            
        
        
        # title prediction
        r = random.random()
        random_title = title if r <= self.random_ratio else random.sample(self.title - {title}, 1)[0]
        tp_label = 1 if r <= self.random_ratio else 0
        
        # title + comment
        if self.mode == 0 :
            sentence = title + self.tokenizer.sep_token + comment
            t_sentence = random_title + self.tokenizer.sep_token + comment
        # comment only    
        elif self.mode == 1 :
            sentence = comment
            t_sentence = comment
        
        if self.mode <= 1 :
            hs_tokked = self.tokenizer(
                sentence,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )
            
            tp_tokked = self.tokenizer(
                t_sentence,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )
            
            return {
                'hs_input_ids' : hs_tokked.input_ids[0],
                'hs_attention_mask' : hs_tokked.attention_mask[0],
                'hs_label' : hs_label,
                'tp_input_ids' : tp_tokked.input_ids[0],
                'tp_attention_mask' : tp_tokked.attention_mask[0],
                'tp_label' : tp_label
            }
            
        elif self.mode == 2 :
            hsc_tokked = self.tokenizer(
                comment,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )
            
            hst_tokked = self.tokenizer(
                title,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )
            
            t_sentence = random_title + self.tokenizer.sep_token + comment
            tp_tokked = self.tokenizer(
                t_sentence,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )
            
            return {
                'hsc_input_ids' : hsc_tokked.input_ids[0],
                'hsc_attention_mask' : hsc_tokked.attention_mask[0],
                'hst_input_ids' : hst_tokked.input_ids[0],
                'hst_attention_mask' : hst_tokked.attention_mask[0],
                'hs_label' : hs_label,
                'tp_input_ids' : tp_tokked.input_ids[0],
                'tp_attention_mask' : tp_tokked.attention_mask[0],
                'tp_label' : tp_label
            }
            
            
    def __len__(self) :
        return len(self.dataset)
    
class TitlePredictionDataModule(pl.LightningDataModule) :
    def __init__(self, args, tokenizer) :
        super().__init__()
        
        self.args = args
        
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
        self.tokenizer = tokenizer
        
        self.setup()
        
    def setup(self, stage=None) :
        self.set_train = TitlePredictionDataset(self.tokenizer, 'train', self.args)
        self.set_valid = TitlePredictionDataset(self.tokenizer, 'valid', self.args)
        self.set_test = TitlePredictionDataset(self.tokenizer, 'test', self.args)
        
    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return  train

    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return valid
    
    def test_dataloader(self) :
        test = DataLoader(self.set_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return test
        