import re
import emoji
from soynlp.normalizer import *
import pandas as pd
import random
import copy

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class BEEPDataset(Dataset) :
    def __init__(self, tokenizer, stage, args) :
        self.tokenizer = tokenizer
        
        self.max_length = args.max_length
        
        self.dataset = pd.read_csv(f'/home/nykim/2024_spring/00_data/processed_beep/beep_{stage}.csv')
        self.dataset.drop_duplicates(subset=['comments'], inplace=True)
        
        self.label = {'offensive' : 2, 'hate' : 1, 'none' : 0}
    
    def cleanse(self, comment, title) :
        comment = emoticon_normalize(comment, num_repeats=2)
        comment = repeat_normalize(comment, num_repeats=2)
        comment = emoji.demojize(comment, language='ko')
        comment = re.sub(r"[:_]", ' ', comment)
        comment = re.sub(r"\s+", " ", comment)
        
        title = re.sub(r'\[[\w\W]+\]', '', title)
        title = re.sub(r'\([\w\W]+\)', '', title)
        return comment, title
    
    def __getitem__(self, idx):
        comment, title = self.cleanse(self.dataset['comments'].iloc[idx], self.dataset['title'].iloc[idx])
        
        label = self.label[self.dataset['hate'].iloc[idx]]
        
        comment_tokked = self.tokenizer(
            comment, 
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        title_tokked = self.tokenizer(
            title,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        sentence = title + '[SEP]' + comment
        tc_tokked = self.tokenizer(
            sentence,
            return_tensors='pt',
            max_length = self.max_length,
            padding='max_length',
            truncation=True
        )
        
        return {
            'c_input_ids' : comment_tokked.input_ids[0],
            'c_attention_mask' : comment_tokked.attention_mask[0],
            't_input_ids' : title_tokked.input_ids[0],
            't_attention_mask' : title_tokked.attention_mask[0],
            'tc_input_ids' : tc_tokked.input_ids[0],
            'tc_attention_mask' : tc_tokked.attention_mask[0],
            'label' : label
        }
        
        
    def __len__(self) :
        return len(self.dataset)

class KOLDDataset(Dataset) :
    def __init__(self, tokenizer, stage, args) :
        self.tokenizer = tokenizer
        
        self.max_length = args.max_length
        
        self.dataset = pd.read_csv(f'/home/nykim/2024_spring/00_data/processed_kold/kold_{stage}.csv')
        self.dataset.drop_duplicates(subset=['comment'], inplace=True)
        
        self.label = {True : 1, False : 0}
    
    def cleanse(self, comment, title) :
        comment = emoticon_normalize(comment, num_repeats=2)
        comment = repeat_normalize(comment, num_repeats=2)
        comment = emoji.demojize(comment, language='ko')
        comment = re.sub(r"[:_]", ' ', comment)
        comment = re.sub(r"\s+", " ", comment)
        
        title = re.sub(r'\[[\w\W]+\]', '', title)
        title = re.sub(r'\([\w\W]+\)', '', title)
        return comment, title
    
    def __getitem__(self, idx):
        comment, title = self.cleanse(self.dataset['comment'].iloc[idx], self.dataset['title'].iloc[idx])
        
        label = self.label[self.dataset['OFF'].iloc[idx]]
        
        comment_tokked = self.tokenizer(
            comment, 
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        title_tokked = self.tokenizer(
            title,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        sentence = title + '[SEP]' + comment
        tc_tokked = self.tokenizer(
            sentence,
            return_tensors='pt',
            max_length = self.max_length,
            padding='max_length',
            truncation=True
        )
        
        return {
            'c_input_ids' : comment_tokked.input_ids[0],
            'c_attention_mask' : comment_tokked.attention_mask[0],
            't_input_ids' : title_tokked.input_ids[0],
            't_attention_mask' : title_tokked.attention_mask[0],
            'tc_input_ids' : tc_tokked.input_ids[0],
            'tc_attention_mask' : tc_tokked.attention_mask[0],
            'label' : label
        }
        
    def __len__(self) :
        return len(self.dataset)

class IMSYPP_ENDataset(Dataset) :
    def __init__(self, tokenizer, stage, args) :
        self.tokenizer = tokenizer
        
        self.max_length = args.max_length
        
        if args.d_name == 'en_IMSyPP_nocon' and stage == 'test' :
            self.dataset = pd.read_csv(f'/home/nykim/HateSpeech/09_TitlePrediction/00_data/en_IMSyPP_nocon_{stage}.csv')
        elif 'en_IMSyPP' in args.d_name :
            self.dataset = pd.read_csv(f'/home/nykim/2024_spring/00_data/processed_en/en_IMSyPP_{stage}.csv')
        
        
        self.dataset.drop_duplicates(subset=['comment'], inplace=True)
        self.dataset.dropna(subset=['title', 'comment', 'hate'], inplace=True)
        self.dataset = self.dataset[(self.dataset['title'] != None)]
        self.dataset = self.dataset[(self.dataset['hate']=='0. appropriate')|(self.dataset['hate']=='1. inappropriate')|
                    (self.dataset['hate']=='2. offensive')|(self.dataset['hate']=='3. violent')]
        
        self.label = {
                '0. appropriate' : 0,
                '1. inappropriate' : 1,
                '2. offensive' : 2, 
                '3. violent' : 3
            }
    
    def cleanse(self, comment, title) :
        comment = emoticon_normalize(comment, num_repeats=2)
        comment = repeat_normalize(comment, num_repeats=2)
        # comment = emoji.demojize(comment, language='en')
        comment = re.sub(r"[:_]", ' ', comment)
        comment = re.sub(r"\s+", " ", comment)
        
        title = re.sub(r'\[[\w\W]+\]', '', title)
        title = re.sub(r'\([\w\W]+\)', '', title)
        return comment, title
    
    def __getitem__(self, idx):
        comment, title = self.cleanse(self.dataset['comment'].iloc[idx], self.dataset['title'].iloc[idx])
        
        label = self.label[self.dataset['hate'].iloc[idx]]
        
        comment_tokked = self.tokenizer(
            comment, 
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        title_tokked = self.tokenizer(
            title,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        sentence = title + '[SEP]' + comment
        tc_tokked = self.tokenizer(
            sentence,
            return_tensors='pt',
            max_length = self.max_length,
            padding='max_length',
            truncation=True
        )
        
        return {
            'c_input_ids' : comment_tokked.input_ids[0],
            'c_attention_mask' : comment_tokked.attention_mask[0],
            't_input_ids' : title_tokked.input_ids[0],
            't_attention_mask' : title_tokked.attention_mask[0],
            'tc_input_ids' : tc_tokked.input_ids[0],
            'tc_attention_mask' : tc_tokked.attention_mask[0],
            'label' : label
        }
        
    def __len__(self) :
        return len(self.dataset) 

    
class IMSYPP_ITDataset(Dataset) :
    def __init__(self, tokenizer, stage, args) :
        self.tokenizer = tokenizer
        
        self.max_length = args.max_length
        
        self.dataset = pd.read_csv(f'/home/nykim/2024_spring/00_data/processed_it/it_IMSyPP_{stage}.csv')
        
        self.dataset.drop_duplicates(subset=['comment'], inplace=True)
        self.dataset.dropna(subset=['title', 'comment', 'hate'], inplace=True)
        self.dataset = self.dataset[(self.dataset['title'] != None)]
        self.dataset = self.dataset[(self.dataset['hate']=='0. appropriato')|(self.dataset['hate']=='1. inappropriato')|
                    (self.dataset['hate']=='2. offensivo')|(self.dataset['hate']=='3. violento')]
        
        self.label = {
                '0. appropriato' : 0,
                '1. inappropriato' : 1,
                '2. offensivo' : 2, 
                '3. violento' : 3
            }
    
    def cleanse(self, comment, title) :
        comment = emoticon_normalize(comment, num_repeats=2)
        comment = repeat_normalize(comment, num_repeats=2)
        # comment = emoji.demojize(comment, language='en')
        comment = re.sub(r"[:_]", ' ', comment)
        comment = re.sub(r"\s+", " ", comment)
        
        title = re.sub(r'\[[\w\W]+\]', '', title)
        title = re.sub(r'\([\w\W]+\)', '', title)
        return comment, title
    
    def __getitem__(self, idx):
        comment, title = self.cleanse(self.dataset['comment'].iloc[idx], self.dataset['title'].iloc[idx])
        
        label = self.label[self.dataset['hate'].iloc[idx]]
        
        comment_tokked = self.tokenizer(
            comment, 
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        title_tokked = self.tokenizer(
            title,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        sentence = title + '[SEP]' + comment
        tc_tokked = self.tokenizer(
            sentence,
            return_tensors='pt',
            max_length = self.max_length,
            padding='max_length',
            truncation=True
        )
        
        return {
            'c_input_ids' : comment_tokked.input_ids[0],
            'c_attention_mask' : comment_tokked.attention_mask[0],
            't_input_ids' : title_tokked.input_ids[0],
            't_attention_mask' : title_tokked.attention_mask[0],
            'tc_input_ids' : tc_tokked.input_ids[0],
            'tc_attention_mask' : tc_tokked.attention_mask[0],
            'label' : label
        }
        
    def __len__(self) :
        return len(self.dataset)
    
class NayakDataModule(pl.LightningDataModule) :
    def __init__(self, args, tokenizer) :
        super().__init__()
        
        self.args = args
        
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
        self.tokenizer = tokenizer
        
        self.setup()
        
    def setup(self, stage=None) :
        if self.args.d_name == 'beep' :
            self.set_train = BEEPDataset(self.tokenizer, 'train', self.args)
            self.set_valid = BEEPDataset(self.tokenizer, 'valid', self.args)
            self.set_test = BEEPDataset(self.tokenizer, 'test', self.args)
            
        if self.args.d_name == 'kold' :
            self.set_train = KOLDDataset(self.tokenizer, 'train', self.args)
            self.set_valid = KOLDDataset(self.tokenizer, 'valid', self.args)
            self.set_test = KOLDDataset(self.tokenizer, 'test', self.args)
            
        if  'en_IMSyPP' in self.args.d_name :
            self.set_train = IMSYPP_ENDataset(self.tokenizer, 'train', self.args)
            self.set_valid = IMSYPP_ENDataset(self.tokenizer, 'valid', self.args)
            self.set_test = IMSYPP_ENDataset(self.tokenizer, 'test', self.args)
            
        if  self.args.d_name == 'it_IMSyPP' :
            self.set_train = IMSYPP_ITDataset(self.tokenizer, 'train', self.args)
            self.set_valid = IMSYPP_ITDataset(self.tokenizer, 'valid', self.args)
            self.set_test = IMSYPP_ITDataset(self.tokenizer, 'test', self.args)
            
    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train
    
    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return valid
    
    def test_dataloader(self) :
        test = DataLoader(self.set_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return test