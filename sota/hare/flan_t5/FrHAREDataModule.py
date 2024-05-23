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
        
        self.dataset = pd.read_csv(f'/home/nykim/2024_spring/02_sota/hare/00_data/beep_{stage}.csv')
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
        comment = self.dataset['comments'].iloc[idx]
        rationale = self.dataset['frhare'].iloc[idx]
        label = self.dataset['hate'].iloc[idx]
        
        query  = f'다음 게시물이 모욕적인지 여부를 결정하십시오. 당신은 다음 옵션 중 하나를 선택해야 합니다.\n게시물: {comment}\n옵션:\n(A) none\n(B) hate\n(C) offensive\n답변:\n'
        if label == 'none' :
            target = '답은 (A) none. '
        elif label == 'hate' :
            target = '답은 (B) hate. '
        elif label == 'offensive' :
            target = '답은 (C) offensive. '
        
        target = target + rationale
        
        query_encoding_dict = self.tokenizer(query, return_tensors='pt', padding='max_length',  max_length=512, truncation=True)
        query_input_ids = query_encoding_dict['input_ids'][0]
        query_masks = query_encoding_dict['attention_mask'][0]
        
        target_encoding_dict = self.tokenizer(target, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        target_input_ids = target_encoding_dict['input_ids'][0]
        target_masks = target_encoding_dict['attention_mask'][0]
        
        target_label = copy.deepcopy(target_input_ids)
        target_label = torch.tensor([-100 if tl == 0 else tl for tl in target_label])
        return {
            'query_input_ids' : query_input_ids,
            'query_attention_mask' : query_masks,
            'target_input_ids' : target_input_ids,
            'target_attention_mask' : target_masks,
            'target_label' : target_label,
            'label': self.label[label]
        }
        
    def __len__(self) :
        return len(self.dataset)

class KOLDDataset(Dataset) :
    def __init__(self, tokenizer, stage, args) :
        self.tokenizer = tokenizer
        
        self.dataset = pd.read_csv(f'/home/nykim/2024_spring/02_sota/hare/00_data/kold_{stage}.csv')
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
        comment = self.dataset['comment'].iloc[idx]
        rationale = self.dataset['frhare'].iloc[idx]
        label = self.dataset['OFF'].iloc[idx]
        
        query  = f'다음 게시물이 모욕적인지 여부를 결정하십시오. 당신은 다음 옵션 중 하나를 선택해야 합니다.\n게시물: {comment}\n옵션:\n(A) False\n(B) True\n답변:\n'
        if label == False :
            target = '답은 (A) False. '
        elif label == True :
            target = '답은 (B) True. '
        
        target = target + rationale
        
        query_encoding_dict = self.tokenizer(query, return_tensors='pt', padding='max_length',  max_length=512, truncation=True)
        query_input_ids = query_encoding_dict['input_ids'][0]
        query_masks = query_encoding_dict['attention_mask'][0]
        
        target_encoding_dict = self.tokenizer(target, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        target_input_ids = target_encoding_dict['input_ids'][0]
        target_masks = target_encoding_dict['attention_mask'][0]
        
        target_label = copy.deepcopy(target_input_ids)
        target_label = torch.tensor([-100 if tl == 0 else tl for tl in target_label])
        return {
            'query_input_ids' : query_input_ids,
            'query_attention_mask' : query_masks,
            'target_input_ids' : target_input_ids,
            'target_attention_mask' : target_masks,
            'target_label' : target_label,
            'label': self.label[label]
        }
        
    def __len__(self) :
        return len(self.dataset)
        
class IMSYPP_ENDataset(Dataset) :
    def __init__(self, tokenizer, stage, args) :
        self.tokenizer = tokenizer
        
        if args.d_name == 'en_IMSyPP_nocon' :
            self.dataset = pd.read_csv(f'/home/nykim/2024_spring/02_sota/hare/00_data/en_IMSyPP_nocon_test.csv')
        else :
            self.dataset = pd.read_csv(f'/home/nykim/2024_spring/02_sota/hare/00_data/en_{stage}.csv')
        self.dataset.drop_duplicates(subset=['comment'], inplace=True)
        self.dataset.dropna(subset=['comment', 'hate'], inplace=True)
        # self.dataset = self.dataset[(self.dataset['title'] != None)]
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
        comment = self.dataset['comment'].iloc[idx]
        rationale = self.dataset['frhare'].iloc[idx]
        label = self.dataset['hate'].iloc[idx]
        
        query  = f'Determine whether the following post is offensive or not. You must choose one of the options.\nPost: {comment}\nOptions: \n(A) appropriate\n(B) inappropriate\n(C) offensive\n(D) violent\nAnswer:\n'
        if label == '0. appropriate' :
            target = 'The answer is (A) appropriate. '
        elif label == '1. inappropriate' :
            target = 'The answer is (B) inappropriate. '
        elif label == '2. offensive' :
            target = 'The answer is (C) offensive. '
        elif label == '3. violent' :
            target = 'The answer is (D) violent. '
        
        target = target + str(rationale)
        
        query_encoding_dict = self.tokenizer(query, return_tensors='pt', padding='max_length',  max_length=512, truncation=True)
        query_input_ids = query_encoding_dict['input_ids'][0]
        query_masks = query_encoding_dict['attention_mask'][0]
        
        target_encoding_dict = self.tokenizer(target, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        target_input_ids = target_encoding_dict['input_ids'][0]
        target_masks = target_encoding_dict['attention_mask'][0]
        
        target_label = copy.deepcopy(target_input_ids)
        target_label = torch.tensor([-100 if tl == 0 else tl for tl in target_label])
        return {
            'query_input_ids' : query_input_ids,
            'query_attention_mask' : query_masks,
            'target_input_ids' : target_input_ids,
            'target_attention_mask' : target_masks,
            'target_label' : target_label,
            'label': self.label[label]
        }
        
    def __len__(self) :
        return len(self.dataset)
    
class IMSYPP_ITDataset(Dataset) :
    def __init__(self, tokenizer, stage, args) :
        self.tokenizer = tokenizer
        
        
        self.dataset = pd.read_csv(f'/home/nykim/2024_spring/02_sota/hare/00_data/it_{stage}.csv')
        
        self.dataset.drop_duplicates(subset=['comment'], inplace=True)
        self.dataset.dropna(subset=['comment', 'hate'], inplace=True)
        # self.dataset = self.dataset[(self.dataset['title'] != None)]
        self.dataset = self.dataset[(self.dataset['hate']=='0. appropriato')|(self.dataset['hate']=='1. inappropriato')|
                    (self.dataset['hate']=='2. offensivo')|(self.dataset['hate']=='3. violento')]
        
        self.label = {
                '0. appropriato' : 0,
                '1. inappropriato' : 1,
                '2. offensivo' : 2, 
                '3. violento' : 3
            }
    
    def cleanse(self, comment) :
        comment = emoticon_normalize(comment, num_repeats=2)
        comment = repeat_normalize(comment, num_repeats=2)
        # comment = emoji.demojize(comment, language='en')
        comment = re.sub(r"[:_]", ' ', comment)
        comment = re.sub(r"\s+", " ", comment)
        
        # title = re.sub(r'\[[\w\W]+\]', '', title)
        # title = re.sub(r'\([\w\W]+\)', '', title)
        return comment
    
    def __getitem__(self, idx):
        comment = self.dataset['comment'].iloc[idx]
        rationale = self.dataset['frhare'].iloc[idx]
        label = self.dataset['hate'].iloc[idx]
        
        query  = f'Stabilisci se il seguente post è offensivo o meno. Devi scegliere una delle opzioni.\nPost: {comment}\nOpzioni:\n(A) appropriato\n(B) inappropriato\n(C) offensivo\n(D) violento\nRisposta:'
        if label == '0. appropriato' :
            target = 'La risposta è (A) appropriato. '
        elif label == '1. inappropriato' :
            target = 'La risposta è (B) inappropriato. '
        elif label == '2. offensivo' :
            target = 'La risposta è (C) offensivo. '
        elif label == '3. violento' :
            target = 'La risposta è (D) violento. '
        
        target = target + rationale
        
        query_encoding_dict = self.tokenizer(query, return_tensors='pt', padding='max_length',  max_length=512, truncation=True)
        query_input_ids = query_encoding_dict['input_ids'][0]
        query_masks = query_encoding_dict['attention_mask'][0]
        
        target_encoding_dict = self.tokenizer(target, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        target_input_ids = target_encoding_dict['input_ids'][0]
        target_masks = target_encoding_dict['attention_mask'][0]
        
        target_label = copy.deepcopy(target_input_ids)
        target_label = torch.tensor([-100 if tl == 0 else tl for tl in target_label])
        return {
            'query_input_ids' : query_input_ids,
            'query_attention_mask' : query_masks,
            'target_input_ids' : target_input_ids,
            'target_attention_mask' : target_masks,
            'target_label' : target_label,
            'label': self.label[label]
        }
        
    def __len__(self) :
        return len(self.dataset)
    
class FrHAREDataModule(pl.LightningDataModule) :
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