from typing import Any, Optional
import pandas as pd
import re

from rouge import Rouge

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score
import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration

class FlanT5FrHARE(pl.LightningModule) :
    def __init__(self, args, tokenizer) :
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        
        self.learning_rate = args.learning_rate
        
        self.ans_dict = {
            '(A)' : 0,
            '(B)' : 1,
            '(C)' : 2,
            '(D)' : 3
        }
        
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        self.model = T5ForConditionalGeneration.from_pretrained(args.model)
        # if args.gpus == 2 :
        #     self.model = DistributedDataParallel(self.model, device_ids=[0,1]).cuda()

        num_labels = None
        if args.d_name == 'beep' :
            num_labels = 3
        elif args.d_name == 'kold' :
            num_labels = 2
        elif 'IMSyPP' in args.d_name :
            num_labels = 4
            
        self.rouge = Rouge()
        self.rouge1 = []
        self.rouge2 = []
        self.rougel = []
        
        self.ma_acc = Accuracy(task='multiclass', num_classes=num_labels, average='macro')
        self.ma_f1 = F1Score(task='multiclass', num_classes=num_labels, average='macro')
        
        self.save_hyperparameters()
        
        self.preds = []
        self.trgts = []
        self.pred_answers = []
        self.answers = []
        
    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        # target_input_ids,
        target_attention_mask,
        labels
    ) :
        outputs = self.model(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            # decoder_input_ids=target_input_ids,
            decoder_attention_mask=target_attention_mask,
            labels=labels
        )
        
        return outputs
    
    def get_rouge(self, preds, targets) :
        scores = []
        for idx in range(len(preds)) :
            try :
                scores.append(self.rouge.get_scores(preds[idx], targets[idx])[0])
            except ValueError :
                scores.append(
                    {
                        'rouge-1' : {'f' : 0.0},
                        'rouge-2' : {'f' : 0.0},
                        'rouge-l' : {'f' : 0.0}
                    }
                )
        
        rouge1 = torch.tensor([score['rouge-1']['f'] for score in scores])
        rouge2 = torch.tensor([score['rouge-2']['f'] for score in scores])
        rougel = torch.tensor([score['rouge-l']['f'] for score in scores])
        
        return rouge1, rouge2, rougel
    
    def default_step(self, batch, batch_idx, state=None) :
        outputs = self(
            query_input_ids=batch['query_input_ids'].to(self.device),
            query_attention_mask=batch['query_attention_mask'].to(self.device),
            # target_input_ids=batch['target_input_ids'].to(self.device),
            target_attention_mask=batch['target_attention_mask'].to(self.device),
            labels=batch['target_label']
        )
        
        loss = outputs.loss
        pred_logits = outputs.logits
        
        preds = torch.argmax(pred_logits, dim=2).to(self.device) # shape : (batch, len_decoder_inputs)
        targets = batch['target_input_ids'].to(self.device)
        trgts = self.tokenizer.batch_decode(targets, skip_special_tokens=True)
        preds = self.tokenizer.batch_decode(preds)
        preds = [re.sub(r"</s>[\w\W]*", "", pred) for pred in preds]
        preds = [re.sub(r"<s>", "", pred) for pred in preds]
        
        rouge1, rouge2, rougel = self.get_rouge(preds, trgts)
        self.rouge1 += [round(r,4) for r in rouge1.tolist()]
        self.rouge2 += [round(r,4) for r in rouge2.tolist()]
        self.rougel += [round(r,4) for r in rougel.tolist()]
        
        rouge1, rouge2, rougel = torch.mean(rouge1), torch.mean(rouge2), torch.mean(rougel)
        # pred_ans = torch.tensor([self.ans_dict[re.findall(r'\(A\)|\(B\)|\(C\)|\(D\)', pred)[0]] for pred in preds])
        # trgt_ans = torch.tensor(batch['label'])
            
        # acc = self.ma_acc(pred_ans, trgt_ans)
        # f1 = self.ma_f1(pred_ans, trgt_ans)
        
        if self.args.gpus == 1 :
            self.log(f"[{state} loss]", loss, prog_bar=True)
            self.log(f"[{state} rouge1]", rouge1, prog_bar=True)
            self.log(f"[{state} rouge2]", rouge2, prog_bar=True)
            self.log(f"[{state} rougel]", rougel, prog_bar=True)
        else :
            self.log(f"[{state} loss]", loss.mean(), prog_bar=True)
            self.log(f"[{state} rouge1]", rouge1.mean(), prog_bar=True)
            self.log(f"[{state} rouge2]", rouge2.mean(), prog_bar=True)
            self.log(f"[{state} rougel]", rougel.mean(), prog_bar=True)
        # self.log(f"[{state} acc]", acc, prog_bar=True)
        # self.log(f"[{state} f1]", f1, prog_bar=True)
        
        return {
            'loss' : loss,
            'rouge1' : rouge1,
            'rouge2' : rouge2,
            'rougel' : rougel
            # 'acc' : acc,
            # 'f1' : f1,
        }
    
    def training_step(self, batch, batch_idx, state='train') :
        result = self.default_step(batch, batch_idx, state)
        self.rouge1.clear()
        self.rouge2.clear()
        self.rougel.clear()
        return result
    
    def validation_step(self, batch, batch_idx, state='valid') :
        result = self.default_step(batch, batch_idx, state)
        self.rouge1.clear()
        self.rouge2.clear()
        self.rougel.clear()
        return result
    
    def test_step(self, batch, batch_idx, state='test') :
        generated = self.model.generate(batch['query_input_ids'].to(self.device), max_length=512)
        trgts = self.tokenizer.batch_decode(batch['target_input_ids'], skip_special_tokens=True)
        preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        pred_ans = torch.tensor([self.ans_dict[re.findall(r'\(A\)|\(B\)|\(C\)|\(D\)', pred)[0]] for pred in preds]).to(self.device)
        trgt_ans = torch.tensor(batch['label']).to(self.device)
        
        self.preds += list(pred_ans.cpu())
        self.trgts += list(trgt_ans.cpu())
        self.pred_answers += list(preds)
        self.answers += list(trgts)
        
        acc = self.ma_acc(pred_ans, trgt_ans)
        f1 = self.ma_f1(pred_ans, trgt_ans)
        
        if self.args.gpus == 1 :
            self.log(f"[{state} acc]", acc, prog_bar=True)
            self.log(f"[{state} f1]", f1, prog_bar=True)
        else :
            self.log(f"[{state} acc]", acc.mean(), prog_bar=True)
            self.log(f"[{state} f1]", f1.mean(), prog_bar=True)
        return {
            'acc' : acc,
            'f1' : f1,
            'trgts' : trgts,
            'preds' : preds
        }
        
    def default_epoch_end(self, outputs, state=None) :
        if self.args.gpus == 1 :
            loss = torch.mean(torch.tensor([output['loss'] for output in outputs]))
            rouge1 = torch.mean(torch.tensor([output['rouge1'] for output in outputs]))
            rouge2 = torch.mean(torch.tensor([output['rouge2'] for output in outputs]))
            rougel = torch.mean(torch.tensor([output['rougel'] for output in outputs]))
            self.log(f'{state}_loss', loss, prog_bar=True)
            self.log(f'{state}_rouge1', rouge1, on_epoch=True, prog_bar=True)
            self.log(f'{state}_rouge2', rouge2, on_epoch=True, prog_bar=True)
            self.log(f'{state}_rougel', rougel, on_epoch=True, prog_bar=True)      
        else :
            loss = torch.mean(torch.tensor([output['loss'].mean() for output in outputs]))
            rouge1 = torch.mean(torch.tensor([output['rouge1'].mean() for output in outputs]))
            rouge2 = torch.mean(torch.tensor([output['rouge2'].mean() for output in outputs]))
            rougel = torch.mean(torch.tensor([output['rougel'].mean() for output in outputs]))
            self.log(f'{state}_loss', loss.mean(), prog_bar=True)
            self.log(f'{state}_rouge1', rouge1.mean(), on_epoch=True, prog_bar=True)
            self.log(f'{state}_rouge2', rouge2.mean(), on_epoch=True, prog_bar=True)
            self.log(f'{state}_rougel', rougel.mean(), on_epoch=True, prog_bar=True)  
        # self.log(f'{state}_acc', acc, prog_bar=True)
        # self.log(f'{state}_f1', f1, prog_bar=True)
        
    def training_epoch_end(self, outputs, state='train') :
        self.default_epoch_end(outputs, state)
    
    def validation_epoch_end(self, outputs, state='valid') :
        self.default_epoch_end(outputs, state)
        
    def test_epoch_end(self, outputs, state='test') :
        if self.args.gpus == 1 :
            acc = torch.mean(torch.tensor([output['acc'] for output in outputs]))
            f1 = torch.mean(torch.tensor([output['f1'] for output in outputs]))
            self.log(f'{state}_acc', acc, prog_bar=True)
            self.log(f'{state}_f1', f1, prog_bar=True)
        else :
            acc = torch.mean(torch.tensor([output['acc'].mean() for output in outputs]))
            f1 = torch.mean(torch.tensor([output['f1'].mean() for output in outputs]))
            self.log(f'{state}_acc', acc.mean(), prog_bar=True)
            self.log(f'{state}_f1', f1.mean(), prog_bar=True)
        df = {
            'pred' : self.preds,
            'trgt' : self.trgts,
            'trgt_r' : self.answers,
            'pred_r' : self.pred_answers
        }
        
        df = pd.DataFrame(df)
        df.to_csv(f'/home/nykim/2024_spring/02_sota/hare/02_csv/{self.args.d_name}.csv')
    
    def configure_optimizers(self) :
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150)
        return [optimizer], [lr_scheduler]
            
            