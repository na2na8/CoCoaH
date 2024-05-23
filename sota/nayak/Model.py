import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score
import pytorch_lightning as pl
from transformers import BertModel, BertConfig, BertPreTrainedModel
from torchensemble import FusionClassifier

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

class SingleEncoder(BertPreTrainedModel) :
    def __init__(self, args, config) :
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.bert = BertModel.from_pretrained(args.model, config=config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(
        self,
        t_input_ids,
        t_attention_mask,
        c_input_ids,
        c_attention_mask,
        tc_input_ids,
        tc_attention_mask,
        label
    ) :
        return_dict=None
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # concat
        tc_outputs = self.bert(
            input_ids=tc_input_ids,
            attention_mask=tc_attention_mask,
            return_dict=return_dict
        )
        tc_pooled_output = tc_outputs[1]
        tc_pooled_output = self.dropout(tc_pooled_output)
        tc_logits = self.classifier(tc_pooled_output)
        
        if label is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (label.dtype == torch.long or label.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(tc_logits.squeeze(), label.squeeze())
                else:
                    loss = loss_fct(tc_logits, label)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(tc_logits.view(-1, self.num_labels), label.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(tc_logits, label)
        
        return {
            'loss' : loss,
            'logits' : tc_logits,
        }
        
class DualEncoder(BertPreTrainedModel) :
    def __init__(self, args, config) :
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.bert = BertModel.from_pretrained(args.model, config=config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(
        self,
        t_input_ids,
        t_attention_mask,
        c_input_ids,
        c_attention_mask,
        tc_input_ids,
        tc_attention_mask,
        label
    ) :
        return_dict=None
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # title embedding
        t_outputs = self.bert(
            input_ids=t_input_ids,
            attention_mask=t_attention_mask,
            return_dict=return_dict
        )
        t_pooled_output = t_outputs[1]
        # comment embedding
        c_outputs = self.bert(
            input_ids=c_input_ids,
            attention_mask=c_attention_mask,
            return_dict=return_dict
        )
        c_pooled_output = c_outputs[1]
        
        avg_pooled_output = (t_pooled_output + c_pooled_output) / 2
        avg_pooled_output = self.dropout(avg_pooled_output)
        logits = self.classifier(avg_pooled_output)
        
        if label is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (label.dtype == torch.long or label.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), label.squeeze())
                else:
                    loss = loss_fct(logits, label)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, label)
        
        return {
            'loss' : loss,
            'logits' : logits,
        }
        
class NayakModel(pl.LightningModule) :
    def __init__(self, args, tokenizer) :
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.learning_rate = args.learning_rate
        
        config = BertConfig.from_pretrained(args.model)
        if args.d_name == 'kold' :
            config.num_labels = 2
        elif args.d_name == 'beep' :
            config.num_labels = 3
        elif 'IMSyPP' in args.d_name :
            config.num_labels = 4
        self.ma_acc = Accuracy(task='multiclass', num_classes=config.num_labels, average='macro')
        self.ma_f1 = F1Score(task='multiclass', num_classes=config.num_labels, average='macro')
        
        if args.mode == 'dual' :
            self.model = DualEncoder(args, config)
        else :
            self.model = SingleEncoder(args, config)
        
        self.save_hyperparameters()
        
        self.softmax = nn.Softmax(dim=1)
        
        self.preds = []
        self.trgts = []
        self.val_preds = []
        self.val_trgts = []
        self.titles = []
        self.comments = []
        
        self.timings = torch.tensor([0.0])
        
    def forward(
        self,
        c_input_ids,
        c_attention_mask,
        t_input_ids,
        t_attention_mask,
        tc_input_ids,
        tc_attention_mask,
        label
    ) :
        outputs = self.model(
                c_input_ids=c_input_ids,
                c_attention_mask=c_attention_mask,
                t_input_ids=t_input_ids,
                t_attention_mask=t_attention_mask,
                tc_input_ids=tc_input_ids,
                tc_attention_mask=tc_attention_mask,
                label=label
            )
        return outputs
    
    def default_step(self, batch, batch_idx, state=None) :
        starter.record()
        outputs = self(
                c_input_ids=batch['c_input_ids'].to(self.device),
                c_attention_mask=batch['c_attention_mask'].to(self.device),
                t_input_ids=batch['t_input_ids'].to(self.device),
                t_attention_mask=batch['t_attention_mask'].to(self.device),
                tc_input_ids=batch['tc_input_ids'].to(self.device),
                tc_attention_mask=batch['tc_attention_mask'].to(self.device),
                label=batch['label'].to(self.device)
            )
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        self.timings = torch.concat((self.timings, torch.tensor([round(curr_time,2)/1000])))
        
        logits = outputs['logits']
        loss = outputs['loss']
        
        preds = logits.argmax(dim=1).to(self.device)
        trgts = batch['label'].to(self.device)
        
        if state == 'train' : 
            self.preds += preds.tolist()
            self.trgts += trgts.tolist()
        if state == 'test' or state == 'valid' :
            self.val_preds += preds.tolist()
            self.val_trgts += trgts.tolist()
            
            if state == 'test' : 
                t_decoded = self.tokenizer.batch_decode(batch['t_input_ids'], skip_special_tokens=True)
                c_decoded = self.tokenizer.batch_decode(batch['c_input_ids'], skip_special_tokens=True)
                
                self.titles += t_decoded
                self.comments += c_decoded
        
        ma_acc = self.ma_acc(preds, trgts)
        ma_f1 = self.ma_f1(preds,trgts)
        
        self.log(f"[{state} loss]", loss, prog_bar=True)
        self.log(f"[{state} ma acc]", ma_acc, prog_bar=True)
        self.log(f"[{state} ma f1]", ma_f1, prog_bar=True)
        
        return {
            'loss' : loss
        }
        
    def default_epoch_end(self, outputs, state=None) :
        loss = torch.mean(torch.tensor([output['loss'] for output in outputs]))
        # acc = torch.mean(torch.tensor([output['acc'] for output in outputs]))
        # f1 = torch.mean(torch.tensor([output['f1'] for output in outputs]))
        if state == 'train' :
            acc = self.ma_acc(torch.tensor(self.preds).to(self.device), torch.tensor(self.trgts).to(self.device))
            f1 = self.ma_f1(torch.tensor(self.preds).to(self.device), torch.tensor(self.trgts).to(self.device))
        elif state == 'valid' :
            acc = self.ma_acc(torch.tensor(self.val_preds).to(self.device), torch.tensor(self.val_trgts).to(self.device))
            f1 = self.ma_f1(torch.tensor(self.val_preds).to(self.device), torch.tensor(self.val_trgts).to(self.device))
        mean = self.timings.mean()
        summ = self.timings.sum()
        
        self.log('total_time', summ, on_epoch=True, prog_bar=True)
        self.log('mean_time', mean, on_epoch=True, prog_bar=True)
        self.log('total_example', len(self.timings), on_epoch=True, prog_bar=True)
        
        self.log(f'{state}_loss', loss, on_epoch=True, prog_bar=True)
        self.log(f'{state}_acc', acc, on_epoch=True, prog_bar=True)
        self.log(f'{state}_f1', f1, on_epoch=True, prog_bar=True)
        
    def training_step(self, batch, batch_idx, state='train') :
        result = self.default_step(batch, batch_idx, state)
        return result
    
    def validation_step(self, batch, batch_idx, state='valid') :
        result = self.default_step(batch, batch_idx, state)
        return result
    
    def test_step(self, batch, batch_idx, state='test') :
        starter.record()
        outputs = self(
                c_input_ids=batch['c_input_ids'].to(self.device),
                c_attention_mask=batch['c_attention_mask'].to(self.device),
                t_input_ids=batch['t_input_ids'].to(self.device),
                t_attention_mask=batch['t_attention_mask'].to(self.device),
                tc_input_ids=batch['tc_input_ids'].to(self.device),
                tc_attention_mask=batch['tc_attention_mask'].to(self.device),
                label=batch['label'].to(self.device)
            )
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        self.timings = torch.concat((self.timings, torch.tensor([round(curr_time,2)/1000])))
            
        logits = outputs['logits']
        # loss = outputs['loss']
        
        preds = self.softmax(logits).tolist()
        trgts = batch['label'].tolist()
        
        batch_f1 = self.ma_f1(torch.tensor(preds).to(self.device), torch.tensor(trgts).to(self.device))
        batch_acc = self.ma_acc(torch.tensor(preds).to(self.device), torch.tensor(trgts).to(self.device))
        
        self.val_preds += preds
        self.val_trgts += trgts
        
        # total_f1 = self.ma_f1(torch.tensor(self.val_preds).to(self.device), torch.tensor(self.val_trgts).to(self.device))
        self.log(f"[{state} ma f1]", batch_f1, prog_bar=True)
        self.log(f"[{state} ma acc]", batch_acc, prog_bar=True)
        # self.log(f"[{state} total ma acc]", total_f1, prog_bar=True)
        
        t_decoded = self.tokenizer.batch_decode(batch['t_input_ids'], skip_special_tokens=True)
        c_decoded = self.tokenizer.batch_decode(batch['c_input_ids'], skip_special_tokens=True)
        
        self.titles += t_decoded
        self.comments += c_decoded
        
        return (preds, trgts, batch_f1, batch_acc)
    
    def training_epoch_end(self, outputs, state='train') :
        self.default_epoch_end(outputs, state)
        self.preds.clear()
        self.trgts.clear()
        
    def validation_epoch_end(self, outputs, state='valid') :
        self.default_epoch_end(outputs, state)
        self.val_preds.clear()
        self.val_trgts.clear()
        
    def test_epoch_end(self, outputs, state='test') :
        # loss = torch.mean(torch.tensor([output['loss'] for output in outputs]))
        # acc = torch.mean(torch.tensor([output['acc'] for output in outputs]))
        # f1 = torch.mean(torch.tensor([output[2] for output in outputs]))
        # acc = torch.mean(torch.tensor([output[3] for output in outputs]))
        ma_acc = self.ma_acc(torch.tensor(self.val_preds).to(self.device), torch.tensor(self.val_trgts).to(self.device))
        ma_f1 = self.ma_f1(torch.tensor(self.val_preds).to(self.device), torch.tensor(self.val_trgts).to(self.device))
        
        mean = self.timings.mean()
        summ = self.timings.sum()
        
        self.log('total_time', summ, on_epoch=True, prog_bar=True)
        self.log('mean_time', mean, on_epoch=True, prog_bar=True)
        self.log('total_example', len(self.timings), on_epoch=True, prog_bar=True)
        
        # self.log(f"{state}_f1", ma_f1, prog_bar=True)
        self.log(f"{state}_ma_f1", ma_f1, prog_bar=True)
        self.log(f"{state}_ma_acc", ma_acc, prog_bar=True)
        
        df = {
            'preds' : self.val_preds,
            'trgts' : self.val_trgts,
            'titles' : self.titles,
            'comments' : self.comments
        }

        # df = pd.DataFrame(df)
        if '/' in self.args.model :
            model_name = self.args.model.split('/')[1]
        else : model_name = self.args.model
        with open(f'/home/nykim/2024_spring/02_sota/nayak/03_csv/{self.args.d_name}-{self.args.mode}-{model_name}.pkl', 'wb') as wp :
            pickle.dump(df, wp)
            wp.close()
        # df.to_csv(f'/home/nykim/HateSpeech/09_TitlePrediction/07_sota/Nayak/results/{self.args.d_name}-{self.args.mode}-{model_name}.csv')
        
        self.val_preds.clear()
        self.val_trgts.clear()
        self.titles.clear()
        self.comments.clear()
        
    def configure_optimizers(self) :
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150)
        return [optimizer], [lr_scheduler]