import pandas as pd
import pickle
# import numpy as np

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score, Recall, Precision
import pytorch_lightning as pl
from transformers import BertModel, RobertaModel, ElectraModel, AutoConfig, BertPreTrainedModel, RobertaPreTrainedModel, ElectraPreTrainedModel

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

class DualTPClassification(BertPreTrainedModel) :
    def __init__(self, config) :
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.config.output_attentions=True
        # lambda value for loss
        self.lam = config.lam
        
        self.bert = BertModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        
        self.dropout = nn.Dropout(classifier_dropout)
        self.hs_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.tp_classifier = nn.Linear(config.hidden_size, 2)
        
        self.post_init()
        
    def forward(
        self,
        c_input_ids=None,
        c_attention_mask=None,
        t_input_ids=None,
        t_attention_mask=None,
        hs_labels=None,
        tp_input_ids=None,
        tp_attention_mask=None,
        tp_labels=None
    ) :
        return_dict=None
        
        t_outputs = self.bert(
            input_ids=t_input_ids,
            attention_mask=t_attention_mask,
            return_dict=return_dict
        )
        t_pooled_output = t_outputs[1]
        
        c_outputs = self.bert(
            input_ids=c_input_ids,
            attention_mask=c_attention_mask,
            return_dict=return_dict
        )
        c_pooled_output = c_outputs[1]
        
        avg_pooled_output = (t_pooled_output + c_pooled_output) / 2
        avg_pooled_output = self.dropout(avg_pooled_output)
        hs_logits = self.hs_classifier(avg_pooled_output)
        
        tp_outputs = self.bert(
            input_ids=tp_input_ids,
            attention_mask=tp_attention_mask,
            return_dict=return_dict
        )
        tp_pooled_output = tp_outputs[1]
        tp_logits = self.tp_classifier(tp_pooled_output)
        
        hs_loss = None
        tp_loss = None
        
        if hs_labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (hs_labels.dtype == torch.long or hs_labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    hs_loss = loss_fct(hs_logits.squeeze(), hs_labels.squeeze())
                else:
                    hs_loss = loss_fct(hs_logits, hs_labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                hs_loss = loss_fct(hs_logits.view(-1, self.num_labels), hs_labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                hs_loss = loss_fct(hs_logits, hs_labels)
                
        if tp_labels is not None :
            loss_fct = CrossEntropyLoss()
            tp_loss = loss_fct(tp_logits.view(-1, 2), tp_labels.view(-1))
            
        loss = self.lam * hs_loss + (1 - self.lam) * tp_loss
        return {
            'loss' : loss,
            'hs_logits' : hs_logits,
            # 't_hidden_states' : t_outputs.hidden_states,
            # 't_attentions' : t_outputs.attentions,
            # 'c_hidden_states' : c_outputs.hidden_states,
            # 'c_attentions' : c_outputs.attentions,
            'tp_logits' : tp_logits,
            # 'tp_hidden_states' : tp_outputs.hidden_states,
            # 'tp_attentions' : tp_outputs.attentions
        }

class SingleTPClassification(BertPreTrainedModel) :
    def __init__(self, config) :
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.config.output_attentions=True
        # lambda value for loss
        self.lam = config.lam
        
        self.bert = BertModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        
        self.dropout = nn.Dropout(classifier_dropout)
        self.hs_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.tp_classifier = nn.Linear(config.hidden_size, 2)
        
        self.post_init()
    
    def mean_pooling(self, last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def forward(
        self, 
        hs_input_ids, 
        hs_attention_mask, 
        hs_labels, 
        tp_input_ids,
        tp_attention_mask,
        tp_labels, 
        return_dict=None
    ) :
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        hs_outputs = self.bert(
            input_ids=hs_input_ids,
            attention_mask=hs_attention_mask,
            return_dict=return_dict
        )
        
        hs_pooled_output = hs_outputs[1]
        # hs_pooled_output = self.mean_pooling(hs_outputs[0], hs_attention_mask)
        hs_pooled_output = self.dropout(hs_pooled_output)
        hs_logits = self.hs_classifier(hs_pooled_output)
        
        tp_outputs = self.bert(
            input_ids=tp_input_ids,
            attention_mask=tp_attention_mask,
            return_dict=return_dict
        )
        tp_pooled_output = tp_outputs[1]
        # tp_pooled_output = self.mean_pooling(tp_outputs[0], tp_attention_mask)
        tp_pooled_output = self.dropout(tp_pooled_output)
        tp_logits = self.tp_classifier(tp_pooled_output)
        
        hs_loss = None
        tp_loss = None
        
        if hs_labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (hs_labels.dtype == torch.long or hs_labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    hs_loss = loss_fct(hs_logits.squeeze(), hs_labels.squeeze())
                else:
                    hs_loss = loss_fct(hs_logits, hs_labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                hs_loss = loss_fct(hs_logits.view(-1, self.num_labels), hs_labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                hs_loss = loss_fct(hs_logits, hs_labels)
                
        if tp_labels is not None :
            loss_fct = CrossEntropyLoss()
            tp_loss = loss_fct(tp_logits.view(-1, 2), tp_labels.view(-1))
            
        loss = self.lam * hs_loss + (1 - self.lam) * tp_loss
        return {
            'loss' : loss,
            'hs_logits' : hs_logits,
            # 'hs_hidden_states' : hs_outputs.hidden_states,
            # 'hs_attentions' : hs_outputs.attentions,
            'tp_logits' : tp_logits,
            # 'tp_hidden_states' : tp_outputs.hidden_states,
            # 'tp_attentions' : tp_outputs.attentions
        }


class HateSpeechDetection(pl.LightningModule) :
    def __init__(self, args, tokenizer) :
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.lam = args.lam
        
        self.d_size = args.data_size
        self.learning_rate = args.learning_rate
        self.csv_path = args.csv_path
        
        self.mode = args.mode
        
        self.softmax = nn.Softmax(dim=1)
        
        config = AutoConfig.from_pretrained(args.model)
        if args.d_name == 'kold' :
            config.num_labels = 2
        elif args.d_name == 'beep' :
            config.num_labels = 3
        elif 'IMSyPP' in args.d_name :
            config.num_labels = 4
        config.lam = args.lam
 
        if args.mode <= 1 :
            self.model = SingleTPClassification.from_pretrained(args.model, config=config)
        else :
            self.model = DualTPClassification.from_pretrained(args.model, config=config)

        # Hate Speech Detection metrics
        self.ma_acc = Accuracy(task='multiclass', num_classes=self.model.config.num_labels, average='macro')
        self.ma_f1 = F1Score(task='multiclass', num_classes=self.model.config.num_labels, average='macro')
        
        # Title Prediction Detection metrics
        self.tp_ma_acc = Accuracy(task='multiclass', num_classes=2, average='macro')
        self.tp_ma_f1 = F1Score(task='multiclass', num_classes=2, average='macro')
        
        self.save_hyperparameters()
        
        self.hs_preds = []
        self.hs_trgts = []
        self.tp_preds = []
        self.tp_trgts = []
        self.hs_sentences = []
        self.tp_sentences = []
        
        self.val_hs_preds = []
        self.val_hs_trgts = []
        self.val_tp_preds = []
        self.val_tp_trgts = []
        
        self.timings = torch.tensor([0.0])
        
        
    def forward(
        self,
        hs_input_ids=None,
        hs_attention_mask=None,
        t_input_ids=None,
        t_attention_mask=None,
        c_input_ids=None,
        c_attention_mask=None,
        hs_labels=None,
        tp_input_ids=None,
        tp_attention_mask=None,
        tp_labels=None, 
    ) :
        if self.mode <= 1 :
            outputs = self.model(
                hs_input_ids=hs_input_ids,
                hs_attention_mask=hs_attention_mask,
                hs_labels=hs_labels,
                tp_input_ids=tp_input_ids,
                tp_attention_mask=tp_attention_mask,
                tp_labels=tp_labels
            )
        else :
            outputs = self.model(
                t_input_ids=t_input_ids,
                t_attention_mask=t_attention_mask,
                c_input_ids=c_input_ids,
                c_attention_mask=c_attention_mask,
                hs_labels=hs_labels,
                tp_input_ids=tp_input_ids,
                tp_attention_mask=tp_attention_mask,
                tp_labels=tp_labels
            )
        
        return outputs
    
    def default_step(self, batch, batch_idx, state=None) :
        starter.record()
        if self.mode <= 1 :
            outputs = self(
                hs_input_ids=batch['hs_input_ids'].to(self.device),
                hs_attention_mask=batch['hs_attention_mask'].to(self.device),
                hs_labels=batch['hs_label'].to(self.device),
                tp_input_ids=batch['tp_input_ids'].to(self.device),
                tp_attention_mask=batch['tp_attention_mask'].to(self.device),
                tp_labels=batch['tp_label'].to(self.device)
            )
        else :
            outputs = self(
                t_input_ids=batch['hst_input_ids'].to(self.device),
                t_attention_mask=batch['hst_attention_mask'].to(self.device),
                c_input_ids=batch['hsc_input_ids'].to(self.device),
                c_attention_mask=batch['hsc_attention_mask'].to(self.device),
                hs_labels=batch['hs_label'].to(self.device),
                tp_input_ids=batch['tp_input_ids'].to(self.device),
                tp_attention_mask=batch['tp_attention_mask'].to(self.device),
                tp_labels=batch['tp_label'].to(self.device)
            )
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        self.timings = torch.concat((self.timings, torch.tensor([round(curr_time,2)/1000])))
        
        loss = outputs['loss']
        hs_logits = outputs['hs_logits']
        tp_logits = outputs['tp_logits']
        
        hs_trgts = batch['hs_label']
        hs_preds = hs_logits.argmax(dim=1).to(self.device)
        
        tp_trgts = batch['tp_label']
        tp_preds = tp_logits.argmax(dim=1).to(self.device)
        
        if state == 'train' :
            self.hs_trgts += hs_trgts.tolist()
            self.hs_preds += hs_preds.tolist()
            self.tp_trgts += tp_trgts.tolist()
            self.tp_preds += tp_preds.tolist()
        elif state == 'test' or state == 'valid' :
            self.val_hs_trgts += hs_trgts.tolist()
            self.val_hs_preds += hs_preds.tolist()
            self.val_tp_trgts += tp_trgts.tolist()
            self.val_tp_preds += tp_preds.tolist()
            if state == 'test' :
                hs_decoded = self.tokenizer.batch_decode(batch['hs_input_ids'], skip_special_tokens=True)
                tp_decoded = self.tokenizer.batch_decode(batch['tp_input_ids'], skip_special_tokens=True)
                
                self.hs_sentences += hs_decoded
                self.tp_sentences += tp_decoded
            
        ma_acc = self.ma_acc(hs_preds, hs_trgts)
        ma_f1 = self.ma_f1(hs_preds, hs_trgts)
        
        tp_ma_acc = self.tp_ma_acc(tp_preds, tp_trgts)
        tp_ma_f1 = self.tp_ma_f1(tp_preds, tp_trgts)
        
        self.log(f"[{state} loss]", loss, prog_bar=True)
        self.log(f"[{state} hs acc]", ma_acc, prog_bar=True)
        self.log(f"[{state} hs f1]", ma_f1, prog_bar=True)
        self.log(f"[{state} tp acc]", tp_ma_acc, prog_bar=True)
        self.log(f"[{state} tp f1]", tp_ma_f1, prog_bar=True)
        
        return {
            'loss' : loss,
            'acc' : ma_acc,
            'f1' : ma_f1
        }
    
    def default_epoch_end(self, outputs, state=None) :
        loss = torch.mean(torch.tensor([output['loss'] for output in outputs]))
        # ma_f1 = torch.mean(torch.tensor([output['f1'] for output in outputs]))
        
        if state == 'train' :
            hs_ma_acc = self.ma_acc(torch.tensor(self.hs_preds).to(self.device), torch.tensor(self.hs_trgts).to(self.device))
            hs_ma_f1 = self.ma_f1(torch.tensor(self.hs_preds).to(self.device), torch.tensor(self.hs_trgts).to(self.device))
            tp_ma_acc = self.tp_ma_acc(torch.tensor(self.tp_preds).to(self.device), torch.tensor(self.tp_trgts).to(self.device))
            tp_ma_f1 = self.tp_ma_f1(torch.tensor(self.tp_preds).to(self.device), torch.tensor(self.tp_trgts).to(self.device))
        else :
            hs_ma_acc = self.ma_acc(torch.tensor(self.val_hs_preds).to(self.device), torch.tensor(self.val_hs_trgts).to(self.device))
            hs_ma_f1 = self.ma_f1(torch.tensor(self.val_hs_preds).to(self.device), torch.tensor(self.val_hs_trgts).to(self.device))
            tp_ma_acc = self.tp_ma_acc(torch.tensor(self.val_tp_preds).to(self.device), torch.tensor(self.val_tp_trgts).to(self.device))
            tp_ma_f1 = self.tp_ma_f1(torch.tensor(self.val_tp_preds).to(self.device), torch.tensor(self.val_tp_trgts).to(self.device))
        mean = self.timings.mean()
        summ = self.timings.sum()
        
        self.log('total_time', summ, on_epoch=True)
        self.log('mean_time', mean, on_epoch=True)
        self.log('total_example', len(self.timings), on_epoch=True)
        
        self.log(f'{state}_loss', loss, on_epoch=True, prog_bar=True)
        
        self.log(f'{state}_hs_ma_acc', hs_ma_acc, on_epoch=True, prog_bar=True)
        self.log(f'{state}_hs_ma_f1', hs_ma_f1, on_epoch=True, prog_bar=True)
        
        self.log(f'{state}_tp_ma_acc', tp_ma_acc, on_epoch=True, prog_bar=True)
        self.log(f'{state}_tp_ma_f1', tp_ma_f1, on_epoch=True, prog_bar=True)
        
        
        
    def training_step(self, batch, batch_idx, state='train') :
        result = self.default_step(batch, batch_idx, state)
        return result
    
    def validation_step(self, batch, batch_idx, state='valid') :
        result = self.default_step(batch, batch_idx, state)
        return result
    
    def test_step(self, batch, batch_idx, state='test') :
        starter.record()
        if self.mode <= 1 :
            outputs = self(
                hs_input_ids=batch['hs_input_ids'].to(self.device),
                hs_attention_mask=batch['hs_attention_mask'].to(self.device),
                hs_labels=batch['hs_label'].to(self.device),
                tp_input_ids=batch['tp_input_ids'].to(self.device),
                tp_attention_mask=batch['tp_attention_mask'].to(self.device),
                tp_labels=batch['tp_label'].to(self.device)
            )
        else :
            outputs = self(
                t_input_ids=batch['hst_input_ids'].to(self.device),
                t_attention_mask=batch['hst_attention_mask'].to(self.device),
                c_input_ids=batch['hsc_input_ids'].to(self.device),
                c_attention_mask=batch['hsc_attention_mask'].to(self.device),
                hs_labels=batch['hs_label'].to(self.device),
                tp_input_ids=batch['tp_input_ids'].to(self.device),
                tp_attention_mask=batch['tp_attention_mask'].to(self.device),
                tp_labels=batch['tp_label'].to(self.device)
            )
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        self.timings = torch.concat((self.timings, torch.tensor([round(curr_time,2)/1000])))
        
        logits = outputs['hs_logits']
        # loss = outputs['loss']
        
        preds = self.softmax(logits).tolist()
        trgts = batch['hs_label'].tolist()
        
        batch_f1 = self.ma_f1(torch.tensor(preds).to(self.device), torch.tensor(trgts).to(self.device))
        batch_acc = self.ma_acc(torch.tensor(preds).to(self.device), torch.tensor(trgts).to(self.device))
        
        self.val_hs_preds += preds
        self.val_hs_trgts += trgts
        
        # total_f1 = self.ma_f1(torch.tensor(self.val_preds).to(self.device), torch.tensor(self.val_trgts).to(self.device))
        self.log(f"[{state} ma f1]", batch_f1, prog_bar=True)
        self.log(f"[{state} ma acc]", batch_acc, prog_bar=True)
        # self.log(f"[{state} total ma acc]", total_f1, prog_bar=True)
        
        if self.mode == 2 :
            t_decoded = self.tokenizer.batch_decode(batch['hst_input_ids'], skip_special_tokens=True)
            c_decoded = self.tokenizer.batch_decode(batch['hsc_input_ids'], skip_special_tokens=True)
        else :
            t_decoded = self.tokenizer.batch_decode(batch['hs_input_ids'], skip_special_tokens=True)
            c_decoded = self.tokenizer.batch_decode(batch['tp_input_ids'], skip_special_tokens=True)
        self.hs_sentences += t_decoded
        self.tp_sentences += c_decoded
        
        return {
            'loss' : outputs['loss']
        }
    
    def training_epoch_end(self, outputs, state='train') :
        self.default_epoch_end(outputs, state)
        
        self.hs_preds.clear()
        self.hs_trgts.clear()
        self.tp_preds.clear()
        self.tp_trgts.clear()
        
    def validation_epoch_end(self, outputs, state='valid') :
        self.default_epoch_end(outputs, state)
        
        self.val_hs_preds.clear()
        self.val_hs_trgts.clear()
        self.val_tp_preds.clear()
        self.val_tp_trgts.clear()
        
    def test_epoch_end(self, outputs, state='test') :
        # self.default_epoch_end(outputs, state)
        ma_acc = self.ma_acc(torch.tensor(self.val_hs_preds).to(self.device), torch.tensor(self.val_hs_trgts).to(self.device))
        ma_f1 = self.ma_f1(torch.tensor(self.val_hs_preds).to(self.device), torch.tensor(self.val_hs_trgts).to(self.device))
        
        mean = self.timings.mean()
        summ = self.timings.sum()
        
        self.log('total_time', summ, on_epoch=True, prog_bar=True)
        self.log('mean_time', mean, on_epoch=True, prog_bar=True)
        self.log('total_example', len(self.timings), on_epoch=True, prog_bar=True)
        
        self.log(f"{state}_ma_f1", ma_f1, prog_bar=True)
        self.log(f"{state}_ma_acc", ma_acc, prog_bar=True)
        
        df = {
            'hs_preds' : self.val_hs_preds,
            'hs_trgts' : self.val_hs_trgts,
            'tp_preds' : self.val_tp_preds,
            'tp_trgts' : self.val_tp_trgts,
            'hs_sentences' : self.hs_sentences,
            'tp_sentences' : self.tp_sentences
        }
        if '/' in self.args.model :
            model_name = self.args.model.split('/')[1]
        else : model_name = self.args.model
        with open(f'/home/nykim/2024_spring/01_TitlePrediction/03_pickles/{self.args.d_name}-{self.args.mode}-{self.lam}-{model_name}.pkl', 'wb') as wp :
            pickle.dump(df, wp)
            wp.close()
        # df = pd.DataFrame(df)
        # df.to_csv(f'{self.csv_path}/{self.args.d_name}-{self.args.mode}-{self.args.lam}-{self.args.random_ratio}-{self.d_size}.csv')
        
        self.val_hs_preds.clear()
        self.val_hs_trgts.clear()
        self.val_tp_preds.clear()
        self.val_tp_trgts.clear()
        self.hs_sentences.clear()
        self.tp_sentences.clear()
        
    def configure_optimizers(self) :
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150)
        return [optimizer], [lr_scheduler]
        
            