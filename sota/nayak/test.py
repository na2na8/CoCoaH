import argparse
import numpy as np
import os
import torch
import random
from datetime import datetime

from transformers import BertTokenizer
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from DataModule import NayakDataModule
from Model import NayakModel
# from TestModel import Ensemble, NayakModel

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    pl.seed_everything(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set random seed : {random_seed}")
    
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model1', type=str, default='beomi/kcbert-base')bert-base-uncased
    # parser.add_argument('--model2', type=str, default='bert-base-multilingual-uncased') dbmdz/bert-base-italian-cased
    parser.add_argument('--model', type=str, default='bert-base-multilingual-uncased')
    # parser.add_argument('--tokenizer1', type=str, default='beomi/kcbert-base')
    # parser.add_argument('--tokenizer2', type=str, default='bert-base-multilingual-uncased')
    parser.add_argument('--tokenizer', type=str, default='bert-base-multilingual-uncased')
    parser.add_argument('--mode', type=str, default='dual') # dual or single
    parser.add_argument('--d_name', type=str, default='kold') # dataset name
    
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--randomseed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=300)
    
    parser.add_argument('--ckpt_save_path', type=str, default='/home/nykim/2024_spring/02_sota/nayak/00_ckpt')
    parser.add_argument('--ckpt_load_path', type=str, default='/home/nykim/2024_spring/02_sota/nayak/00_ckpt/kold/bert-base-multilingual-uncased/dual/kold-bert-base-multilingual-uncased-dual-epoch=02-valid_f1=0.788.ckpt')
    # parser.add_argument('--ckpt_load_path2', type=str, default='/home/nykim/HateSpeech/09_TitlePrediction/07_sota/Nayak/ckpt/beep/beomi/kcbert-base/dual/beep-beomi/kcbert-base-dual-epoch=01-valid_f1=0.641.ckpt')
    # parser.add_argument('--ckpt_load_path3', type=str, default='/home/nykim/HateSpeech/09_TitlePrediction/07_sota/Nayak/ckpt/beep/bert-base-multilingual-uncased/single/beep-bert-base-multilingual-uncased-single-epoch=03-valid_f1=0.551.ckpt')
    # parser.add_argument('--ckpt_load_path4', type=str, default='/home/nykim/HateSpeech/09_TitlePrediction/07_sota/Nayak/ckpt/beep/bert-base-multilingual-uncased/dual/beep-bert-base-multilingual-uncased-dual-epoch=03-valid_f1=0.554.ckpt')
    
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    print(args.ckpt_load_path)
    
    torch.cuda.synchronize()
    
    tokenizer = BertTokenizer.from_pretrained(args.model)
    # tokenizer2 = BertTokenizer.from_pretrained(args.tokenizer2)
    
    set_random_seed(random_seed=args.randomseed)
    
    device = torch.device("cuda")
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.ckpt_save_path, args.d_name, 'ensemble'))
    
    trainer = pl.Trainer(
        logger=tb_logger,
        default_root_dir=args.ckpt_save_path,
        accelerator='gpu',
        devices=[args.gpu]
    )
    
    dm = NayakDataModule(args, tokenizer).test_dataloader()
    model = NayakModel.load_from_checkpoint(checkpoint_path=args.ckpt_load_path, args=args, tokenizer=tokenizer)
    
    # start = datetime.now()
    trainer.test(model, dm)
    # end = datetime.now()
    # diff = end - start
    # print(diff)
    # print(diff.seconds)
    # print(diff.microseconds/1000)
    # model1 = None
    
    # dm = NayakDataModule(args, tokenizer).test_dataloader()
    # model2 = NayakModel.load_from_checkpoint(checkpoint_path=args.ckpt_load_path2, args=args, model=args.model1, mode='dual', tokenizer=tokenizer1)
    # trainer.test(model2, dm)
    # model2 = None
    
    # dm = NayakDataModule(args, tokenizer).test_dataloader()
    # model3 = NayakModel.load_from_checkpoint(checkpoint_path=args.ckpt_load_path3, args=args, model=args.model2, mode='single', tokenizer=tokenizer2)
    # trainer.test(model3, dm)
    # model3 = None
    
    # dm = NayakDataModule(args, tokenizer).test_dataloader()
    # model4 = NayakModel.load_from_checkpoint(checkpoint_path=args.ckpt_load_path4, args=args, model=args.model2, mode='dual', tokenizer=tokenizer2)
    # trainer.test(model4, dm)
    # model4 = None
    
    