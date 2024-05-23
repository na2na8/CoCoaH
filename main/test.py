import argparse
import numpy as np
import os
import torch
import random
from datetime import datetime

from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from DataModule import TitlePredictionDataModule
from Model import HateSpeechDetection

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
    parser.add_argument('--model', type=str, default='beomi/kcbert-base') # beomi/kcbert-base
    parser.add_argument('--tokenizer', type=str, default='beomi/kcbert-base')
    parser.add_argument('--d_name', type=str, default='kold') # dataset name
    parser.add_argument('--mode', type=int, default=1) # input mode : 0 : title + comment, 1 : comment + title
    parser.add_argument('--lam', type=float, default=1.0) # lambda for loss 0~1
    parser.add_argument('--random_ratio', type=float, default=0.5) # answer title ratio for title prediction
    parser.add_argument('--data_size', type=float, default=1.0)
    
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--randomseed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=300)
    
    parser.add_argument('--csv_path', type=str, default='/home/nykim/2024_spring/01_TitlePrediction/02_csv/')
    parser.add_argument('--ckpt_save_path', type=str, default='/home/nykim/2024_spring/01_TitlePrediction/04_test')
    parser.add_argument('--ckpt_load_path', type=str, default='')
    
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    torch.cuda.synchronize()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    set_random_seed(random_seed=args.randomseed)
    
    device = torch.device("cuda")
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.ckpt_save_path, args.d_name, str(args.mode)))
    
    trainer = pl.Trainer(
        logger=tb_logger,
        default_root_dir=args.ckpt_save_path,
        accelerator='gpu',
        devices=[0, 1],
        strategy='ddp'
    )
    
    dm = TitlePredictionDataModule(args, tokenizer).test_dataloader()
    
    model = HateSpeechDetection.load_from_checkpoint(checkpoint_path=args.ckpt_load_path, args=args, tokenizer=tokenizer)

    trainer.test(model, dm)
