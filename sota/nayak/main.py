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

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
    parser.add_argument('--model', type=str, default='bert-base-multilingual-uncased') # bert-base-uncased
    # parser.add_argument('--model', type=str, default='monologg/kobert') beomi/kcbert-base
    # parser.add_argument('--tokenizer', type=str, default='bert-base-multilingual-uncased') # beomi/kcbert-base
    # parser.add_argument('--tokenizer', type=str, default='monologg/kobert')
    parser.add_argument('--mode', type=str, default='single') # dual or single
    parser.add_argument('--d_name', type=str, default='beep') # dataset name
    
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--randomseed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=300) # 512
    
    parser.add_argument('--ckpt_save_path', type=str, default='/home/nykim/2024_spring/02_sota/nayak/00_ckpt')
    
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    tokenizer = BertTokenizer.from_pretrained(args.model)
    
    torch.cuda.synchronize()
    # tokenizer2 = BertTokenizer.from_pretrained(args.tokenizer2)
    set_random_seed(random_seed=args.randomseed)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                                    monitor='valid_f1',
                                                    dirpath=os.path.join(args.ckpt_save_path, args.d_name, args.model, str(args.mode)),
                                                    filename=f'{args.d_name}-{args.model}-{args.mode}' + '-{epoch:02d}-{valid_f1:.3f}',
                                                    verbose=False,
                                                    save_last=False,
                                                    mode='max',
                                                    save_top_k=1,
                                                    )
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.ckpt_save_path, args.d_name, args.model,str(args.mode)))
    lr_logger = pl.callbacks.LearningRateMonitor()
    
    device = torch.device("cuda")
    
    # trainer = pl.Trainer(
    #     logger=tb_logger,
    #     callbacks=[checkpoint_callback, lr_logger],
    #     default_root_dir=args.ckpt_save_path,
    #     max_epochs=args.epoch,
    #     accelerator='gpu',
    #     devices=[args.gpu]
    # )
    
    dm = NayakDataModule(args, tokenizer)
    
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_logger],
        default_root_dir=args.ckpt_save_path,
        max_epochs=args.epoch,
        accelerator='gpu',
        devices=[args.gpu],
        # precision=16,
        # strategy='ddp'
    )
    
    model = NayakModel(args, tokenizer)
    trainer.fit(model, dm)
