import argparse
import numpy as np
import os
import torch
import random

from transformers import T5Tokenizer, T5TokenizerFast
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from FrHAREDataModule import FrHAREDataModule
from FrHARE import FlanT5FrHARE

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DIVICES']="0,1"

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
    # parser.add_argument('--model', type=str, default='paust/pko-t5-small') # google/flan-t5-small
    parser.add_argument('--model', type=str, default='google/flan-t5-small') # google/flan-t5-small
    # parser.add_argument('--tokenizer', type=str, default='paust/pko-t5-small')
    parser.add_argument('--tokenizer', type=str, default='google/flan-t5-small')
    parser.add_argument('--d_name', type=str, default='en_IMSyPP') # dataset name
    
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--randomseed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=512)
    
    parser.add_argument('--ckpt_save_path', type=str, default='/home/nykim/2024_spring/02_sota/hare/01_code/flan_t5/ckpt')
    
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    if args.d_name == 'kold' or args.d_name == 'beep' :
        tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer)
    else :
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    set_random_seed(random_seed=args.randomseed)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                                    monitor='valid_rouge2',
                                                    dirpath=os.path.join(args.ckpt_save_path, args.d_name),
                                                    filename=f'{args.d_name}-{args.randomseed}' + '-{epoch:02d}-{valid_rouge2:.3f}',
                                                    verbose=False,
                                                    save_last=False,
                                                    mode='max',
                                                    save_top_k=1,
                                                    )
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.ckpt_save_path, args.d_name))
    lr_logger = pl.callbacks.LearningRateMonitor()
    
    device = torch.device("cuda")
    
    dm = FrHAREDataModule(args, tokenizer)
    
    if args.gpus == 2 :
        trainer = pl.Trainer(
            logger=tb_logger,
            callbacks=[checkpoint_callback, lr_logger],
            default_root_dir=args.ckpt_save_path,
            max_epochs=args.epoch,
            accelerator='gpu',
            gpus=[0,1],
            precision=16,
            # devices=[args.gpu]
            # devices=[0, 1],
            strategy="ddp"
        )
    elif args.gpus == 1 :
        trainer = pl.Trainer(
            logger=tb_logger,
            callbacks=[checkpoint_callback, lr_logger],
            default_root_dir=args.ckpt_save_path,
            max_epochs=args.epoch,
            accelerator='gpu',
            gpus=[args.gpu],
        )
    
    model = FlanT5FrHARE(args, tokenizer)
    trainer.fit(model, dm)