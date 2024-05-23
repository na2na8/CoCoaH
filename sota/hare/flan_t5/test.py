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
    parser.add_argument('--model', type=str, default='google/flan-t5-small') # paust/pko-t5-small
    parser.add_argument('--tokenizer', type=str, default='google/flan-t5-small') # google/flan-t5-small
    parser.add_argument('--d_name', type=str, default='en_IMSyPP_nocon') # dataset name
    
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--randomseed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=512)
    
    parser.add_argument('--ckpt_save_path', type=str, default='/home/nykim/2024_spring/02_sota/hare/01_code/flan_t5/ckpt')
    parser.add_argument('--ckpt_load_path', type=str)
    
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    
    if args.d_name == 'kold' or args.d_name == 'beep' :
        tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer)
    else :
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    set_random_seed(random_seed=args.randomseed)
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.ckpt_save_path, args.d_name))
    
    device = torch.device("cuda")
    
    trainer = pl.Trainer(
        logger=tb_logger,
        default_root_dir=args.ckpt_save_path,
        accelerator='gpu',
        devices=[args.gpu]
    )
    
    dm = FrHAREDataModule(args, tokenizer).test_dataloader()
    
    model = FlanT5FrHARE.load_from_checkpoint(checkpoint_path=args.ckpt_load_path, args=args, tokenizer=tokenizer)
    trainer.test(model, dm)