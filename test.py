import argparse

import pandas as pd
import numpy as np

import glob

from tqdm import tqdm

import torch

import timm

from src.dataset import CustomDataset
from src.model import CustomModel
from src.utils import to_dtype

import warnings
warnings.filterwarnings(action = "ignore")
    
def test(args, model, loader):
    model.eval()
    
    val_acc1, val_acc5 = [], []
    for _, sample in enumerate(tqdm(loader)):
        inputs, targets = [x.to(args.device) for x in sample]

        inputs = inputs.reshape(-1, args.clip_length, 3, args.input_size, args.input_size)
        
        with torch.no_grad():
            outputs = model(inputs)
            
            if args.view == "single":
                outputs = torch.softmax(outputs, dim = -1)
                
            elif args.view == "multi":
                outputs = outputs.reshape(args.batch_size, args.n_temporal * args.n_spatial, -1)
                outputs = torch.softmax(outputs, dim = -1).mean(1)
                
            else:
                raise NotImplementedError()

            acc1, acc5 = timm.utils.accuracy(outputs, targets, topk = (1, 5))
            
            val_acc1.append(acc1.cpu().item())
            val_acc5.append(acc5.cpu().item())
            
    print('acc1 : ', np.mean(val_acc1))
    print('acc5 : ', np.mean(val_acc5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type = str, default = "MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--pretrained", type = bool, default = True)
    
    parser.add_argument("--pretrained_dir", type = str, default = "huggingface")
    parser.add_argument("--pretrained_name", type = str, default = "MCG-NJU/videomae-base")
    parser.add_argument("--mode", type = str, default = "supervised")
    
    parser.add_argument("--trained_dir", type = str, default = "weights")
    
    parser.add_argument("--n_temporal", type = int, default = 5)
    parser.add_argument("--n_spatial", type = int, default = 3)
    parser.add_argument("--clip_length", type = int, default = 16)
    parser.add_argument("--sample_rate", type = int, default = 4)
    parser.add_argument("--shortest_edge", type = int, default = 224)
    
    parser.add_argument("--input_size", type = int, default = 224)
    parser.add_argument("--n_sample", type = int, default = -1)
    
    parser.add_argument("--batch_size", type = int, default = 2)
    parser.add_argument("--n_worker", type = int, default = 16)
    
    parser.add_argument("--device", type = str, default = "cuda")
    parser.add_argument("--dtype", type = str, default = "bfloat16")
    
    parser.add_argument("--view", type = str, default = "multi")
    
    parser.add_argument("--attn_implementation", type = str, default = "sdpa")
    parser.add_argument("--hidden_dropout_prob", type = float, default = 0.0)
    parser.add_argument("--attention_probs_dropout_prob", type = float, default = 0.0)
    parser.add_argument("--head_dropout_prob", type = float, default = 0.0)
    
    args = parser.parse_args()
    
    val = pd.read_csv(f"data/Kinetics-400/kinetics400_val_list_videos.txt", sep = " ", header = None)
    
    if args.n_sample > 0:
        val = val[:args.n_sample]

    dataset = CustomDataset(args, df = val, split = "val", is_training = False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = args.batch_size,
        num_workers = args.n_worker,
        shuffle = False,
        drop_last = True,
        )
    
    model = CustomModel(args)
    model = to_dtype(model, args.dtype)
    model = model.to(args.device)
    
    checkpoint = sorted(glob.glob(args.trained_dir + "/*.bin"))[-1]
    checkpoint = torch.load(checkpoint, weights_only = True, map_location = "cpu")
    
    model.load_state_dict(checkpoint)
    
    test(args, model, loader)