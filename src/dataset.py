import numpy as np

import random

import torch

import decord

from transformers import VideoMAEImageProcessor

from src.utils import resize_shortest_edge, pad1d, to_dtype
from src.transforms import CustomTransform
from src.mask import CustomMask

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, df, split, is_training):
        self.args = args
        
        self.df = df
        self.split = split
        self.is_training = is_training

        self.transform = CustomTransform(args, is_training = is_training)
        
        if args.mode == "self-supervised":
            self.mask_generator = CustomMask(args)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path, label = self.df.loc[index]
        
        buffer = decord.VideoReader(f"data/Kinetics-400/videos_{self.split}/" + path, num_threads = 1, ctx = decord.cpu(0))
        
        indices = np.arange(len(buffer))[::self.args.sample_rate]
        indices = pad1d(indices, self.args.clip_length)

        if self.args.view == "single":
            if self.is_training:
                start_id = random.randint(0, len(indices) - self.args.clip_length)
            else:
                start_id = (len(indices) - self.args.clip_length)//2
                
            end_id = start_id + self.args.clip_length
            indices = indices[start_id:end_id]
            
        elif self.args.view == "multi":
            pass
        
        else:
            raise NotImplementedError()

        buffer.seek(0)
        buffer = buffer.get_batch(indices).asnumpy()
        
        if self.args.view == "single":
            video = self.transform(buffer)
            
        elif self.args.view == "multi":
            buffer = resize_shortest_edge(buffer, self.args.shortest_edge)
            
            temporal_step = (buffer.shape[0] - self.args.clip_length) / (self.args.n_temporal - 1)
            spatial_step = (max(buffer.shape[1], buffer.shape[2]) - self.args.shortest_edge) / (self.args.n_spatial - 1)
            
            video = []
            for i in range(self.args.n_temporal):
                for j in range(self.args.n_spatial):
                    temporal_start = int(i * temporal_step)
                    spatial_start = int(j * spatial_step)
                    
                    if buffer.shape[1] >= buffer.shape[2]:
                        x = buffer[temporal_start:temporal_start + self.args.clip_length, spatial_start:spatial_start + self.args.shortest_edge, :, :]
                    else:
                        x = buffer[temporal_start:temporal_start + self.args.clip_length, :, spatial_start:spatial_start + self.args.shortest_edge, :]
                        
                    x = self.transform(x)
                    
                    video.append(x)
            video = torch.stack(video, dim = 0)
            
        else:
            raise NotImplementedError()
        
        video = to_dtype(video, self.args.dtype)
        
        if self.args.mode in ["supervised", "test"]:
            label = torch.tensor(label, dtype = torch.long)
            return video, label
        
        elif self.args.mode == "self-supervised":
            return video, self.mask_generator()

        else:
            raise NotImplementedError()            
    