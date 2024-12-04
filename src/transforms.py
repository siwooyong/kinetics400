import numpy as np

import torch
import torch.nn as nn

from torchvision import transforms

import src.augmentation.video_transforms as video_transforms

class ToTensor:
    def __call__(self, x):
        x = torch.as_tensor(x, dtype = torch.float)
        return x
    
class Transpose:
    def __call__(self, x):
        x = x.transpose(0, 3, 1, 2)
        return x
    
class CustomRandaug:
    def __init__(self, args):
        self.transform = video_transforms.create_random_augment(
            input_size = (args.crop_size, args.crop_size),
            auto_augment = args.aa,
            interpolation = args.train_interpolation
            )
         
    def __call__(self, x):
        x = [transforms.ToPILImage()(_) for _ in x]
        x = self.transform(x)
        x = np.stack([np.asarray(_) for _ in x], axis = 0)
        return x

class CustomTransform(nn.Module):
    def __init__(
        self, 
        args, 
        is_training, 
        rescale_factor = 255.0, 
        image_mean = (0.485, 0.456, 0.406), 
        image_std = (0.229, 0.224, 0.225),
        antialias = False,
        inplace = True,
        ):
        
        super(CustomTransform, self).__init__()
        
        self.args = args
        self.is_training = is_training
        
        if is_training:
            
            if args.mode == "supervised":
                self.transform = transforms.Compose([
                    CustomRandaug(args),
                    Transpose(),
                    ToTensor(),
                    transforms.RandomResizedCrop(size = args.input_size, antialias = antialias),
                    transforms.RandomHorizontalFlip(p = args.hflip_prob),
                    transforms.Lambda(lambda x: x / rescale_factor),
                    transforms.Normalize(mean = image_mean, std = image_std, inplace = inplace),
                    transforms.RandomErasing(p = args.erase_prob),
                    ])
                
            elif args.mode == "self-supervised":
                self.transform = transforms.Compose([
                    Transpose(),
                    ToTensor(),
                    transforms.RandomResizedCrop(size = args.input_size, antialias = antialias),
                    transforms.RandomHorizontalFlip(p = args.hflip_prob),
                    transforms.Lambda(lambda x: x / rescale_factor),
                    transforms.Normalize(mean = image_mean, std = image_std, inplace = inplace),
                    ])
            
            else:
                raise NotImplementedError()
            
        else:
            self.transform = transforms.Compose([
                Transpose(),
                ToTensor(),
                transforms.Resize(size = args.input_size, antialias = antialias),
                transforms.CenterCrop(size = args.input_size),
                transforms.Lambda(lambda x: x / rescale_factor),
                transforms.Normalize(mean = image_mean, std = image_std, inplace = inplace),
                ])
            
    def forward(self, x):        
        x = self.transform(x)
        return x