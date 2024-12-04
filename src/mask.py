import torch

class CustomMask:
    def __init__(self, args):
        super(CustomMask, self).__init__()
        self.args = args 
        
        assert (self.args.input_size % self.args.patch_size) == 0
        assert (self.args.clip_length % self.args.tubelet_size) == 0
        
        self.n_patch = (self.args.input_size // self.args.patch_size) ** 2
        self.n_mask = int(self.n_patch * args.mask_ratio)
        self.n_tubelet = self.args.clip_length // self.args.tubelet_size
        

    def __call__(self):
        mask = torch.cat([
            torch.ones(self.n_mask, dtype = torch.bool), 
            torch.zeros(self.n_patch - self.n_mask, dtype = torch.bool)
            ], dim = 0)
        mask = mask[torch.randperm(self.n_patch)]
        
        mask = mask.unsqueeze(0)
        mask = mask.repeat(self.n_tubelet, 1)
        mask = mask.reshape(-1)
        return mask