import glob

import torch
import torch.nn as nn

from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEModel, VideoMAEForPreTraining

class CustomModel(nn.Module):
    def __init__(self, args):
        super(CustomModel, self).__init__()
        self.args = args
        
        if args.mode == "supervised":
            config = VideoMAEConfig.from_pretrained(args.model_name)
            config._attn_implementation = args.attn_implementation
            config.hidden_dropout_prob = args.hidden_dropout_prob
            config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
            config.use_mean_pooling = False
 
            self.backbone = VideoMAEModel(config)
            
            if args.pretrained:
                
                if args.pretrained_dir == "huggingface":
                    pretrained_model = VideoMAEForPreTraining.from_pretrained(args.pretrained_name)
                    
                else:
                    checkpoint = sorted(glob.glob("./" + args.pretrained_dir + "/*.bin"))[-1]
                    checkpoint = torch.load(checkpoint, weights_only = True, map_location = "cpu")
                    checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items() if k.startswith("backbone.")}
                    
                    pretrained_config = VideoMAEConfig.from_pretrained(args.pretrained_name)
                    pretrained_model = VideoMAEForPreTraining(pretrained_config)
                    pretrained_model.load_state_dict(checkpoint)
                    
                self.backbone.load_state_dict(pretrained_model.videomae.state_dict())
                    
            self.backbone.layernorm = None
            self.norm = nn.LayerNorm(config.hidden_size) 
            self.dropout = nn.Dropout(p = args.head_dropout_prob)
            self.out = nn.Linear(config.hidden_size, config.num_labels) 
            
            torch.nn.init.xavier_uniform_(self.out.weight)
            
        elif args.mode == "test":
            self.backbone = VideoMAEForVideoClassification.from_pretrained(args.model_name)
            
        elif args.mode == "self-supervised":
            config = VideoMAEConfig.from_pretrained(args.model_name)
            config.norm_pix_loss = args.norm_pix_loss
            config._attn_implementation = args.attn_implementation
            config.hidden_dropout_prob = args.hidden_dropout_prob
            config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
            config.use_mean_pooling = False
            
            self.backbone = VideoMAEForPreTraining(config)
            
        else:
            raise NotImplementedError()
            

    def forward(self, x, mask = None):
        if self.args.mode == "supervised":
            x = self.backbone(x).last_hidden_state
            x = self.norm(x.mean(1))
            x = self.dropout(x)
            x = self.out(x)
            
        elif self.args.mode == "test":
            x = self.backbone(x).logits 
        
        elif self.args.mode == "self-supervised":
            x = self.backbone(x, bool_masked_pos = mask)

        else:
            raise NotImplementedError()
        
        return x