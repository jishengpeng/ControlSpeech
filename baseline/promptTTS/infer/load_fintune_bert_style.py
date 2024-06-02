import torch
from torch import nn
import torch.nn.functional as F


class Bert_Style_Finetune(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre_net = nn.Linear(768, 256)
        self.activate = nn.ReLU()
        self.pitch_head = nn.Linear(256, 3) 
        self.dur_head = nn.Linear(256, 3) 
        self.energy_head = nn.Linear(256, 3) 
        self.emotion_head = nn.Linear(256, 8) 


    def forward(self,style_embed,**kwargs):
        ret = {}
        x = style_embed
        padding_mask = (x.abs().sum(-1)==0)
        x = self.activate(self.pre_net(x))

        x = x * (1 - padding_mask.unsqueeze(-1).float())
        x = x.sum(dim=1) / (1 - padding_mask.float()
                            ).sum(dim=1, keepdim=True)  # Compute average
        ret['pooling_embed'] = x
        
        ret["emotion_logits"] = self.emotion_head(x)
        ret["pitch_logits"] = self.pitch_head(x)
        ret["dur_logits"] = self.dur_head(x)
        ret["energy_logits"] = self.energy_head(x)
        return ret

