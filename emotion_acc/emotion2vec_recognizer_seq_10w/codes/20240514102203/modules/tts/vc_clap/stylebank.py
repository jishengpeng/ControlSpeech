import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class StaticSelfAttn(nn.Module):
    def __init__(self, n_pos, d_model, nhead):
        super(StaticSelfAttn, self).__init__()
        # Kaiming Initialize
        self.weight = nn.Parameter(torch.randn(1, 1, nhead, n_pos, n_pos) * np.sqrt(2 / n_pos), requires_grad=True)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.nhead = nhead
        self.d_model = d_model
        assert self.d_model % self.nhead == 0

    def forward(self, x):
        """
        :param x: T x B x C
        :return: T x B x C
        """
        T, B, C = x.shape
        x = self.value_proj(x).reshape(T, B, self.nhead, C//self.nhead).permute((3, 1, 2, 0)).unsqueeze(-2)  # c B H 1 T
        x = torch.matmul(x, self.weight).squeeze(-2).permute((3, 1, 2, 0)).reshape(T, B, C)
        return self.out_proj(x)
    

class CrossSelfAttnBlock(nn.Module):
    def __init__(self, n_pos, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", **kwargs):
        super(CrossSelfAttnBlock, self).__init__()
        self.self_attn = StaticSelfAttn(n_pos, d_model, nhead)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, cross_res=True, return_attn=False):
        """
        This module is different from transformer decoder layer in the below aspects:
        1. Do cross attention first, then do self attention
        2. Self attention is implemented with static self attention.
        3. Cross attention don't have residue connection when this module is used as the first layer.
        """
        attn_maps = {}
        tgt2, attn_map = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        if return_attn:
            attn_maps['cross_attn'] = attn_map.detach().cpu().numpy()
        if cross_res:
            tgt = tgt + self.dropout1(tgt2)
        else:
            tgt = self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.self_attn(tgt)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn_maps
    

class StyleBankExtractor(nn.Module):
    def __init__(self, dim_in, n_layers, n_emb, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(StyleBankExtractor, self).__init__()
        self.prototype = nn.Parameter(torch.randn(n_emb, d_model))
        self.proj_in = nn.Linear(dim_in, d_model)
        self.layers = nn.ModuleList([CrossSelfAttnBlock(n_emb, d_model, nhead, dim_feedforward, dropout, activation) for _ in range(n_layers)])

    def forward(self, memory, return_attn=False, padding_mask=None):
        """
        :param memory: T x B x C
        :return: T x B x C
        """
        _, B, _ = memory.shape
        m = self.proj_in(memory)
        output = self.prototype.unsqueeze(1).repeat((1, B, 1))
        attn_maps = {}
        for idx, mod in enumerate(self.layers):
            # Todo: Whether the first cross attention should use residue connection
            if idx == 0:
                cross_res = False
            else:
                cross_res = True
            output, attn_map = mod(output, m, cross_res=cross_res, return_attn=return_attn, memory_key_padding_mask=padding_mask)
            attn_maps[idx] = attn_map 
        return F.normalize(output, dim=-1), attn_maps

# dim_in, n_layers, n_emb, d_model, nhead = 192, 2, 32, 192, 2
# net = StyleBankExtractor(dim_in, n_layers, n_emb, d_model, nhead)
# input = torch.randn((136,16,192))
# padding = torch.zeros((15,16,192))
# input = torch.cat([input,padding],dim=0)
# print(input.shape)
# padding_mask = (input.transpose(0,1).sum(dim=-1)).eq(0).data
# out,_ = net(input,padding_mask=padding_mask)
# print(out.shape)