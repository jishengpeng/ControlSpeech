import torch
import torch.nn.functional as F
from typing import Sequence
import numpy as np

def generate_time_mask(
    spec: torch.Tensor,
    # max_length: int = 100,
    ratio: 0.1,
    num_mask: int = 1,
    replace_with_zero: bool = True,
):
    """Apply mask along the specified direction.
    Args:
        spec: (Length, Freq)
        spec_lengths: (Length): Not using lengths in this implementation
        mask_width_range: Select the width randomly between this range
    """

    org_size = spec.size()

    # D = Length
    D = spec.shape[0]
    # # mask_length: (num_mask)
    # if int(D*0.10) < int(D*0.14):
    #     mask_width_range = [int(D*0.10), int(D*0.14)]
    # else:
    #     mask_width_range = [int(D*0.10), int(D*0.14)+1]
    # mask_length = torch.randint(
    #     mask_width_range[0],
    #     mask_width_range[1],
    #     (num_mask,1),
    #     device=spec.device,
    # )
    mask_length = int(D*ratio)

    # mask_pos: (num_mask)
    mask_pos = torch.randint(
        0, max(1, D - mask_length), (num_mask,1), device=spec.device
    )

    # aran: (1, D)
    aran = torch.arange(D, device=spec.device)[None, :]
    # spec_mask: (num_mask, D)
    spec_mask = (mask_pos <= aran) * (aran < (mask_pos + mask_length))
    # Multiply masks: (num_mask, D) -> (D)
    spec_mask = spec_mask.any(dim=0).float()
    return spec_mask

def generate_alignment_aware_time_mask(
    spec: torch.Tensor,
    mel2ph,
    # max_length: int = 100,
    ratio: 0.1,
    num_mask: int = 1,
    replace_with_zero: bool = True,
):
    # obtain ph-level mask
    ph_mask = np.zeros((mel2ph.max()+1).item())
    ph_seq_idx = np.arange(0, mel2ph.max(), dtype=float) # start from 1 to match the mel2ph
    mask_ph_idx = np.random.choice(ph_seq_idx, size=int((mel2ph.max()+1)*ratio), replace=False).astype(np.uint8)
    ph_mask[mask_ph_idx] = 1.0
    ph_mask = torch.from_numpy(ph_mask).float()

    # obtain mel-level mask
    ph_mask = F.pad(ph_mask, [1, 0])
    mel2ph_ = mel2ph
    mel_mask = torch.gather(ph_mask, 0, mel2ph_)  # [B, T, H]

    return mel_mask

def random_crop_with_prob(mel, p):
    assert mel.dim()==2,'mel should be 2D data'
    ### 复制三份
    T,C = mel.size()
    mel = mel.unsqueeze(0).expand(3,-1,-1).reshape(-1,C)
    T,C = mel.size()
    ## 1. 0.5-1.0
    ## 2. 0.2-1.0
    if torch.rand(1).item()<p:
        start_index = torch.randint(0,int(T*0.8),(1,)).item()
        crop_length = torch.randint(int(T*0.2),T,(1,)).item()
        end_index = min(start_index+crop_length,T)
        cropped_mel = mel[start_index:end_index,:]
        return cropped_mel
    else:
        return mel