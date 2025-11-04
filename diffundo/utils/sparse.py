import torch
from torch import Tensor


def keep_topk_elements(tensor: Tensor, k: int) -> Tensor:
    _, topk_indices = torch.topk(tensor, k=k, dim=1)
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    return tensor * mask
