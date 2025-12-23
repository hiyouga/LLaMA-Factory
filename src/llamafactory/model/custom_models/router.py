import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

import torch.distributed as dist

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


from .kernels.RouterLogits import _RouterLogits

RouterLogits = _RouterLogits.apply
class GroupedRouter(nn.Module):

        """
        this model return a B,N,H,N attn map
        """

