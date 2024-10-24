'''This Implementation is an adapted version for vision tasks from the paper Differential Transformer https://arxiv.org/abs/2410.05258'''

import math
import torch
import torch.nn.functional as F
from torch import nn

# Assuming these are available in your environment
# from .kernel.rotary import apply_rotary_emb
# from flash_attn import flash_attn_func
try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from torch.nn import LayerNorm as RMSNorm  # Fallback to LayerNorm if RMSNorm is unavailable

def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads // 2  # Divided by 2 for Q1/Q2
        assert (
            self.head_dim * num_heads * 2 == embed_dim
        ), "Embedding dimension must be divisible by num_heads * 2"

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Lambda parameters
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )

        self.dropout = nn.Dropout(dropout)
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)

    def forward(
        self,
        x,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape q, k, v for multi-head attention
        # q, k: [batch_size, seq_len, num_heads * 2, head_dim]
        q = q.view(bsz, tgt_len, self.num_heads * 2, self.head_dim)
        k = k.view(bsz, tgt_len, self.num_heads * 2, self.head_dim)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim * 2)

        # Transpose for attention computation
        # q, k: [batch_size, num_heads * 2, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        # v: [batch_size, num_heads, seq_len, head_dim * 2]
        v = v.transpose(1, 2)

        # Scale queries
        q = q * self.scaling

        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # [batch_size, num_heads * 2, seq_len, seq_len]
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float("-inf"))

        # Compute softmax over attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Compute lambda values
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1)).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2)).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Reshape attention weights to separate the two components
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, tgt_len)
        
        # Compute difference of attention weights
        attn_weights_diff = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]  # [batch_size, num_heads, tgt_len, tgt_len]

        # Apply dropout to attention weights
        attn_weights_diff = self.dropout(attn_weights_diff)

        # Compute attention output
        attn_output = torch.matmul(attn_weights_diff, v)  # [batch_size, num_heads, tgt_len, head_dim * 2]
        
        attn_output = self.subln(attn_output)
        # Transpose and reshape for sub-layer normalization
        # [batch_size, tgt_len, num_heads, head_dim * 2]
         # Apply sub-layer normalization
        

          # [batch_size, tgt_len, embed_dim]

       
       

        # Multiply by (1 - lambda_init)
        attn_output = attn_output * (1 - self.lambda_init)
        attn_output = attn_output.transpose(1, 2).contiguous() 
        attn_output = attn_output.view(bsz, tgt_len, -1)
        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output
