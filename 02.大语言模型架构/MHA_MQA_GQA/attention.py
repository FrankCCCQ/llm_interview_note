import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dims, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dims = dims
        self.head_dim = dims // num_heads
        self.wq = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.wk = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.wv = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.wo = nn.Linear(self.hidden_dims, self.hidden_dims)
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        batch_size = x.size()[0]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = self._split_head(q)
        k = self._split_head(k)
        v = self._split_head(v)

        scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(self.head_dim)
        if attention_mask is not None:
            scores += attention_mask
        attention_prob = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_prob, v)
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        if self.dropout:
            output = self.dropout(output)
        output = self.wo(output)
        return output


    def _split_head(self, x):
        batch_size, seq_len = x.size()[:2]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        return x
        

    
class MultiQueryAttention(nn.Module):
    def __init__(self, dims, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dims = dims
        self.head_dims = dims // num_heads
        self.wq = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.wk = nn.Linear(self.hidden_dims, self.head_dims)
        self.wv = nn.Linear(self.hidden_dims, self.head_dims)
        self.wo = nn.Linear(self.hidden_dims, self.hidden_dims)
        if dropout:
            self.dropout = nn.Dropout(dropout)


    def forward(self, x, attention_mask=None):
        batch_size = x.size()[0]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = self._split_head(q)
        k = self._split_head(k,1)
        v = self._split_head(v,1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(self.head_dims)
        if attention_mask is not None:
            scores += attention_mask
        attention_prob = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_prob, v)
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.head_dims)
        if self.dropout:
            output = self.dropout(output)
        output = self.wo(output)
        return output
        
        
    def _split_head(self, x, num_heads=None):
        batch_size, seq_len = x.size()[:2]
        if num_heads is None:
            x = x.view(batch_size, seq_len, self.num_heads, self.head_dims).transpose(1,2)
        else:
            x = x.view(batch_size, seq_len, num_heads, self.head_dims).transpose(1,2)
        return x

class GroupQueryAttention(nn.Module):
    def __init__(self, dims, num_heads, num_groups, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups    
        self.hidden_dims = dims
        self.head_dims = dims // num_heads
        self.wq = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.wk = nn.Linear(self.hidden_dims, self.head_dims * self.num_groups)
        self.wv = nn.Linear(self.hidden_dims, self.head_dims * self.num_groups)
        self.wo = nn.Linear(self.hidden_dims, self.hidden_dims)
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        batch_size = x.size()[0]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = self._split_head(q)
        k = self._split_head(k, self.num_groups)
        v = self._split_head(v, self.num_groups)
        scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(self.head_dims)
        if attention_mask is not None:
            scores += attention_mask
        attention_prob = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_prob, v)
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.head_dims)
        if self.dropout:
            output = self.dropout(output)
        output = self.wo(output)
        return output
    
    def _split_head(self, x, num_groups=None):
        batch_size, seq_len = x.size()[:2]
        if num_groups is None:
            x = x.view(batch_size, seq_len, self.num_heads, self.head_dims).transpose(1,2)
        else:
            x = x.view(batch_size, seq_len, self.num_groups, self.head_dims).transpose(1,2)
            x = x[:, : ,None, :, :].expand(batch_size, num_groups, self.num_heads // num_groups, seq_len, self.head_dims).reshape(batch_size, self.num_heads // num_groups * num_groups , seq_len, self.head_dims)
        return x
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) 
        
    def forward(self, x):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)
        

class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b =  nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a =  nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b =  nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo =  nn.Linear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale


    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                    torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x