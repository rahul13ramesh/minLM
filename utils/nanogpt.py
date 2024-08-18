"""
Code from nanoGPT
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False
    Set bias=False to make training faster
    """
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    One operation of multi-head self attention (MHSA).
    Calculate Query, Key, Value and pass through MHSA
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)

        # output projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        # attention heads
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x):
        """
        Compute self attention output to be added to residual stream
        """
        B, T, C = x.size()

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=0,
            is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y

    def get_attention(self, Q, K, B, T, C):
        bias = torch.tril(torch.ones(C, C)).view(1, 1, C, C)
        att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))
        att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        return att


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc    = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    One self-attention block
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        """
        Add to residual stream after self-attention and MLP.
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class nanoGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.context_size, cfg.n_embd),
            h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.LM_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight typing
        self.transformer.wte.weight = self.LM_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * cfg.n_layer))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()

        # Compute position/token embeddings
        tok_emb = self.transformer.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)

        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.LM_head(x)
        return logits 
