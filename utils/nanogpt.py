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

        self.resid_dropout = nn.Dropout(cfg.dropout)

        # attention heads
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

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
            dropout_p=self.dropout,
            is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

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
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
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

        if cfg.position_encoding == 'sinusoidal':
            wpe_encoding = SinusoidalEmbedding(cfg.n_embd, cfg.context_size)
            self.te_scale = math.sqrt(self.cfg.n_embd)
        elif cfg.position_encoding == 'learnable':
            wpe_encoding = nn.Embedding(cfg.context_size, cfg.n_embd)
            self.te_scale = 1
        else:
            raise NotImplementedError


        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = wpe_encoding,
            drop = nn.Dropout(cfg.dropout),
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

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        # A100 GPU bfloat16 peak flops is 312 TFLOPS
        flops_promised = {'a100': 312e12, 'a10': 125e12, '4090': 82.5} 

        N = self.get_num_params()
        cfg = self.cfg
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.context_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        mfu = flops_achieved / flops_promised['a10']
        return mfu

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        # Exclude sinusoidal embedding
        n_params = sum(p.numel() for p in self.parameters())
        if hasattr(self.transformer.wpe, 'weight') and  non_embedding:
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
        tok_emb = self.transformer.wte(idx) * self.te_scale
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos) 

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.LM_head(x)
        return logits 


class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len=5000):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_seq_len, dim)

        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        freq = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))


        pe[:, 0::2] = torch.sin(position * freq)
        pe[:, 1::2] = torch.cos(position * freq)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    @torch.no_grad
    def forward(self, x):
        return self.pe[:, :x.size(0)]


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self, dim, max_seq_len, base=10000, precision=torch.half, save_inv_freqs=False
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=save_inv_freqs)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision
        self.max_seq_len = max_seq_len
        self.base = base
        self.dim = dim

        # precompute cos_cached, sin_cached in fp32
        cos_cached, sin_cached, inv_freq = self._prepare_cache(
            max_seq_len, precision, base
        )

        self.register_buffer("inv_freq", inv_freq, persistent=save_inv_freqs)
        self.cos_cached = cos_cached
        self.sin_cached = sin_cached

    def _prepare_cache(self, seq_len, precision, base):
        # precompute cos_cached, sin_cached in fp32
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        t = torch.arange(seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos_cached = emb.cos()[:, None, None, :]
        sin_cached = emb.sin()[:, None, None, :]

        return (
            cos_cached.to(precision),
            sin_cached.to(precision),
            inv_freq.to(precision),
        )

    def forward(self, x, seq_dim=0, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]

        assert seq_len <= self.max_seq_len

        if seq_len != self.max_seq_len:
            # y, z, _ = self._prepare_cache(seq_len, self.precision, self.base)
            return (
                self.cos_cached[:seq_len, ...].to(x.device),
                self.sin_cached[:seq_len, ...].to(x.device),
            )
        else:
            return self.cos_cached.to(x.device), self.sin_cached.to(x.device)


# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_torch(
    q, k, cos, sin, offset: int = 0
):  # jitting fails with bf16
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
