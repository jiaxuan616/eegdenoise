"""EEGDNet / DeT model for EEG denoising."""

import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        hidden_dim = 2 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=1, dropout=0.1):
        super().__init__()
        dim_head = dim
        inner_dim = dim_head * heads
        project_out = not (heads == 1)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
            qkv,
        )

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dropout=dropout),
                        FeedForward(dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = self.norm(attn(x) + x)
            x = self.norm(ff(x) + x)
        return x


class EEGDNet(nn.Module):
    """
    EEGDNet official DeT-style model adapted to this repository.

    Input:
        x: [B, L], [B, 1, L], or [B, L, 1]
    Output:
        out: [B, L]
    """

    def __init__(
        self,
        datanum,
        patch_len=None,
        depth=6,
        heads=1,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super().__init__()

        seq_len = datanum

        if patch_len is None:
            # 默认切成 8 个 patch，和官方代码常见 512/64 的形式一致
            patch_len = seq_len // 8

        assert seq_len % patch_len == 0, (
            f"seq_len={seq_len} must be divisible by patch_len={patch_len}"
        )

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len

        dim = patch_len

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b (p1 p2) -> b p1 p2",
                p1=self.num_patches,
                p2=patch_len,
            ),
            nn.Linear(patch_len, dim),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
        )

        self.to_seq_embedding = nn.Sequential(
            nn.Linear(dim, patch_len),
            Rearrange(
                "b p1 p2 -> b (p1 p2)",
                p1=self.num_patches,
                p2=patch_len,
            ),
        )

    def forward(self, x):
        # 兼容你当前 Complex_CNN 的输入习惯
        if x.dim() == 3:
            if x.shape[1] == 1:
                x = x.squeeze(1)          # [B, 1, L] -> [B, L]
            elif x.shape[-1] == 1:
                x = x.squeeze(-1)         # [B, L, 1] -> [B, L]
            else:
                raise ValueError(f"Unsupported input shape: {x.shape}")

        if x.dim() != 2:
            raise ValueError(f"Expected input shape [B, L], got {x.shape}")

        if x.shape[-1] != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, got {x.shape[-1]}"
            )

        x = self.to_patch_embedding(x)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.to_seq_embedding(x)

        return x
