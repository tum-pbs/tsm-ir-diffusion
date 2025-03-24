# Based on the code from the paper: "Benchmarking Autoregressive Conditional Diffusion Models for Turbulent Flow Simulation"

# @misc{kohl2024benchmarking,
#       title={Benchmarking Autoregressive Conditional Diffusion Models for Turbulent Flow Simulation}, 
#       author={Georg Kohl and Li-Wei Chen and Nils Thuerey},
#       year={2024},
#       eprint={2309.01745},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG}
# }

from utils.header import *


def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)    


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, mult=2, dropout=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = mult * in_channels
        num_of_groups = 1

        self.time_emb = nn.Sequential(
            nn.GELU(), nn.Linear(time_emb_dim, in_channels)
            if time_emb_dim is not None else None
        )

        self.pre_conv = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels)

        self.double_conv = nn.Sequential(
            nn.GroupNorm(num_of_groups, in_channels),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.GELU(),
            nn.Dropout2d(p=dropout),
            nn.GroupNorm(num_of_groups, mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros'),
        )

        
        if in_channels != out_channels:
            self.in_out_comp = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.in_out_comp = nn.Identity()

    def forward(self, x, t=None):
        h = self.pre_conv(x)
        
        if self.time_emb is not None and t is not None:
            emb = self.time_emb(t)
            emb = rearrange(emb, "b c -> b c 1 1")
            h += emb

        h = self.double_conv(h)
        
        return h + self.in_out_comp(x)


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, t):
        return self.mlp(t)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)