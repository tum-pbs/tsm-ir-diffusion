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

from functools import partial


class UNet_conditional(nn.Module):
    def __init__(self, c_in=4, c_out=2, time_dim=256, config=None):
        super().__init__()
        self.config = config
        self.sim_time = self.config['UNet']['sim_time']
        self.grid_size = self.config['general']['grid_size']
        if type(self.grid_size) == str:
            self.grid_size = self.config['general']['grid_size'].split(',')
            self.grid_size = [int(i) for i in self.grid_size]
        else:
            self.grid_size = [self.grid_size, self.grid_size]
        self.time_dim = self.config['UNet']['time_dim']
        self.model_capacity = self.config['UNet']['model_capacity']
        self.base_channels = self.config['UNet']['base_channels']
        self.is_diffusion = self.config['general']['is_diffusion']
        self.airfoil_case = self.config['dataset']['name'] == 'airfoil'

        self.channels = c_in
        dim = self.base_channels
        dim_mults = np.array(config['UNet']['level_multipliers'].split(',')).astype(int) 
        dim_mults = np.append(np.array([1]),dim_mults)

        init_dim = dim // 3 * 2

        self.init_conv = nn.Conv2d(self.channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        

        block_klass = partial(DoubleConv, mult=1)

        if self.is_diffusion:
            time_dim = dim * 4
            self.time_mlp1 = TimeMLP(dim,time_dim)
        else:
            time_dim = None
            self.time_mlp1 = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = c_out
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )



    def forward(self, x, t, initial_cond, t_labels, Res):
        if self.airfoil_case:
            x = torch.concat([x, initial_cond], dim=1)
        else:
            if self.is_diffusion:
                assert x is not None and t is not None, "Either x or t inputs were None for the DDPM"
            else:
                assert x is None and t is None, "Both x and t inputs must be None for the UNet"

            assert Res is not None
            assert t_labels is not None
            
            shape = initial_cond.shape[-2:]
            Res = Res.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[0], shape[1])
            t_labels = (t_labels - 5) / 3.1622776601683795
            t_labels = t_labels.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[0], shape[1])

            if self.is_diffusion:
                x = [x, initial_cond, t_labels, Res]
            else:
                x = [initial_cond, t_labels, Res]
            x = torch.concat(x, dim=1)

        x = self.init_conv(x)


        t = self.time_mlp1(t) if self.time_mlp1 is not None else None
        
        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        
        return self.final_conv(x)    