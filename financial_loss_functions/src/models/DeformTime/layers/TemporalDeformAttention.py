import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_
from src.models.DeformTime.layers.MLP import MLP
from src.models.DeformTime.utils.functions import num_patches


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    From: https://github.com/huggingface/pytorch-image-models

    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    DropPath is dropping an entire sample from the batch while Dropout is dropping random values
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerScale(nn.Module):
    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1, 1))
        else:
            return x * self.weight.view(-1, 1, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return x

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c l -> b l c')
        x = self.norm(x)
        return rearrange(x, 'b l c -> b c l')


class LayerNormProxy2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DeformAtten1D(nn.Module):
    '''
        max_offset (int): The maximum magnitude of the offset residue. Default: 14.
    '''
    def __init__(self, seq_len, d_model, n_heads, dropout, kernel=5, n_groups=4, no_off=False, rpb=True) -> None:
        super().__init__()
        self.offset_range_factor = kernel
        self.no_off = no_off
        self.seq_len = seq_len
        self.d_model = d_model 
        self.n_groups = n_groups
        self.n_group_channels = self.d_model // self.n_groups
        self.n_heads = n_heads
        self.n_head_channels = self.d_model // self.n_heads
        self.n_group_heads = self.n_heads // self.n_groups
        self.scale = self.n_head_channels ** -0.5
        self.rpb = rpb

        self.proj_q = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Linear(self.d_model, self.d_model)
        kernel_size = kernel
        self.stride = 1
        pad_size = kernel_size // 2 if kernel_size != self.stride else 0
        self.proj_offset = nn.Sequential(
            nn.Conv1d(self.n_group_channels, self.n_group_channels, kernel_size=kernel_size, stride=self.stride, padding=pad_size),
            nn.Conv1d(self.n_group_channels, 1, kernel_size=1, stride=self.stride, padding=pad_size),
        )

        self.scale_factor = self.d_model ** -0.5  # 1/np.sqrt(dim)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.d_model, self.seq_len))
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        dtype, device = x.dtype, x.device
        x = x.permute(0,2,1) # B, C, L

        q = self.proj_q(x) # B, C, L

        group = lambda t: rearrange(t, 'b (g d) n -> (b g) d n', g = self.n_groups)

        grouped_queries = group(q)

        offset = self.proj_offset(grouped_queries) # B * g 1 Lg
        offset = rearrange(offset, 'b 1 n -> b n')

        def grid_sample_1d(feats, grid, *args, **kwargs):
            # does 1d grid sample by reshaping it to 2d
            grid = rearrange(grid, '... -> ... 1 1')
            grid = F.pad(grid, (1, 0), value = 0.)
            feats = rearrange(feats, '... -> ... 1')
            # the backward of F.grid_sample is non-deterministic
            # See for details: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            out = F.grid_sample(feats, grid, **kwargs) 
            return rearrange(out, '... 1 -> ...')
        
        def normalize_grid(arange, dim = 1, out_dim = -1):
            # normalizes 1d sequence to range of -1 to 1
            n = arange.shape[-1]
            return 2.0 * arange / max(n - 1, 1) - 1.0

        if self.offset_range_factor >= 0 and not self.no_off:
            offset = offset.tanh().mul(self.offset_range_factor)

        if self.no_off:
            x_sampled = F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride)
        else:
            grid = torch.arange(offset.shape[-1], device = device)
            vgrid = grid + offset
            vgrid_scaled = normalize_grid(vgrid)

            x_sampled = grid_sample_1d(
                group(x),
                vgrid_scaled,
            mode = 'bilinear', padding_mode = 'zeros', align_corners = False)[:,:,:L]
            
        if not self.no_off:
            x_sampled = rearrange(x_sampled,'(b g) d n -> b (g d) n', g = self.n_groups)
        q = q.reshape(B * self.n_heads, self.n_head_channels, L)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, L)
        if self.rpb:
            v = self.proj_v(x_sampled)
            v = (v + self.relative_position_bias_table).reshape(B * self.n_heads, self.n_head_channels, L)
        else:
            v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, L)

        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1) # softmax: attention[0,0,:].sum() = 1

        out = torch.einsum('b i j , b j d -> b i d', attention, v) 
        
        return self.proj_out(rearrange(out, '(b g) l c -> b c (g l)', b=B))


class DeformAtten2D(nn.Module):
    '''
        max_offset (int): The maximum magnitude of the offset residue. Default: 14.
    '''
    def __init__(self, seq_len, d_model, n_heads, dropout, kernel=5, n_groups=4, no_off=False, rpb=True) -> None:
        super().__init__()
        self.offset_range_factor = kernel
        self.no_off = no_off
        self.f_sample = False
        self.seq_len = seq_len
        self.d_model = d_model # (512)
        self.n_groups = n_groups
        self.n_group_channels = self.d_model // self.n_groups
        self.n_heads = n_heads
        self.n_head_channels = self.d_model // self.n_heads
        self.n_group_heads = self.n_heads // self.n_groups
        self.scale = self.n_head_channels ** -0.5
        self.rpb = rpb

        self.proj_q = nn.Conv2d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Linear(self.d_model, self.d_model)
        kernel_size = kernel
        self.stride = 1
        pad_size = kernel_size // 2 if kernel_size != self.stride else 0
        self.proj_offset = nn.Sequential( 
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kernel_size=kernel_size, stride=self.stride, padding=pad_size),
            nn.Conv2d(self.n_group_channels, 2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.scale_factor = self.d_model ** -0.5  # 1/np.sqrt(dim)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.d_model, self.seq_len, 1))
            trunc_normal_(self.relative_position_bias_table, std=.02)


    def forward(self, x, mask=None):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2) # B, C, H, W

        q = self.proj_q(x) # B, 1, H, W

        offset = self.proj_offset(q) # B, 2, H, W

        if self.offset_range_factor >= 0 and not self.no_off:
            offset = offset.tanh().mul(self.offset_range_factor)

        def create_grid_like(t, dim = 0):
            h, w, device = *t.shape[-2:], t.device

            grid = torch.stack(torch.meshgrid(
                torch.arange(w, device = device),
                torch.arange(h, device = device),
            indexing = 'xy'), dim = dim)

            grid.requires_grad = False
            grid = grid.type_as(t)
            return grid
        
        def normalize_grid(grid, dim = 1, out_dim = -1):
            # normalizes a grid to range from -1 to 1
            h, w = grid.shape[-2:]
            grid_h, grid_w = grid.unbind(dim = dim)

            grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
            grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

            return torch.stack((grid_h, grid_w), dim = out_dim)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        else:
            grid =create_grid_like(offset)
            vgrid = grid + offset
            vgrid_scaled = normalize_grid(vgrid)
            # the backward of F.grid_sample is non-deterministic
            x_sampled = F.grid_sample(
                x,
                vgrid_scaled,
            mode = 'bilinear', padding_mode = 'zeros', align_corners = False)[:,:,:H,:W]
              
        if not self.no_off:
            x_sampled = rearrange(x_sampled, '(b g) c h w -> b (g c) h w', g=self.n_groups)
        q = q.reshape(B * self.n_heads, H, W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, H, W)
        if self.rpb:
            v = self.proj_v(x_sampled)
            v = (v + self.relative_position_bias_table).reshape(B * self.n_heads, H, W)
        else:
            v = self.proj_v(x_sampled).reshape(B * self.n_heads, H, W)

        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        out = torch.einsum('b i j , b j d -> b i d', attention, v)
        
        return self.proj_out(out.reshape(B, H, W, C))


class CrossDeformAttn(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, dropout, droprate, 
                 n_days=1, window_size=4, patch_len=7, stride=3, no_off=False) -> None:
        super().__init__()
        self.n_days = n_days
        self.seq_len = seq_len
        # 1d size: B*n_days, subseq_len, C
        # 2d size: B*num_patches, 1, patch_len, C
        self.subseq_len = seq_len // n_days + (1 if seq_len % n_days != 0 else 0)
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = num_patches(self.seq_len, self.patch_len, self.stride)

        self.layer_norm = LayerNorm(d_model)

        # 1D
        self.ff1 = nn.Linear(d_model, d_model, bias=True)
        self.ff2 = nn.Linear(self.subseq_len, self.subseq_len, bias=True)
        # Deform attention
        self.deform_attn = DeformAtten1D(self.subseq_len, d_model, n_heads, dropout, kernel=window_size, no_off=no_off) 
        self.attn_layers1d = nn.ModuleList([self.deform_attn])

        self.mlps1d = nn.ModuleList(
            [ 
                MLP([d_model, d_model], final_relu=True, drop_out=0.0) for _ in range(len(self.attn_layers1d))
            ]
        )
        self.drop_path1d = nn.ModuleList(
            [
                DropPath(droprate) if droprate > 0.0 else nn.Identity() for _ in range(len(self.attn_layers1d))
            ]
        )
        #######################################
        # 2D
        d_route = 1
        self.conv_in = nn.Conv2d(1, d_route, kernel_size=1, bias=True)
        self.conv_out = nn.Conv2d(d_route, 1, kernel_size=1, bias=True)
        self.deform_attn2d = DeformAtten2D(self.patch_len, d_route, n_heads=1, dropout=dropout, kernel=window_size, n_groups=1, no_off=no_off)
        self.write_out = nn.Linear(self.num_patches*self.patch_len, self.seq_len)

        self.attn_layers2d = nn.ModuleList([self.deform_attn2d])

        self.mlps2d = nn.ModuleList(
            [ 
                MLP([d_model, d_model], final_relu=True, drop_out=0.0) for _ in range(len(self.attn_layers2d))
            ]
        )
        self.drop_path2d = nn.ModuleList(
            [
                DropPath(droprate) if droprate > 0.0 else nn.Identity() for _ in range(len(self.attn_layers2d))
            ]
        )

        self.fc = nn.Linear(2*d_model, d_model)
        
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        n_day = self.n_days 
        B, L, C = x.shape

        x = self.layer_norm(x)

        padding_len = (n_day - (L % n_day)) % n_day
        x_padded = torch.cat((x, x[:, [0], :].expand(-1, padding_len, -1)), dim=1)
        x_1d = rearrange(x_padded, 'b (seg_num ts_d) d_model -> (b ts_d) seg_num d_model', ts_d=n_day) 
        # attn on 1D
        for d, attn_layer in enumerate(self.attn_layers1d):
            x0 = x_1d
            x_1d = attn_layer(x_1d)
            x_1d = self.drop_path1d[d](x_1d) + x0
            x0 = x_1d
            x_1d = self.mlps1d[d](self.layer_norm(x_1d))
            x_1d = self.drop_path1d[d](x_1d) + x0
        x_1d = rearrange(x_1d, '(b ts_d) seg_num d_model -> b (seg_num ts_d) d_model', ts_d=n_day)[:,:L,:]

        # Patch attn on 2D
        x_unfold = x.unfold(dimension=-2, size=self.patch_len, step=self.stride)
        x_2d = rearrange(x_unfold, 'b n c l -> (b n) l c').unsqueeze(-3)
        x_2d = rearrange(x_2d, 'b c h w -> b h w c')
        for d, attn_layer in enumerate(self.attn_layers2d):
            x0 = x_2d
            x_2d = attn_layer(x_2d)
            x_2d = self.drop_path2d[d](x_2d) + x0
            x0 = x_2d
            x_2d = self.mlps2d[d](self.layer_norm(x_2d.permute(0,1,3,2))).permute(0,1,3,2)
            x_2d = self.drop_path2d[d](x_2d) + x0
        x_2d = rearrange(x_2d, 'b h w c -> b c h w')
        x_2d = rearrange(x_2d, '(b n) 1 l c -> b (n l) c', b=B)
        x_2d = self.write_out(x_2d.permute(0,2,1)).permute(0,2,1)

        x = torch.concat([x_1d, x_2d], dim=-1)
        x = self.fc(x)

        return x, None
    
