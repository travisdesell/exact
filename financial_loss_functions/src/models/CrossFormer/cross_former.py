import torch
import torch.nn as nn
from einops import repeat

from src.models.CrossFormer.cross_encoder import Encoder
from src.models.CrossFormer.cross_decoder import Decoder
from src.models.CrossFormer.cross_embed import DSW_embedding

from math import ceil

from src.models.registry import NNModelLibrary

@NNModelLibrary.register(category='transformer', name='CrossFormer')
class CrossFormerWrapper(nn.Module):
    """
    Wrapper that adapts CrossFormer for portfolio weight generation.
    It uses CrossFormer as a feature extractor (predicting next-step features)
    and then maps those features to asset weights via a linear layer.
    """
    def __init__(
        self,
        input_size: int,           # number of features (251)
        hidden_size: int,          # d_model in CrossFormer
        num_layers: int,           # e_layers
        num_stocks: int,           # number of assets (50)
        nheads: int,               # n_heads
        dropout: float,
        max_seq_len: int,          # in_len
        seg_len: int,              # segment length for DSW embedding
        win_size: int,             # segment merging factor for HED
        factor: int,               # number of routers in cross-dim attention
        expansion_factor: int, # determines d_ff = hidden_size * expansion_factor
        equal_prior: bool,  # initialize linear layer to near zero
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stocks = num_stocks
        self.equal_prior = equal_prior

        # Instantiate CrossFormer with out_len=1 (predict one future step)
        self.crossformer = Crossformer(
            data_dim=input_size,
            in_len=max_seq_len,
            out_len=1,                     # we only need a single representation
            seg_len=seg_len,
            win_size=win_size,
            factor=factor,
            d_model=hidden_size,
            d_ff=hidden_size * expansion_factor,
            n_heads=nheads,
            e_layers=num_layers,
            dropout=dropout,
            baseline=False
        )

        # Linear layer to map CrossFormer's output features to portfolio weights
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, num_stocks)
        
        if equal_prior:
            # Initialize weights near zero so initial output is near uniform
            nn.init.uniform_(self.fc.weight, a=-1e-3, b=1e-3)
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_size)
        # CrossFormer forward returns (batch, out_len, data_dim)
        out = self.crossformer(x)           # (B, 1, input_size)
        out = out.squeeze(1)                 # (B, input_size)
        out = self.dropout(out)
        logits = self.fc(out)                # (B, num_stocks)
        return torch.softmax(logits, dim=-1)

class Crossformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, baseline = False, device=torch.device('cuda:0')):
        super(Crossformer, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size

        self.baseline = baseline

        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_out_len // seg_len), factor = factor)
        
    def forward(self, x_seq):
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)


        return base + predict_y[:, :self.out_len, :]