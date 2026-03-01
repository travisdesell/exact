import torch
from torch import nn

from src.models.DeformTime.layers.TemporalDeformAttention import Encoder, CrossDeformAttn
from src.models.DeformTime.layers.Embed import Deform_Temporal_Embedding, Local_Temporal_Embedding
from math import ceil

class Layernorm(nn.Module):
    def __init__(self, dim):
        super(Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias

class Model(nn.Module):
    def __init__(self, configs) -> None:
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.d_model = configs.d_model
        self.f_dim = configs.enc_in
        self.c_out = configs.c_out
        self.dropout = configs.dropout
        self.kernel_size = configs.kernel

        # Embedding
        if configs.enc_in == 1:
            self.enc_value_embedding = Deform_Temporal_Embedding(self.f_dim, self.d_model, freq='d')
        else:
            self.s_group = 4
            assert self.d_model % self.s_group == 0
            # Embedding local patches
            self.pad_in_len = ceil(1.0 * configs.enc_in / self.s_group) * self.s_group
            self.enc_value_embedding = Local_Temporal_Embedding(self.pad_in_len//self.s_group, self.d_model, self.pad_in_len-configs.enc_in, self.s_group)

        self.pre_norm = nn.LayerNorm(configs.d_model)
        # Encoder
        n_days = [1,configs.n_reshape,configs.n_reshape]
        assert len(n_days) > self.e_layers-1
        drop_path_rate=configs.dropout
        dpr = [x.item() for x in torch.linspace(drop_path_rate, drop_path_rate, self.e_layers)]
        self.encoder = Encoder(
            [
                CrossDeformAttn(seq_len=configs.seq_len, 
                                d_model=configs.d_model, 
                                n_heads=configs.n_heads, 
                                dropout=configs.dropout, 
                                droprate=dpr[l], 
                                n_days=n_days[l], 
                                window_size=configs.kernel, 
                                patch_len=configs.patch_len, 
                                stride=configs.stride) for l in range(configs.e_layers)
            ],
            norm_layer=Layernorm(configs.d_model)
        )

        # GRU layers
        self.gru = torch.nn.GRU(
            self.d_model, self.d_model, self.d_layers, batch_first=True, dropout=configs.dropout
        )

        # MLP layer
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.pred_len)
        )

        # Projection layer
        self.projection = nn.Linear(self.d_model, self.f_dim)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        assert x_enc.shape[-1] == self.f_dim

        # Series Stationarization adopted from NSformer, optional
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x_enc = self.enc_value_embedding(x_enc)
        x_enc = self.pre_norm(x_enc)

        # Deformed attention
        enc_out, _ = self.encoder(x_enc) 

        # Decoder
        h0 = torch.zeros(self.d_layers, x_enc.size(0), self.d_model).requires_grad_().to(x_enc.device)
        out, _ = self.gru(enc_out, h0.detach())
        out = self.fc(out.permute(0,2,1)).permute(0,2,1)

        # Projection
        out = self.projection(out)
        out = out * std_enc + mean_enc

        return out
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]



