import torch
from torch import nn
from src.models.registry import NNModelLibrary

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


@NNModelLibrary.register(category='transformer')
class DeformTime(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_stocks: int,
            max_seq_len: int,
            e_layers: int,
            d_layers: int,
            d_model: int,
            nheads: int,
            kernel_size: int,
            dropout: float,
            n_reshape: int,
            patch_len: int,
            stride: int
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.num_stocks = num_stocks
        
        self.d_layers = d_layers
        self.d_model = d_model

        # Embedding
        if self.input_size == 1:
            self.enc_value_embedding = Deform_Temporal_Embedding(self.input_size, self.d_model, freq='d')
        else:
            self.s_group = 4
            assert self.d_model % self.s_group == 0
            # Embedding local patches
            self.pad_in_len = ceil(1.0 * self.input_size / self.s_group) * self.s_group
            self.enc_value_embedding = Local_Temporal_Embedding(self.pad_in_len//self.s_group, self.d_model, self.pad_in_len-self.input_size, self.s_group)

        self.pre_norm = nn.LayerNorm(self.d_model)
        # Encoder
        n_days = [1,n_reshape,n_reshape]
        assert len(n_days) > e_layers-1
        drop_path_rate=dropout
        dpr = [x.item() for x in torch.linspace(drop_path_rate, drop_path_rate, e_layers)]
        self.encoder = Encoder(
            [
                CrossDeformAttn(seq_len=max_seq_len, 
                                d_model=self.d_model, 
                                n_heads=nheads, 
                                dropout=dropout, 
                                droprate=dpr[l], 
                                n_days=n_days[l], 
                                window_size=kernel_size, 
                                patch_len=patch_len, 
                                stride=stride) for l in range(e_layers)
            ],
            norm_layer=Layernorm(self.d_model)
        )

        # GRU layers
        self.gru = torch.nn.GRU(
            self.d_model, self.d_model, self.d_layers, batch_first=True, dropout=dropout
        )

        # @author: Atharva Vaidya - This head helps in converting the encoded DeformTime context into portfolio logits expected by the loss pipeline.
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.num_stocks)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        assert x_enc.shape[-1] == self.input_size

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
        # Extract the final GRU state so one context vector represents the allocation decision.
        context = out[:, -1, :]
        # Apply the portfolio head to convert the context vector into allocation logits.
        return self.fc(context)
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        logits = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # Normalize the logits so the downstream losses receive portfolio weights.
        return torch.softmax(logits, dim=-1)

