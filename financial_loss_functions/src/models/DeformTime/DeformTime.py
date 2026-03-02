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
            seq_len: int,
            e_layers: int,
            d_layers: int,
            d_model: int,
            attention_heads: int,
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
                CrossDeformAttn(seq_len=seq_len, 
                                d_model=self.d_model, 
                                n_heads=attention_heads, 
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
        """
        Variables
            • x_enc
                type: tensor of numbers
                usage: used to store the input feature window that is passed from the
                       trainer for DeformTime encoding
            • x_mark_enc
                type: tensor of numbers or empty value
                usage: reserved encoder marker input that is accepted for interface
                       compatibility but is not used in this runtime path
            • x_dec
                type: tensor of numbers or empty value
                usage: reserved decoder input kept for compatibility with the
                       surrounding model interface
            • x_mark_dec
                type: tensor of numbers or empty value
                usage: reserved decoder marker input kept for compatibility with the
                       surrounding model interface
            • mean_enc
                type: tensor of numbers
                usage: used to store the per-window mean so the input can be
                       stationarized before attention is applied
            • std_enc
                type: tensor of numbers
                usage: used to store the per-window standard deviation so the input
                       scaling remains numerically stable
            • enc_out
                type: tensor of numbers
                usage: used to store the encoded deformable-attention representation
                       produced by the encoder
            • h0
                type: tensor of numbers
                usage: used to store the initial hidden state passed into the GRU
            • out
                type: tensor of numbers
                usage: used to store the GRU sequence output before collapsing it to
                       a single allocation context
            • context
                type: tensor of numbers
                usage: used to store the last GRU state which becomes the compact
                       portfolio-allocation context for the final head

        forecast now extracts a portfolio-allocation context from the encoded sequence
        instead of reconstructing the original feature space. @author: Atharva Vaidya
        """
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
        """
        Variables
            • x_enc
                type: tensor of numbers
                usage: used to store the encoded input window passed from the trainer
                       for a portfolio-weight prediction
            • x_mark_enc
                type: tensor of numbers or empty value
                usage: reserved encoder marker input that is kept for interface
                       compatibility
            • x_dec
                type: tensor of numbers or empty value
                usage: reserved decoder input that is kept for interface compatibility
            • x_mark_dec
                type: tensor of numbers or empty value
                usage: reserved decoder marker input that is kept for interface
                       compatibility
            • mask
                type: tensor of numbers or empty value
                usage: reserved mask input accepted for compatibility with the model
                       interface
            • logits
                type: tensor of numbers
                usage: used to store the raw portfolio scores returned from forecast
                       before normalization

        forward normalizes the DeformTime logits into portfolio weights expected by
        the training and loss pipeline. @author: Atharva Vaidya
        """
        logits = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # Normalize the logits so the downstream losses receive portfolio weights.
        return torch.softmax(logits, dim=-1)

