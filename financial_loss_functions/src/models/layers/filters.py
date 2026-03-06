import torch 
from torch import nn

class FFTSpectralFilter(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super().__init__()
        self.seq_len = seq_len
        self.n_freq = seq_len // 2 + 1
        
        # Store as two separate REAL parameters to avoid 'norm' errors
        # Initialize as 1.0 (Real) and 0.0 (Imag) so it starts as Identity
        self.filter_real = nn.Parameter(torch.ones(self.n_freq, hidden_size))
        self.filter_imag = nn.Parameter(torch.zeros(self.n_freq, hidden_size))
        
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Optional: Initialize with a Low-Pass prior (Dampen high frequencies)
        with torch.no_grad():
            lp_prior = torch.linspace(1.0, 0.1, steps=self.n_freq).view(-1, 1)
            self.filter_real.copy_(lp_prior)

    def forward(self, x):
        # x: (B, T, H)
        
        # 1. Transform to Frequency Domain
        x_fft = torch.fft.rfft(x, dim=1) # (B, n_freq, H) complex
        
        # 2. Reconstruct the complex filter from real/imag parts
        # This happens on-the-fly, so the optimizer only sees real weights
        complex_filter = torch.complex(self.filter_real, self.filter_imag)
        
        # 3. Apply filter
        # x_fft is (B, n_freq, H), filter is (n_freq, H)
        x_fft = x_fft * complex_filter.unsqueeze(0)
        
        # 4. Transform back to Time Domain
        x_filtered = torch.fft.irfft(x_fft, n=self.seq_len, dim=1)
        
        return x + self.alpha * x_filtered

class WaveletDenoiseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable threshold for denoising
        self.threshold = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        # Simple Haar Decomposition: (Average, Difference)
        # x: (B, T, H)
        even = x[:, 0::2, :]
        odd = x[:, 1::2, :]
        
        approx = (even + odd) / 2.0
        detail = (even - odd) / 2.0
        
        # Soft-thresholding (Denoising the 'Detail' part)
        # If noise is small, kill it. If large, dampen it.
        detail = torch.sign(detail) * torch.relu(torch.abs(detail) - self.threshold)
        
        # Reconstruction (Inverse Haar)
        out_even = approx + detail
        out_odd = approx - detail
        
        # Interleave back together
        res = torch.empty_like(x)
        res[:, 0::2, :] = out_even
        res[:, 1::2, :] = out_odd
        return res