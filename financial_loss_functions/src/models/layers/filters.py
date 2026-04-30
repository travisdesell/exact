import torch 
from torch import nn

class RobustNormalization(nn.Module):
    """
    Robust feature normalization using median and interquartile range (IQR).

    Applies (x - median) / (IQR + eps). Optionally adds learnable affine transformation.
    """
    def __init__(self, feature_dim, median, iqr, eps: float = 1e-8):
        """Initializes robust normalization with fixed median and IQR.

        Args:
            feature_dim (int): Number of features (last dimension of input).
            median (torch.Tensor or list[float]): Median values for each feature (length feature_dim).
            iqr (torch.Tensor or list[float]): Interquartile range for each feature (length feature_dim).
            eps (float, optional): Small constant to avoid division by zero. Default = 1e-8.
        """
        super().__init__()
        self.register_buffer('median', torch.tensor(median, dtype=torch.float32))
        self.register_buffer('iqr', torch.tensor(iqr, dtype=torch.float32))

        self.eps = eps
        # Optional affine: scale and shift (learnable)
        self.affine = False  # set True if you want learnable scale/shift
        if self.affine:
            self.weight = nn.Parameter(torch.ones(feature_dim))
            self.bias = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x):
        """Normalizes the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F) or (B, F).

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        # x shape: (B, T, F) or (B, F)
        x = (x - self.median) / (self.iqr + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x

class FFTSpectralFilter(nn.Module):
    """
    Learnable frequency domain filter using FFT.

    Applies a complex filter (real + imaginary parts) to the Fourier transform of the input,
    then returns the time domain signal plus a residual connection.
    """
    def __init__(self, seq_len, hidden_size):
        """
        Initialises the FFT spectral filter.

        Args:
            seq_len (int): Length of the input sequence (T).
            hidden_size (int): Hidden dimension (H).
        """
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
        """
        Applies the learnable spectral filter.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, H).

        Returns:
            torch.Tensor: Filtered output of shape (B, T, H).
        """
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
    """
    Single-level Haar wavelet denoising layer with learnable soft-threshold.

    Applies Haar decomposition (approximation and detail), soft-thresholds the 
    detail coefficients, and reconstructs the signal. The threshold is learnable.
    """
    def __init__(self):
        """Initialises the wavelet denoising layer with a learnable threshold."""
        super().__init__()
        # Learnable threshold for denoising
        self.threshold = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        """
        Denoises the input using Haar wavelet transform.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, H). Sequence length T must be even.

        Returns:
            torch.Tensor: Denoised tensor of the same shape (B, T, H).
        """
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