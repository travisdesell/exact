from typing import List, Optional

import torch
import torch.nn as nn
from torch import clamp, tensor


# ---------------------------------------------------------------------------
# Original standalone loss functions (unchanged)
# ---------------------------------------------------------------------------

def raw_sharpe_loss(
        weights: tensor, returns: tensor, eps: float = 1e-8
    ):
    """
    @param weights torch.tensor (B, N)
    @param returns torch.tensor (B, T_out, N) -- raw returns

    @return batch average Sharpe Ratio. 
        Negative since NN has to maximize Sharpe Ratio but minimize loss
    """
    # portfolio returns per step
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T_out)
    
    mean = port.mean(dim=1)          # (B,)
    std = port.std(dim=1) + eps      # (B,)
    sharpe = mean / std              # (B,)
    # maximize Sharpe -> minimize negative Sharpe
    return -sharpe.mean()

def differentiable_sharpe_loss(
        weights: tensor, returns: tensor, eps: float = 1e-6
    ):
    port_ret = (weights.unsqueeze(1) * returns).sum(-1)   # (B, T)
    mean = port_ret.mean(dim=1)
    var  = port_ret.var(dim=1)          # variance, not std
    # Avoiding the sqrt entirely
    return -(mean / (var.sqrt() + eps)).mean()
    # even more stable:
    # return -(mean**2 / (var + eps)).mean()

def raw_sortino_loss(
        weights: tensor, returns: tensor, target: float = 0.0, eps: float = 1e-8
    ):
    """
    @param weights torch.tensor (B, N)
    @param returns torch.tensor (B, T_out, N) -- raw returns
    @param target float Minimum acceptable return (MAR), often 0 for risk-free rate adjusted.
    @param eps float Epsilon value to avoid divide by zero error

    @return batch average Sortino Ratio. 
        Negative since NN has to maximize Sortino but minimize loss
    """
    # Portfolio returns per step
    port = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T_out)
    
    # Downside deviation: std of negative deviations from target
    downside = clamp(target - port, min=0.0)  # (B, T_out), only positive for downside
    downside_std = downside.std(dim=1) + eps  # (B,)
    
    mean = port.mean(dim=1)  # (B,)
    sortino = mean / downside_std  # (B,)
    
    # Maximize Sortino → minimize negative Sortino
    return -sortino.mean()


# ---------------------------------------------------------------------------
# Backward-compatible wrappers so existing functions accept the new
# 3-or-4 arg call signature used by the modified Trainer.
# ---------------------------------------------------------------------------

def sharpe_loss_compat(weights, returns, features=None, fundamentals=None):
    return differentiable_sharpe_loss(weights, returns)


def sortino_loss_compat(weights, returns, features=None, fundamentals=None):
    return raw_sortino_loss(weights, returns)


# ---------------------------------------------------------------------------
# Composite S/R Loss
# ---------------------------------------------------------------------------

class CompositeSRLoss(nn.Module):
    """
    Multi-component loss that augments the differentiable Sharpe loss
    with four S/R-aware regularization penalties:

    L_total = L_sharpe
              + alpha * L_price_action
              + beta  * L_psychological
              + gamma * L_macro
              + delta * L_fundamental

    The model output (softmax portfolio weights) is unchanged; the loss
    uses the raw input features ``xb`` and an optional pre-computed
    fundamentals tensor to compute the penalty terms.
    """

    def __init__(
        self,
        num_tickers: int,
        num_features_per_ticker: int,
        ret_feature_idx: int,
        turnover_feature_idx: int,
        illiq_feature_idx: int,
        ba_spread_feature_idx: int,
        macro_col_indices: Optional[List[int]] = None,
        alpha: float = 0.10,
        beta: float = 0.05,
        gamma: float = 0.10,
        delta: float = 0.10,
        psych_thresholds: Optional[List[float]] = None,
        psych_sigma: float = 0.01,
        ema_span: int = 10,
    ):
        """
        @param num_tickers int  Number of assets N.
        @param num_features_per_ticker int  Features F per ticker in the
            flat layout produced by Reshaper (T_NxF).
        @param ret_feature_idx int  Index of the RET feature within a
            ticker's F-wide feature block.
        @param turnover_feature_idx int  Index of TURNOVER.
        @param illiq_feature_idx int  Index of ILLIQUIDITY.
        @param ba_spread_feature_idx int  Index of BA_SPREAD.
        @param macro_col_indices list[int]  Indices of macro features
            within a ticker's feature block (after broadcast).
        @param alpha float  Weight for price-action S/R penalty.
        @param beta float  Weight for psychological-level penalty.
        @param gamma float  Weight for macro-regime penalty.
        @param delta float  Weight for SEC-fundamental penalty.
        @param psych_thresholds list[float]  Return-space thresholds.
        @param psych_sigma float  Gaussian kernel bandwidth.
        @param ema_span int  EMA lookback for return smoothing.
        """
        super().__init__()
        self.N = num_tickers
        self.F = num_features_per_ticker
        self.ret_idx = ret_feature_idx
        self.turn_idx = turnover_feature_idx
        self.illiq_idx = illiq_feature_idx
        self.spread_idx = ba_spread_feature_idx
        self.macro_idx = macro_col_indices or []

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.psych_sigma = psych_sigma
        self.ema_span = ema_span

        if psych_thresholds is None:
            psych_thresholds = [
                0.0, 0.01, -0.01, 0.02, -0.02,
                0.05, -0.05, 0.10, -0.10, 0.20, -0.20,
            ]
        self.register_buffer(
            "psych_thresholds",
            torch.tensor(psych_thresholds, dtype=torch.float32),
        )

        # Learnable regime weights (one per macro feature)
        if self.macro_idx:
            self.regime_weights = nn.Parameter(
                torch.zeros(len(self.macro_idx))
            )
        else:
            self.regime_weights = None

        # EMA decay factor
        ema_alpha = 2.0 / (ema_span + 1)
        self.register_buffer(
            "ema_alpha", torch.tensor(ema_alpha, dtype=torch.float32)
        )

    # -- helpers -------------------------------------------------------------

    def _extract_ticker_feature(
        self, features: torch.Tensor, feat_idx: int,
    ) -> torch.Tensor:
        """
        From flat (B, T, N*F) extract one feature for all tickers -> (B, T, N).
        """
        B, T, _ = features.shape
        indices = [
            ticker_i * self.F + feat_idx for ticker_i in range(self.N)
        ]
        idx_tensor = torch.tensor(
            indices, device=features.device, dtype=torch.long
        )
        return features[:, :, idx_tensor]  # (B, T, N)

    def _ema_smooth(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exponential moving average along dim=1 (time).
        x: (B, T, N)
        """
        B, T, N = x.shape
        ema = torch.zeros(B, N, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            ema = self.ema_alpha * x[:, t, :] + (1 - self.ema_alpha) * ema
            outputs.append(ema)
        return torch.stack(outputs, dim=1)  # (B, T, N)

    @staticmethod
    def _z_score(x: torch.Tensor, dim: int = 1, eps: float = 1e-8):
        """Z-score along *dim*."""
        mu = x.mean(dim=dim, keepdim=True)
        sigma = x.std(dim=dim, keepdim=True) + eps
        return (x - mu) / sigma

    # -- sub-losses ----------------------------------------------------------

    def _price_action_penalty(
        self,
        weights: torch.Tensor,
        features: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Detect S/R conditions from microstructure features and penalise
        portfolio weights that ignore these signals.
        """
        turnover = self._extract_ticker_feature(features, self.turn_idx)
        illiq = self._extract_ticker_feature(features, self.illiq_idx)
        spread = self._extract_ticker_feature(features, self.spread_idx)
        rets = self._extract_ticker_feature(features, self.ret_idx)

        turn_z = self._z_score(turnover)[:, -1, :]     # (B, N)
        illiq_z = self._z_score(illiq)[:, -1, :]
        spread_z = self._z_score(spread)[:, -1, :]

        smoothed = self._ema_smooth(rets)
        reversal_signal = smoothed[:, -1, :] - smoothed[:, -2, :] if smoothed.shape[1] >= 2 else torch.zeros_like(smoothed[:, -1, :])

        confidence = torch.sigmoid(turn_z + illiq_z + spread_z)
        direction = torch.tanh(smoothed[:, -1, :])

        equal_w = 1.0 / self.N
        weight_diff = weights - equal_w

        return -(confidence * direction * weight_diff).mean()

    def _psychological_penalty(
        self,
        weights: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalise when cumulative returns sit near psychological thresholds
        and the model ignores this.
        """
        rets = self._extract_ticker_feature(features, self.ret_idx)  # (B, T, N)

        cum_ret = (1 + rets).prod(dim=1) - 1  # (B, N)

        # Distance to nearest threshold: (B, N, K)
        thresholds = self.psych_thresholds  # (K,)
        diffs = cum_ret.unsqueeze(-1) - thresholds.unsqueeze(0).unsqueeze(0)
        abs_diffs = diffs.abs()
        min_dist, nearest_idx = abs_diffs.min(dim=-1)  # (B, N)

        proximity = torch.exp(-min_dist.pow(2) / (2 * self.psych_sigma ** 2))

        nearest_thresh = thresholds[nearest_idx]  # (B, N)
        direction = torch.sign(cum_ret - nearest_thresh)
        direction = torch.where(direction == 0, torch.ones_like(direction), direction)

        equal_w = 1.0 / self.N
        weight_diff = weights - equal_w

        return -(proximity * direction * weight_diff).mean()

    def _macro_regime_penalty(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute risk-on/risk-off regime from macro features and penalise
        portfolio returns that are misaligned with the regime direction.
        """
        if self.regime_weights is None or not self.macro_idx:
            return torch.tensor(0.0, device=weights.device)

        macro_features = []
        for idx in self.macro_idx:
            mf = self._extract_ticker_feature(features, idx)  # (B, T, N)
            macro_features.append(mf.mean(dim=-1))  # average across tickers -> (B, T)

        macro_stack = torch.stack(macro_features, dim=-1)  # (B, T, M)

        # Rate of change over the window
        delta_macro = macro_stack[:, -1, :] - macro_stack[:, 0, :]  # (B, M)

        regime = torch.tanh(
            (delta_macro * self.regime_weights.unsqueeze(0)).sum(dim=-1)
        )  # (B,)

        port_ret = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T_out)
        mean_port_ret = port_ret.mean(dim=1)  # (B,)

        return -(regime * mean_port_ret).mean()

    def _fundamental_penalty(
        self,
        weights: torch.Tensor,
        fundamentals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalise allocations misaligned with fundamental quality scores.

        @param fundamentals torch.Tensor (B, N) composite z-scored scores
            (positive = improving, negative = deteriorating).
        """
        equal_w = 1.0 / self.N
        weight_diff = weights - equal_w
        return -(fundamentals * weight_diff).mean()

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        fundamentals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        @param weights (B, N) portfolio allocation weights
        @param returns (B, T_out, N) future returns
        @param features (B, T_in, N*F) raw input window (from xb)
        @param fundamentals (B, N) per-ticker composite fundamental scores
        """
        loss = differentiable_sharpe_loss(weights, returns)

        if features is not None:
            loss = loss + self.alpha * self._price_action_penalty(
                weights, features, returns
            )
            loss = loss + self.beta * self._psychological_penalty(
                weights, features
            )
            loss = loss + self.gamma * self._macro_regime_penalty(
                weights, returns, features
            )

        if fundamentals is not None:
            loss = loss + self.delta * self._fundamental_penalty(
                weights, fundamentals
            )

        return loss