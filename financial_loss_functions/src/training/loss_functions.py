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
# Learnable modules for multi-timeframe S/R and macro override
# ---------------------------------------------------------------------------

class TimeframeImportanceFn(nn.Module):
    """
    Maps normalised lookback duration in [0, 1] to an importance weight
    in (0, 1).  Initialised with a monotonic bias so that longer
    lookbacks produce higher importance by default.

    Architecture: Linear(1, H) -> Softplus -> Linear(H, 1) -> Sigmoid
    """

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
        )
        # Monotonic-bias init: positive weights so larger inputs → larger outputs
        with torch.no_grad():
            self.net[0].weight.fill_(1.0)
            self.net[0].bias.fill_(0.0)
            self.net[2].weight.fill_(1.0)
            self.net[2].bias.fill_(0.0)

    def forward(self, normed_lookbacks: torch.Tensor) -> torch.Tensor:
        """
        @param normed_lookbacks (W,) values in [0, 1]
        @return (W,) importance weights in (0, 1)
        """
        out = self.net(normed_lookbacks.unsqueeze(-1))  # (W, 1)
        return torch.sigmoid(out.squeeze(-1))            # (W,)


class MacroOverrideGate(nn.Module):
    """
    Learnable gate that maps macro rate-of-change to an override
    weight omega in (0, 1).

    omega → 1 : macro signals dominate (extreme macro conditions)
    omega → 0 : technical/fundamental signals dominate (calm macro)
    """

    def __init__(self, num_macro_features: int, hidden: int = 8):
        super().__init__()
        self.baseline = nn.Parameter(torch.tensor(0.0))
        self.proj = nn.Sequential(
            nn.Linear(num_macro_features, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
        )

    def forward(self, delta_macro: torch.Tensor) -> torch.Tensor:
        """
        @param delta_macro (B, M) macro rate-of-change
        @return (B,) override weight in (0, 1)
        """
        activation = self.proj(delta_macro).squeeze(-1)  # (B,)
        return torch.sigmoid(self.baseline + activation)


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

    Optional enhancements (all backward-compatible, disabled by default):
    - Multi-timeframe S/R hierarchy via rolling-window pivots
    - Adaptive macro-override gate that re-weights penalties
    - Sector-aware penalty scaling
    - Ticker-specific macro sensitivity
    - Cross-ticker correlation guard
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
        # --- Multi-timeframe S/R ---
        sr_use_multi_timeframe: bool = False,
        sr_lookback_windows: Optional[List[int]] = None,
        sr_pivot_threshold: float = 0.02,
        sr_importance_hidden: int = 8,
        # --- Macro override gate ---
        use_macro_override: bool = False,
        macro_override_hidden: int = 8,
        # --- Sector-aware penalty ---
        sector_ids: Optional[List[int]] = None,
        # --- Ticker macro sensitivity ---
        ticker_macro_sensitivity: Optional[torch.Tensor] = None,
        # --- Correlation guard ---
        corr_matrix: Optional[torch.Tensor] = None,
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
        @param sr_use_multi_timeframe bool  Enable multi-TF S/R hierarchy.
        @param sr_lookback_windows list[int]  Lookback periods in days.
        @param sr_pivot_threshold float  Soft activation threshold for S/R.
        @param sr_importance_hidden int  Hidden size for importance MLP.
        @param use_macro_override bool  Enable adaptive macro override gate.
        @param macro_override_hidden int  Hidden size for gate MLP.
        @param sector_ids list[int]  Integer sector ID per ticker (len N).
        @param ticker_macro_sensitivity Tensor  (N,) macro sensitivity per ticker.
        @param corr_matrix Tensor  (N, N) return correlation matrix.
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

        # ---- Multi-timeframe S/R hierarchy ----
        self.use_multi_tf = sr_use_multi_timeframe
        self.sr_pivot_threshold = sr_pivot_threshold
        if self.use_multi_tf:
            windows = sr_lookback_windows or [5, 10, 21, 42, 63, 105]
            self.sr_windows = sorted(windows)
            max_w = float(max(self.sr_windows))
            self.register_buffer(
                "sr_lookback_normed",
                torch.tensor([w / max_w for w in self.sr_windows], dtype=torch.float32),
            )
            self.importance_fn = TimeframeImportanceFn(sr_importance_hidden)
        else:
            self.sr_windows = []
            self.importance_fn = None

        # ---- Macro override gate ----
        self.use_macro_override = use_macro_override
        if use_macro_override and self.macro_idx:
            self.macro_override = MacroOverrideGate(
                len(self.macro_idx), macro_override_hidden
            )
        else:
            self.macro_override = None

        # ---- Sector-aware penalty scaling ----
        if sector_ids is not None:
            self.register_buffer(
                "sector_ids",
                torch.tensor(sector_ids, dtype=torch.long),
            )
            self.num_sectors = len(set(sector_ids))
        else:
            self.sector_ids = None
            self.num_sectors = 0

        # ---- Ticker macro sensitivity ----
        if ticker_macro_sensitivity is not None:
            self.register_buffer("ticker_macro_sensitivity", ticker_macro_sensitivity)
        else:
            self.ticker_macro_sensitivity = None

        # ---- Correlation guard ----
        if corr_matrix is not None:
            self.register_buffer("corr_matrix", corr_matrix)
        else:
            self.corr_matrix = None

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

    def _compute_delta_macro(
        self, features: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Shared helper: macro rate-of-change over the input window.

        @return (B, M) or None if no macro features.
        """
        if not self.macro_idx:
            return None
        macro_list = []
        for idx in self.macro_idx:
            mf = self._extract_ticker_feature(features, idx)  # (B, T, N)
            macro_list.append(mf.mean(dim=-1))  # average across tickers -> (B, T)
        macro_stack = torch.stack(macro_list, dim=-1)  # (B, T, M)
        return macro_stack[:, -1, :] - macro_stack[:, 0, :]  # (B, M)

    def _detect_pivots(
        self, prices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Detect support/resistance levels at multiple lookback windows.

        @param prices (B, T, N) cumulative price proxy
        @return (B, W, N)  normalised position in [-1, +1] per window.
            +1 = at window high (resistance), -1 = at window low (support).
        """
        B, T, N = prices.shape
        current = prices[:, -1, :]  # (B, N)
        scores = []
        for w in self.sr_windows:
            w_clamped = min(w, T)
            window_slice = prices[:, -w_clamped:, :]         # (B, w', N)
            w_high = window_slice.max(dim=1).values           # (B, N)
            w_low = window_slice.min(dim=1).values            # (B, N)
            w_range = (w_high - w_low).clamp(min=1e-8)
            midpoint = (w_high + w_low) / 2.0
            score = ((current - midpoint) / (w_range / 2.0)).clamp(-1.0, 1.0)
            scores.append(score)
        return torch.stack(scores, dim=1)  # (B, W, N)

    def _sector_avg_confidence(
        self, confidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sector average confidence and map back to per-ticker.

        @param confidence (B, N)
        @return (B, N) sector-average confidence for each ticker's sector.
        """
        B, N = confidence.shape
        sector_avg = torch.zeros(B, self.num_sectors, device=confidence.device)
        sector_count = torch.zeros(self.num_sectors, device=confidence.device)
        for s in range(self.num_sectors):
            mask = (self.sector_ids == s)  # (N,)
            if mask.any():
                sector_avg[:, s] = confidence[:, mask].mean(dim=-1)
                sector_count[s] = mask.float().sum()
        # Map sector averages back to each ticker
        return sector_avg[:, self.sector_ids]  # (B, N)

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

        When multi-timeframe is enabled, rolling-window pivot detection
        augments the microstructure confidence with S/R level strength
        weighted by a learnable importance function.
        """
        turnover = self._extract_ticker_feature(features, self.turn_idx)
        illiq = self._extract_ticker_feature(features, self.illiq_idx)
        spread = self._extract_ticker_feature(features, self.spread_idx)
        rets = self._extract_ticker_feature(features, self.ret_idx)

        turn_z = self._z_score(turnover)[:, -1, :]     # (B, N)
        illiq_z = self._z_score(illiq)[:, -1, :]
        spread_z = self._z_score(spread)[:, -1, :]

        smoothed = self._ema_smooth(rets)

        confidence = torch.sigmoid(turn_z + illiq_z + spread_z)
        direction = torch.tanh(smoothed[:, -1, :])

        # ---- Multi-timeframe S/R hierarchy ----
        sr_active = None
        if self.use_multi_tf and self.importance_fn is not None:
            cum_prices = (1 + rets).cumprod(dim=1)                  # (B, T, N)
            pivot_scores = self._detect_pivots(cum_prices)          # (B, W, N)
            importance = self.importance_fn(self.sr_lookback_normed) # (W,)
            imp_sum = importance.sum().clamp(min=1e-8)

            # Importance-weighted aggregation
            imp = importance[None, :, None]                         # (1, W, 1)
            weighted_strength = (pivot_scores.abs() * imp).sum(dim=1) / imp_sum  # (B, N)
            weighted_direction = (pivot_scores * imp).sum(dim=1) / imp_sum       # (B, N)

            # Soft gate: activates when aggregate S/R strength exceeds threshold
            sr_active = torch.sigmoid(
                (weighted_strength - self.sr_pivot_threshold) / 0.01
            )  # (B, N)

            # Multiplicative boost to microstructure confidence
            confidence = confidence * (1.0 + sr_active)
            # Blend EMA direction with S/R direction
            sr_dir = torch.tanh(weighted_direction)
            direction = (1.0 - sr_active) * direction + sr_active * sr_dir

        # ---- Sector-aware scaling ----
        if self.sector_ids is not None:
            sec_avg = self._sector_avg_confidence(confidence)
            confidence = confidence * (1.0 + sec_avg)

        equal_w = 1.0 / self.N
        weight_diff = weights - equal_w

        penalty = -(confidence * direction * weight_diff).mean()

        # ---- Correlation guard (only when multi-TF active) ----
        if (
            self.corr_matrix is not None
            and sr_active is not None
        ):
            w_col = weights.unsqueeze(-1)                          # (B, N, 1)
            port_corr = (
                w_col * self.corr_matrix.unsqueeze(0) * w_col.transpose(-1, -2)
            ).sum(dim=(-1, -2))                                    # (B,)
            sr_corr_penalty = (sr_active.mean(dim=-1) * port_corr).mean()
            penalty = penalty + sr_corr_penalty

        return penalty

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

        # Use cumulative sum (not product) — features are RobustScaler-normalized,
        # so (1+rets).prod() diverges. Sum over the window is well-behaved.
        cum_ret = rets.sum(dim=1)  # (B, N)

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
        delta_macro: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute risk-on/risk-off regime from macro features and penalise
        portfolio returns that are misaligned with the regime direction.

        @param delta_macro (B, M) pre-computed macro delta (avoids recomputing).
        """
        if self.regime_weights is None or not self.macro_idx:
            return torch.tensor(0.0, device=weights.device)

        if delta_macro is None:
            delta_macro = self._compute_delta_macro(features)

        regime = torch.tanh(
            (delta_macro * self.regime_weights.unsqueeze(0)).sum(dim=-1)
        )  # (B,)

        port_ret = (weights.unsqueeze(1) * returns).sum(dim=-1)  # (B, T_out)

        # ---- Ticker macro sensitivity ----
        if self.ticker_macro_sensitivity is not None:
            # Weight individual returns by macro sensitivity before portfolio sum
            sens = self.ticker_macro_sensitivity.unsqueeze(0)  # (1, N)
            weighted_port = (weights * sens).sum(dim=-1)       # (B,)
            # Combine: portfolio return scaled by sensitivity-weighted exposure
            mean_port_ret = (weighted_port * port_ret.mean(dim=1))
        else:
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
        # Skip if fundamentals have no cross-sectional variance (e.g. broken SEC data)
        if fundamentals.var(dim=-1).mean() < 1e-4:
            return torch.tensor(0.0, device=weights.device)

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
            # Pre-compute macro delta (shared by override gate and regime penalty)
            delta_macro = self._compute_delta_macro(features)

            # ---- Adaptive macro override gate ----
            if self.macro_override is not None and delta_macro is not None:
                omega = self.macro_override(delta_macro).mean()  # scalar
            else:
                omega = torch.tensor(0.0, device=weights.device)

            # Scale penalty weights: macro boosted, others reduced
            alpha_eff = self.alpha * (1.0 - omega)
            beta_eff = self.beta * (1.0 - omega)
            gamma_eff = self.gamma * (1.0 + omega)
            delta_eff = self.delta * (1.0 - omega)

            loss = loss + alpha_eff * self._price_action_penalty(
                weights, features, returns
            )
            loss = loss + beta_eff * self._psychological_penalty(
                weights, features
            )
            loss = loss + gamma_eff * self._macro_regime_penalty(
                weights, returns, features, delta_macro
            )

            if fundamentals is not None:
                loss = loss + delta_eff * self._fundamental_penalty(
                    weights, fundamentals
                )
        elif fundamentals is not None:
            # No features but fundamentals provided
            loss = loss + self.delta * self._fundamental_penalty(
                weights, fundamentals
            )

        return loss
