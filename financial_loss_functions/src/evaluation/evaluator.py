import numpy as np
import pandas as pd
from typing import Callable

from cvxopt import matrix, solvers

class Evaluator:
    """
    Class to evaulate and compare all generated weights from all models/methods,
    for all windows againsts each other as well as benchmarks.
    """
    spread_cost_factor = 0.5
    
    def __init__(
            self, 
            eval_returns: np.ndarray | None, 
            ba_eval: np.ndarray | None = None, 
            metrics_lib: dict[str, Callable] | None = None,
            all_daily_returns: dict[str, np.ndarray] | None = None
        ):
        """
        Initialize Evaluator to calculate the portfolio returns, 
        evaulate them and compare all generated weights.

        Args:
            eval_windows (np.ndarray | None): Array of evaluation (out-of-sample)
                return windows for all stocks. Must provide array or None.
            ba_eval (np.ndarray | None): Array of Bid-Ask Spreads for each stock on the 
                first day of every window. This is used to calculate Bid-Ask Spread trading costs, 
                if provided. Default = None.
            metrics_lib (dict[str, Callable] | None): Metrics library dictionary containing metric 
                name and metric function. Default = None.
        """

        if eval_returns is not None and isinstance(eval_returns, np.ndarray):
            if eval_returns.ndim != 3:
                raise ValueError(
                    f'ERROR: Evaluation Returns must have 3 dim, got {self.eval_returns.ndim}.'
                )
            
            self.eval_returns = eval_returns
                
            if isinstance(ba_eval, np.ndarray) and ba_eval.ndim == 2:
                self.ba_eval = ba_eval
            else:
                print(
                    '!Evaluator did not get BA Spread data, or incorrect shape.',
                    'Not accounting for trading costs.!'
                )
                self.ba_eval = None
            
            # Returns for each window
            self.all_daily_returns = {} # Add all returns for every window

        else:
            if all_daily_returns is None:
                raise ValueError(
                    'If out-of-sample evaluation data is not provided, daily returns of all models must be provided.'
                )
            else:
                self.eval_returns = None
                self.ba_eval = None
                self.all_daily_returns = all_daily_returns

        # Per-strategy allocation weights, keyed by model name. Populated by
        # calc_pf_daily_rets so turnover and cost-adjusted returns can be derived
        # later without recomputing them from the model.
        self.all_weights: dict[str, np.ndarray] = {}

        self.metrics_lib = metrics_lib
    
    def _calc_step_ba_costs(
            self,
            prev_weights: np.ndarray | None, 
            curr_weights: np.ndarray,
            first_d_bas: np.ndarray
        ) -> np.float64:
        
        # Compute cost
        if prev_weights is None:
            delta = curr_weights # For the first step
        else:
            delta = np.abs(curr_weights - prev_weights) # For steps after the first step

        # Calculate BA spread costs
        cost = self.spread_cost_factor * np.sum(delta * first_d_bas)

        return cost
    
    @staticmethod
    def _calc_net_returns(pf_daily_rets: np.ndarray, cost: float) -> np.ndarray:
        # Apply cost to the first  to get Net returns
        pf_daily_rets[0] = (1 + pf_daily_rets[0]) * (1 - cost) - 1

        return pf_daily_rets

    def calc_pf_daily_rets(self, eval_weights: np.ndarray, model_name: str):
        """
        Calculates daily returns for the given portfolio weights for each given window.
        Portfolio Weights (n,) x Returns (T, n) = weighted returns.

        @param eval_weights np.ndarray
            Portfolio allocation weights for which weighted returns need to be calculated
        @param model_name str
            Name of the model which generated the portfolio allocation weights
        """
        
        pf_daily_returns = []

        if eval_weights.ndim == 2:
            if self.ba_eval is not None:
                
                prev_weights = None 
                # Initialized to None because the first step will not have previous weights

                # Iterating over window samples
                for i in range(eval_weights.shape[0]):
                    weights = eval_weights[i]  # Shape: (50,)
                    returns_matrix = self.eval_returns[i]  # Shape: (60, 50) - time steps x assets
                    first_d_bas = self.ba_eval[i] # Shape: (50,) 1 stime step x assets

                    cost = self._calc_step_ba_costs(prev_weights, weights, first_d_bas)
                    
                    # Calculate daily gross portfolio returns (dot product at each time step)
                    daily_returns = np.dot(returns_matrix, weights) # Shape: (60,)
                    
                    # Calculate net returns
                    daily_returns = self._calc_net_returns(daily_returns, cost)
                    
                    pf_daily_returns.append(daily_returns)

                    prev_weights = weights # update previous weights
            
            else:
                # Iterating over window samples
                for i in range(eval_weights.shape[0]):
                    weights = eval_weights[i]  # Shape: (50,)
                    returns_matrix = self.eval_returns[i]  # Shape: (60, 50) - time steps x assets
                    
                    # Calculate daily portfolio returns (dot product at each time step)
                    daily_returns = np.dot(returns_matrix, weights)
                    pf_daily_returns.append(daily_returns) # Shape: (50,)
             
            self.all_daily_returns[model_name] = np.array(pf_daily_returns)
            # Cache the raw weights so turnover and cost-adjusted returns can be
            # derived per strategy without rerunning the model.
            self.all_weights[model_name] = np.asarray(eval_weights)
        else:
            print(
                f'DEBUG: Evaluation weights array must have only 2 dims, got {eval_weights.ndim}.'
                f'Skipping {model_name}!'
            )
    
    def get_rets_for_one(self, model_name: str) -> np.ndarray:
        return self.all_daily_returns.get(model_name)

    def update_rets_for_one(self, model_name: str, new_returns: np.ndarray):
        if model_name in self.all_daily_returns:
            self.all_daily_returns.update({model_name: new_returns})
        else:
            raise Warning(
                f'Returns for {model_name} do not exist, hence not updating any returns.'
            )
    
    def add_benchmark_rets(self, bench_name: str, bench_rets: np.ndarray):
        """
        Add benchmark returns for the respective evalulation output windows. eg., S&P500 daily returns
        """
        self.all_daily_returns.update({bench_name: bench_rets})

    def add_benchmark_weights(self, bench_name: str, weights: np.ndarray):
        """
        Attach per-window weights for a benchmark/strategy whose daily returns
        were added separately (e.g. classical calculators like MinVariance).
        Enables turnover and cost-adjusted returns for that strategy.
        """
        if weights.ndim != 2:
            raise ValueError(
                f"add_benchmark_weights: expected 2D (num_windows, N), got {weights.ndim}D."
            )
        self.all_weights[bench_name] = np.asarray(weights)

    def calc_turnover_for_all(self) -> pd.Series:
        """
        Per-strategy mean one-way turnover across consecutive windows, using
        cached weights. Strategies with no cached weights (benchmark returns
        added via add_benchmark_rets without weights) are skipped.

        Returns:
            pd.Series indexed by strategy name.
        """
        from src.evaluation.metrics import turnover as _turnover

        if not self.all_weights:
            print('No weights cached; nothing to compute turnover for.')
            return pd.Series(dtype=float)

        out = {name: float(_turnover(w)) for name, w in self.all_weights.items()}
        return pd.Series(out, name="turnover")

    def calc_cost_adjusted_daily_rets(
            self, model_name: str, cost_bps: float = 10.0
        ) -> np.ndarray:
        """
        Return a copy of `model_name`'s per-window daily returns with a uniform
        transaction cost charged on the first day of each window, proportional
        to the one-way turnover relative to the previous window.

        Unlike the bid-ask path in calc_pf_daily_rets (which needs per-asset
        spreads), this method applies a single round-trip cost assumption
        expressed in basis points of gross traded notional.

        Args:
            model_name (str): strategy whose weights have been cached.
            cost_bps (float): one-way transaction cost in basis points applied
                to L1 turnover (default 10 bps = 0.10 %).

        Returns:
            np.ndarray of shape (num_windows, T) — a fresh copy, not in-place.

        Raises:
            KeyError: if weights or daily returns for `model_name` are missing.
        """
        if model_name not in self.all_weights:
            raise KeyError(f"No cached weights for '{model_name}'. Run calc_pf_daily_rets first.")
        if model_name not in self.all_daily_returns:
            raise KeyError(f"No cached daily returns for '{model_name}'.")

        weights = self.all_weights[model_name]                  # (num_windows, N)
        daily_returns = np.array(self.all_daily_returns[model_name], copy=True)
        cost_rate = float(cost_bps) / 1e4

        prev = None
        for i in range(weights.shape[0]):
            delta = weights[i] if prev is None else np.abs(weights[i] - prev)
            cost = cost_rate * delta.sum()
            daily_returns[i] = self._calc_net_returns(daily_returns[i], cost)
            prev = weights[i]
        return daily_returns

    def _daily_rets_calcd_check(self):
        if not self.all_daily_returns:
            raise ValueError(
                'No daily returns calculated.',
                'Run calc_pf_daily_rets and calc_eq_wt_daily_rets first.'
            )

    def calc_metric_performance(
            self, metric_func: Callable, mean: bool= False
        ) -> pd.Series | pd.DataFrame:
        """
        Calculate per-window performance of all portfolios (incl. Equal Weight)
        based on given metric. 

        Args:
            metric_func (Callable): Metric function to used to calculate a portfolio metric.
            mean (bool): If True, mean of the metric over the entire provided split of returned.

        Returns:
            Dict[str, list]: Dictionary containing calculated performance metric for each validation window
        """
        self._daily_rets_calcd_check()
        
        metric_perfomances = {}
        for model, all_rets in self.all_daily_returns.items():
            model_rets = []
            for i in range(all_rets.shape[0]):
                window_metric = metric_func(all_rets[i])
                model_rets.append(round(window_metric, 4))
            
            metric_perfomances[model] = model_rets
        if mean:
            return pd.DataFrame(metric_perfomances).mean()
        else:
            return pd.DataFrame(metric_perfomances)
    
    def calc_metric_performance_ci(
            self,
            metric_func: Callable,
            n_boot: int = 1000,
            ci: float = 0.95,
            seed: int | None = None,
        ) -> pd.DataFrame:
        """
        For each stored strategy, flatten all windows into a single daily-return
        series and report bootstrap point / lower / upper / p-value.

        Uses the stationary bootstrap (Politis & Romano, 1994) so short-range
        serial correlation in daily returns is preserved.

        Args:
            metric_func (Callable): scalar metric mapping (T,)-array → float.
            n_boot (int): bootstrap resamples per strategy.
            ci (float): two-sided coverage.
            seed (int | None): RNG seed (None = non-reproducible draws).

        Returns:
            pd.DataFrame indexed by strategy name with columns
            ['point', 'lower', 'upper', 'p_value_gt_zero'].
        """
        from src.evaluation.metrics import bootstrap_metric_ci

        self._daily_rets_calcd_check()

        rows = {}
        for model, all_rets in self.all_daily_returns.items():
            flat = np.asarray(all_rets).flatten()
            rows[model] = bootstrap_metric_ci(
                flat, metric_func,
                n_boot=n_boot, ci=ci, seed=seed,
            )
        return pd.DataFrame.from_dict(rows, orient="index")

    def calc_paired_diff_ci(
            self,
            model_a: str,
            model_b: str,
            metric_func: Callable,
            n_boot: int = 1000,
            ci: float = 0.95,
            seed: int | None = None,
        ) -> dict:
        """
        Bootstrap CI for metric(model_a) - metric(model_b). Positive lower bound
        on the CI is the usual "A significantly beats B" claim.
        """
        from src.evaluation.metrics import bootstrap_paired_diff_ci

        self._daily_rets_calcd_check()
        if model_a not in self.all_daily_returns:
            raise KeyError(f"No daily returns stored for '{model_a}'")
        if model_b not in self.all_daily_returns:
            raise KeyError(f"No daily returns stored for '{model_b}'")

        a = np.asarray(self.all_daily_returns[model_a]).flatten()
        b = np.asarray(self.all_daily_returns[model_b]).flatten()
        return bootstrap_paired_diff_ci(
            a, b, metric_func,
            n_boot=n_boot, ci=ci, seed=seed,
        )

    def calc_avg_performance(self) -> pd.DataFrame | None:
        
        if self.metrics_lib:
            all_metrics_perf = []
            for met_name, met_func in self.metrics_lib.items():
                met_perf = self.calc_metric_performance(met_func, mean=True)
                met_perf.name = met_name
                all_metrics_perf.append(met_perf)
            
            avg_perf = pd.concat(all_metrics_perf, axis=1)

            return avg_perf

        else:
            print('No metrics library or dict provided. Cannot run average performance over metrics.')
            return None

    def get_all_daily_returns(self):
        return self.all_daily_returns
    
    def update_spread_cost_factor(self, spread_cost_factor: float):
        if spread_cost_factor > 1.0:
            raise ValueError('Spread Cost factor cannot be greater than 1.')
        else:
            self.spread_cost_factor = spread_cost_factor

class EqualWeightCalculator:
    """
    Class to calculate weights and daily returns for an equal weight portfolio.
    """
    def __init__(self, eval_returns: np.ndarray):
        """
        Initialize EqualWeightCalculator to calculate equal weights for the 
        given stocks and calculate its returns.

        Args:
            eval_windows (np.ndarray): Array of evaluation (out-of-sample)
                return windows for all stocks.
        """
        self.eval_returns = eval_returns
        
        # Equal weight for all stocks
        self.eq_weights = None

        # Returns for the equal weight portfolio
        self.eq_weights_rets = None
    
    @staticmethod
    def _equal_weight_pf(num_tickers: int) -> np.ndarray:
        """
        Calculates simple equal weights for a portfolio
        weight for each stock = 1/num_tickers
        
        Args:
            num_tickers (int): Number of tickers in the dataset

        Returns:
            np.array: Equal weight portfolio allocation weights
        """
        return np.full((num_tickers), 1/num_tickers)
    
    def calc_eq_wt_daily_rets(self) -> np.ndarray: 
        """
        Calculates daily returns for the Equal Weighted portfolio for each given window.
        """
        # For equal weight portfolio
        self.eq_weights = self._equal_weight_pf(self.eval_returns.shape[2])
        
        eq_wt_daily_returns = []
        
        for i in range(self.eval_returns.shape[0]):
            returns = self.eval_returns[i]  # Shape: (50, 50)
            daily_returns = np.dot(returns, self.eq_weights)
            eq_wt_daily_returns.append(daily_returns)  # Shape: (50,)

        self.eq_weights_rets = np.array(eq_wt_daily_returns)

        return self.eq_weights_rets

    def get_eq_weights(self) -> np.ndarray | None:
        """
        Getter function to get equal weights for the number of stocks in the provideed data.

        Returns:
            eq_weights (np.ndarray | None): Array containing weights for every window from the data.
        """
        if self.eq_weights:
            return self.eq_weights
        else:
            print(
                'WARNING: No equal weights calculated.',
                'Run `EqualWeightCalculator.calc_eq_wt_daily_rets()` first.'
            )
            return None

class MinVarianceCalculator:
    """
    Classical long-only minimum-variance benchmark.

    For each evaluation window, solves
        min  w^T Σ w
        s.t. sum(w) = 1, w >= 0
    where Σ is the sample covariance of in-window returns. The resulting weights
    are then applied to the *same* window's returns to produce daily returns,
    matching the out-of-sample convention already used by EqualWeightCalculator.
    """

    def __init__(self, eval_returns: np.ndarray, shrink: float = 0.0):
        """
        Args:
            eval_returns (np.ndarray): (num_windows, T, N) out-of-sample return windows.
            shrink (float): Optional linear shrinkage of Σ toward a scaled identity,
                cov_shrunk = (1-s)·cov + s·(tr(cov)/N)·I. Default = 0 (no shrinkage).
        """
        if eval_returns.ndim != 3:
            raise ValueError(
                f"MinVarianceCalculator: eval_returns must be 3D, got {eval_returns.ndim}D."
            )
        if not (0.0 <= shrink < 1.0):
            raise ValueError(f"shrink must be in [0, 1), got {shrink}")
        self.eval_returns = eval_returns
        self.shrink = shrink
        self.weights: np.ndarray | None = None
        self.daily_returns: np.ndarray | None = None

    @staticmethod
    def _solve_long_only_min_var(cov: np.ndarray) -> np.ndarray:
        """Solve the standard simplex-constrained min-variance QP via cvxopt."""
        N = cov.shape[0]
        # cvxopt solves: min 0.5 x^T P x + q^T x  s.t. Gx <= h, Ax = b.
        P = matrix(cov.astype(np.float64))
        q = matrix(np.zeros(N))
        G = matrix(-np.eye(N))          # -x <= 0  => x >= 0
        h = matrix(np.zeros(N))
        A = matrix(np.ones((1, N)))     # sum(x) == 1
        b = matrix(np.ones(1))
        solvers.options["show_progress"] = False
        sol = solvers.qp(P, q, G, h, A, b)
        w = np.array(sol["x"]).flatten()
        # Tiny negative values can slip through; clamp and renormalize.
        w = np.clip(w, 0.0, None)
        s = w.sum()
        return w / s if s > 0 else np.full(N, 1.0 / N)

    def _shrink_cov(self, cov: np.ndarray) -> np.ndarray:
        if self.shrink <= 0.0:
            return cov
        N = cov.shape[0]
        target = (np.trace(cov) / N) * np.eye(N)
        return (1.0 - self.shrink) * cov + self.shrink * target

    def calc_min_var_daily_rets(self) -> np.ndarray:
        """
        For each window, fit Σ on the window's returns, solve for min-var weights,
        and produce the per-step portfolio returns on the same window. Stores and
        returns an array of shape (num_windows, T).
        """
        num_windows = self.eval_returns.shape[0]
        weights_per_window = []
        daily_returns_per_window = []

        for i in range(num_windows):
            window = self.eval_returns[i]                # (T, N)
            cov = np.cov(window, rowvar=False)           # (N, N)
            cov = self._shrink_cov(cov)
            w = self._solve_long_only_min_var(cov)       # (N,)
            weights_per_window.append(w)
            daily_returns_per_window.append(window @ w) # (T,)

        self.weights = np.stack(weights_per_window, axis=0)       # (num_windows, N)
        self.daily_returns = np.stack(daily_returns_per_window)   # (num_windows, T)
        return self.daily_returns

    def get_weights(self) -> np.ndarray | None:
        return self.weights


class EqualRiskContribCalculator:
    """
    Classical Equal-Risk-Contribution (ERC) / risk-parity benchmark.

    For each evaluation window, finds long-only weights where each asset
    contributes equally to portfolio variance:
        RC_i = w_i · (Σ w)_i   must all equal σ_p^2 / N.

    Uses the iterative fixed-point update from Maillard, Roncalli & Teiletche
    (2010), which is simple, monotonic, and robust enough for small N.
    """

    def __init__(
        self,
        eval_returns: np.ndarray,
        shrink: float = 0.0,
        max_iter: int = 500,
        tol: float = 1e-8,
    ):
        if eval_returns.ndim != 3:
            raise ValueError(
                f"EqualRiskContribCalculator: eval_returns must be 3D, got {eval_returns.ndim}D."
            )
        if not (0.0 <= shrink < 1.0):
            raise ValueError(f"shrink must be in [0, 1), got {shrink}")
        self.eval_returns = eval_returns
        self.shrink = shrink
        self.max_iter = max_iter
        self.tol = tol
        self.weights: np.ndarray | None = None
        self.daily_returns: np.ndarray | None = None

    @staticmethod
    def _solve_erc(cov: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
        """Maillard-Roncalli-Teiletche fixed-point iteration for ERC weights."""
        N = cov.shape[0]
        w = np.full(N, 1.0 / N)
        for _ in range(max_iter):
            mrc = cov @ w                         # marginal risk contributions (N,)
            # Target step: w_i ∝ 1 / mrc_i (so w_i · mrc_i is equalized).
            mrc_safe = np.where(mrc > 0, mrc, 1e-12)
            w_new = 1.0 / mrc_safe
            w_new = w_new / w_new.sum()
            if np.max(np.abs(w_new - w)) < tol:
                w = w_new
                break
            w = w_new
        return w

    def _shrink_cov(self, cov: np.ndarray) -> np.ndarray:
        if self.shrink <= 0.0:
            return cov
        N = cov.shape[0]
        target = (np.trace(cov) / N) * np.eye(N)
        return (1.0 - self.shrink) * cov + self.shrink * target

    def calc_erc_daily_rets(self) -> np.ndarray:
        num_windows = self.eval_returns.shape[0]
        weights_per_window = []
        daily_returns_per_window = []

        for i in range(num_windows):
            window = self.eval_returns[i]
            cov = np.cov(window, rowvar=False)
            cov = self._shrink_cov(cov)
            w = self._solve_erc(cov, self.max_iter, self.tol)
            weights_per_window.append(w)
            daily_returns_per_window.append(window @ w)

        self.weights = np.stack(weights_per_window, axis=0)
        self.daily_returns = np.stack(daily_returns_per_window)
        return self.daily_returns

    def get_weights(self) -> np.ndarray | None:
        return self.weights


def filter_models(
        avg_perf: pd.DataFrame, bench_name: str, bench_met: str, keep: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter out models that do not beat the benchmark (eg. Equal_Weight) and keep ones that do.

    Args:
        avg_perf (pd.DataFrame): Average Performance of all models across all metrics.
        bench_name (str): String name of the benchmark that exists in the avg_per dataframe.
        bench_met (str): String name of the metric that should be used to compare the models.
        keep (list[str]): List of all benchmarks or indexes to keep.
    
    Returns:
        tuple: A tuple containing,
            - filtered_avg_perf (pd.DataFrame): Dataframe containing only models that 
            outperformed the benchmark on the specified metric.
            - filtered_models (list[str]): List of names of the models that beat the benchmark.
    """

    # Get the equal‑weight Metric (Sharpe) value
    ew_sharpe = avg_perf.loc[bench_name, bench_met]

    # Create mask: keep if (1) it's a benchmark OR (2) its Sharpe > ew_sharpe
    mask = avg_perf.index.isin(keep) | (avg_perf[bench_met] > ew_sharpe)

    filtered_df = avg_perf[mask]

    filtered_models = filtered_df.index[
        ~filtered_df.index.isin(keep)
    ].to_list()

    return filtered_df, filtered_models