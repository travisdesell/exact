import warnings
import numpy as np
import pandas as pd
from typing import Callable

class Evaluator:
    """
    Class to evaluate and compare all generated weights from all models/methods,
    for all windows againsts each other as well as benchmarks (if provided).

    Attributes:
        spread_cost_factor (float): Spread cost factor is used to calculate single direction buy 
            transaction costs.
        rounding_digits (int): Round digits after the decimal point.
    """
    spread_cost_factor = 0.5
    rounding_digits = 4
    
    def __init__(
            self, 
            eval_returns: np.ndarray | None, 
            ba_eval: np.ndarray | None = None, 
            metrics_lib: dict[str, Callable] | None = None,
            all_daily_returns: dict[str, np.ndarray] | None = None
        ):
        """
        Initialize Evaluator to calculate the portfolio returns, account for Bid-Ask spread costs, 
        evaulate them and compare all generated weights with market benchmarks.

        Args:
            eval_windows (np.ndarray | None): Array of evaluation (out-of-sample)
                return windows for all stocks. Must provide array or None.
            ba_eval (np.ndarray | None): Array of Bid-Ask Spreads for each stock on the 
                first day of every window. This is used to calculate Bid-Ask Spread trading costs, 
                if provided. Default = None.
            metrics_lib (dict[str, Callable] | None): Metrics library dictionary containing metric 
                name and metric function. Default = None.
            all_daily_returns (dict[str, np.ndarray] | None): All daily returns for all provided 
                models/approaches.
                Must be provided if 'eval_windows' is not provided. Default = None.
        Raises:
            ValueError: If evaluation returns data does not have 3 dimensions.
            ValueError: If 'all_daily_returns' and 'eval_returns' is not provided.
        """

        if eval_returns is not None and isinstance(eval_returns, np.ndarray):
            if eval_returns.ndim != 3:
                raise ValueError(
                    f'ERROR: Evaluation Returns must have 3 dim, got {eval_returns.ndim}.'
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
                    'If out-of-sample evaluation data is not provided, daily returns of all ' \
                    'models must be provided.'
                )
            else:
                self.eval_returns = None
                self.ba_eval = None
                self.all_daily_returns = all_daily_returns

        self.metrics_lib = metrics_lib
    
    def _calc_step_ba_costs(
            self,
            prev_weights: np.ndarray | None, 
            curr_weights: np.ndarray,
            first_d_bas: np.ndarray
        ) -> np.float64:
        """
        Calculate the Bid-Ask Spread transaction cost for each walk step.
        
        Args:
            prev_weights (np.ndarray | None): Portfolio allocation weights from the previous step.
            curr_weights (np.ndarray): Portfolio allocation weights for the current step.
            first_d_bas (np.ndarray): Array of BA spreads, for all stocks, for the first day of 
                the rebalance period.
        
        Returns:
            cost (np.float64): Cost value for rebalancing on the first day of the rebalancing period.
        """
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
        """
        Calculate net returns by accounting for Bid-Ask spread cost for the first day of 
        rebalancing on the gross returns.

        Args:
            pf_daily_rets (np.ndarray): Gross daily returns for the rebalancing period.
            cost (float | np.float64): Cost value to be applied on the first day of rebalancing.
        
        Returns:
            pf_daily_rets (np.ndarray): Net returns afteraccounting for BA spread costs.
        """
        # Apply cost to the first day to get Net returns
        pf_daily_rets[0] = (1 + pf_daily_rets[0]) * (1 - cost) - 1

        return pf_daily_rets

    #### TODO: Continue docstring updates from here
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
            warnings.warn(f'Returns for {model_name} do not exist. Not updating.')
    
    def add_benchmark_rets(self, bench_name: str, bench_rets: np.ndarray):
        """
        Add benchmark returns for the respective evalulation output windows. eg., S&P500 daily returns
        """
        self.all_daily_returns.update({bench_name: bench_rets})

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
                model_rets.append(round(window_metric, self.rounding_digits))
            
            metric_perfomances[model] = model_rets
        if mean:
            return pd.DataFrame(metric_perfomances).mean()
        else:
            return pd.DataFrame(metric_perfomances)
    
    def calc_avg_performance(self) -> pd.DataFrame | None:
        """
        Calculates average portfolio performance metrics over all windows of the walk forward.
        This does not calculate actual overall performance of a portfolio for the entire data split period.
        
        Returns:
            avg_perf (pd.DataFrame | None): Average portfolio performances over all windows.
        """
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

    def _combine_rets_winds(self) -> dict[str, np.ndarray]:
        """
        Combine all windows to make one time series over entire validation period.
        
        Returns:
            combined_returns (dict[str, np.ndarray]): Dictionary containing the daily returns 
                for each model over the entire out-of-sample period.
        """
        combined_returns = {
            model: np.concatenate(arr) for model, arr in self.all_daily_returns.items()
        }

        return combined_returns

    def _calc_overall_metric_perf(
            self, metric_func: Callable, daily_rets: dict[str, np.ndarray]
        ) -> dict[str, np.float64]:

        metric_perfomances = {}
        for model, all_rets in daily_rets.items():
            metric_value = metric_func(all_rets)
            metric_perfomances[model] = round(metric_value, self.rounding_digits)
        
        return metric_perfomances
   
    def calc_pf_performances(self) -> pd.DataFrame | None:
        """
        Calculate the overall portfolio performances for all metrics by concatenating returns 
        from all windows and then calculating portfolio the portfolio performance metrics for one
        time series array of returns for the entire evaluation (out-of-sample) period.

        Returns:
            all_metric_perf (pd.DataFrame | None): Performance of each portfolio for the entire
                evaluation (out-of-sample) period.
        """
        self._daily_rets_calcd_check()
        if self.metrics_lib:
            
            # Combine all windows to make one time series over entire validation period.
            combined_returns = {
                model: np.concatenate(arr) for model, arr in self.all_daily_returns.items()
            }

            all_metric_perf = {}
            for met_name, met_func in self.metrics_lib.items():
                met_perf = self._calc_overall_metric_perf(met_func, combined_returns)
                all_metric_perf[met_name] = met_perf

            return pd.DataFrame(all_metric_perf)
        else:
            print(
                'No metrics library or dict provided. Cannot run portfolio performances over metrics.'
            )
            return None

    def get_all_daily_returns(self):
        """
        Get daily returns for all windows and all models
        
        Returns:
            all_daily_returns (dict[str, np.ndarray]): Dictionary containing the daily returns for 
                all windows and all models.
        """
        return self.all_daily_returns
    
    def update_spread_cost_factor(self, spread_cost_factor: float):
        """
        Update the Bid-Ask spread cost factor. Factor value must be < 1.0.

        Args:
            spread_cost_factor (float): New spread cost factor value.
        
        Raises:
            ValueError: Spread Cost factor cannot be greater than 1.
        """
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
        if self.eq_weights is not None:
            return self.eq_weights
        else:
            print(
                'WARNING: No equal weights calculated.',
                'Run `EqualWeightCalculator.calc_eq_wt_daily_rets()` first.'
            )
            return None