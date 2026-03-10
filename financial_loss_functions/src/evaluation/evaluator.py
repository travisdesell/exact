import numpy as np
import pandas as pd
from typing import Callable

class Evaluator:
    """
    Class to evaulate and compare all generated weights from all models/methods,
    for all windows againsts each other as well as benchmarks.
    """
    def __init__(self, eval_returns: np.ndarray, metrics_lib: dict|None=None):
        """
        Initialize Evaluator instance to evaulate and compare all generated weights.

        @param eval_returns np.ndarray
            Daily returns which are used to evulate all methods/models
        """
        # Returns by window
        self.eval_returns = eval_returns
        if self.eval_returns is None:
            raise ValueError('Evaluation returns is None.')
        
        self.metrics_lib = metrics_lib

        # Different Weights
        self.eq_weights = None
        
        # Returns for each window
        self.all_daily_returns = {} # Add all returns for every window

    @staticmethod
    def _equal_weight_pf(num_tickers: int) -> np.ndarray:
        """
        Calculates simple equal weights for a portfolio
        weight for each stock = 1/num_tickers
        
        @param num_tickers int number of tickers in the dataset

        @return np.array equal weight portfolio allocation weights
        """
        return np.full((num_tickers), 1/num_tickers)

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
        
        # Iterating over window samples
        for i in range(eval_weights.shape[0]):
            weights = eval_weights[i]  # Shape: (50,)
            returns = self.eval_returns[i]  # Shape: (50, 50) - time steps x assets
            
            # Calculate daily portfolio returns (dot product at each time step)
            daily_returns = np.dot(returns, weights)
            pf_daily_returns.append(daily_returns) # Shape: (50,)
            
        self.all_daily_returns[model_name] = np.array(pf_daily_returns)
    
    def add_benchmark_rets(self, bench_name: str, bench_rets: np.ndarray):
        """
        Add benchmark returns for the respective evalulation output windows. eg., S&P500 daily returns
        """
        self.all_daily_returns[bench_name] = bench_rets

    def _daily_rets_calcd_check(self):
        if not self.all_daily_returns:
            raise ValueError(
                'No daily returns calculated.',
                'Run calc_pf_daily_rets and calc_eq_wt_daily_rets first.'
            )

    def calc_eq_wt_daily_rets(self): 
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

        self.all_daily_returns['Equal Weight'] = np.array(eq_wt_daily_returns)

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