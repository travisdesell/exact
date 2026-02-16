import numpy as np
import pandas as pd
import src.evaluation.metrics as metrics

class Evaluator:
    """
    Class to evaulate and compare all generated weights from all models/methods,
    for all windows againsts each other as well as benchmarks.
    """
    def __init__(self, eval_returns: np.ndarray):
        """
        Initialize Evaluator instance to evaulate and compare all generated weights.

        @param eval_returns np.ndarray
            Daily returns which are used to evulate all methods/models
        """
        # Returns by window
        self.eval_returns = eval_returns

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

    def calc_total_performance(self, metric: str) -> pd.DataFrame:
        """
        Calculate per-window performance of all portfolios (incl. Equal Weight)
        based on given metric. 

        @param metric str
            String name of the metric to be calculated. `returns` or `sharpe`

        @return Dict[str, list]
            Dictionary containing calculated performance metric for each validation window
        """
        self._daily_rets_calcd_check()
        
        total_perfomances = {}
        for model, all_rets in self.all_daily_returns.items():
            model_rets = []
            for i in range(all_rets.shape[0]):
                if metric == 'returns':
                    window_metric = metrics.cumulative_return(all_rets[i])
                elif metric == 'sharpe':
                    window_metric = metrics.basic_sharpe(all_rets[i])
                
                model_rets.append(round(window_metric, 4))
            
            total_perfomances[model] = model_rets
        
        return pd.DataFrame(total_perfomances)

    def get_all_daily_returns(self):
        return self.all_daily_returns