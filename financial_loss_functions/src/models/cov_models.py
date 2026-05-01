#### -------------------- All Covariance based Models (Classical/Tradional) -------------------- ####
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples
from src.models.registry import TradModelLibrary

# Try cvxopt; fallback to scipy
try:
    import cvxopt as opt
    from cvxopt import matrix, solvers
    CVXOPT_AVAILABLE = True
except Exception:
    CVXOPT_AVAILABLE = False


# ---------- Naive Minimum Variance Portfolio ---------- #
# Uses vectors of zeros for q_vec (expected returns)
@TradModelLibrary.register()
class NaiveMVP:
    @staticmethod
    def calculate_weights(cov: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Naive Implmentation of Minimum Variance Portfolio.

        Args:
            cov (pd.DataFrame | np.array): Covariance matrix of the returns.
        
        Returns:
            np.ndarray: Allocation weights of the portfolio, which sum to 1.
        """
        cov = np.array(cov)
        n = cov.shape[0]

        S = opt.matrix(cov)
        q = opt.matrix(np.zeros(n))  # no expected returns, pure variance minimization

        # Constraints: weights >= 0 and sum(weights) = 1
        G = -opt.matrix(np.eye(n))           # -I * x <= 0  ⟹ x >= 0
        h = opt.matrix(0.0, (n,1))
        A = opt.matrix(1.0, (1,n))           # sum weights = 1
        b = opt.matrix(1.0)

        solvers.options['show_progress'] = False
        sol = solvers.qp(S, q, G, h, A, b)

        return np.array(sol['x']).flatten()


# ---------- Base Quadratic Optimizer ---------- #
class BaseQuadraticOptimizer:
    """
    Shared quadratic-solver utilities.

    Solves problems of the form:
    minimize 0.5 x^T P x + q^T x
    s.t. A x = b   (equality)
    G x <= h   (inequality)
    bounds on x (optional)
    """

    def __init__(self, solver: str = 'auto', reg: float|str = 1e-8):
        """
        Shared quadratic-optimizer utilities initalizer.

        Args:
            solver (str): 'auto', 'cvxopt' or 'scipy'. Python module to be used for optimization.
                Default = 'auto', auto detects is cvxopt is avaiable or uses SciPy as a fallback.

            reg (float | str): small ridge added to diagonal of covariance to stabilize inversion.
                - If float >= 0: used as ridge added to diagonal of P.
                - If 'auto': ridge = eps * trace(P)/n where eps = 1e-8 (safe default).
        """
        self.solver = solver
        self.reg = reg
        self._cvx_available = CVXOPT_AVAILABLE # Flag for availability of cvxopt

    @staticmethod
    def _ensure_symmetry(mat: np.ndarray, constant: float = 0.5) -> np.ndarray:
        """
        Ensures symmetry of matrix by multiplying the sum of the matrix 
        and its transpose by a constant. constant * (mat + mat.T).

        Args:
            mat (np.ndarray): Matrix that will be transformed to ensure symmetry.
            constant (float) Constant value to be multiplied. Default = 0.5.

        Returns:
            np.ndarray: Symmetric matrix
        """
        mat = np.asarray(mat, dtype=float)
        return constant * (mat + mat.T)

    @staticmethod
    def _to_numpy(mat: pd.DataFrame) -> np.ndarray:
        """
        Converts matrix from dataframe to numpy array or enforces float if 
        already numpy array.

        Args:
            mat (pd.Dataframe): Matrix in a dataframe to be converted to a numpy array. 

        Returns:
            np.ndarray: Matrix as a numpy array.
        """
        if isinstance(mat, pd.DataFrame):
            return mat.values
        return np.asarray(mat, dtype=float)

    def _safe_inv(self, mat: np.ndarray) -> np.ndarray:
        """
        Numerically safe inverse that:
        - ensures symmetry
        - computes numeric ridge via _compute_ridge
        - returns inverse of (mat + ridge * I)

        Args:
            mat (np.ndarray): Matrix to be inverted safely.

        Returns:
            np.ndarray: Inverted matrix 
        """
        mat = self._ensure_symmetry(mat)
        ridge = self._compute_ridge(mat)
        # Always add the ridge: it's tiny and makes inversion stable and consistent.
        if ridge != 0.0:
            mat_r = mat + ridge * np.eye(mat.shape[0])
        else:
            mat_r = mat
        # Now inverting matrix
        return np.linalg.inv(mat_r)

    def _compute_ridge(self, P: np.ndarray) -> float:
        """
        Compute ridge value based on P, used to stabilize inversion of matrix.

        Args:
            P (np.ndarray): Matrix used to calculate the ridge value.
        Returns:
            float: Ridge value.
        """
        if isinstance(self.reg, str) and str(self.reg.lower()) == 'auto':
            # scale by matrix size and trace so ridge is relative to magnitude of P
            eps = 1e-8
            tr = float(np.trace(P))
            n = P.shape[0]
            # if trace is zero (degenerate), fallback to eps
            if tr == 0.0:
                return eps
            return eps * (tr / n)
        else:
            # ensure numeric
            return float(self.reg)

    def _qp_solve(
            self,
            P: np.ndarray,
            q: np.ndarray,
            A: np.ndarray | None = None,
            b: np.ndarray | None = None,
            G: np.ndarray | None = None,
            h: np.ndarray | None = None,
            bounds: tuple[tuple[float, float], ...] | None = None
        ) -> tuple[np.ndarray, bool]:
        """
        Solve Quadratic Problem using CVXOPT if available (and requested) else SciPy SLSQP.

        Args:
            P (np.ndarray): (n,n) symmetric positive semidef
            q (np.ndarray): (n,) vector
            A (np.ndarray | None): (m_eq, n) equality matrix
            b (np.ndarray | None): (m_eq,) equality RHS
            G (np.ndarray | None): (m_ineq, n) inequality matrix (G x <= h)
            h (np.ndarray | None): (m_ineq,) inequality RHS
            bounds: (tuple[tuple[float, float], ...] | None): tuple of (low, high) per variable or None

        Returns:
            x (tuple[np.ndarray, bool]): (n,), success (bool)
        """
        P = self._ensure_symmetry(P)
        q = np.asarray(q, dtype=float).flatten()
        n = P.shape[0]

        # compute and apply ridge
        ridge = self._compute_ridge(P)
        if ridge != 0.0:
            P = P + ridge * np.eye(n)

        # Check to use cvxopt or not
        use_cvx = (self.solver == 'cvxopt') or (
            self.solver == 'auto' and self._cvx_available
        )

        if use_cvx and self._cvx_available:
            # Build CVXOPT matrices (must be double, cvxopt.matrix)
            Pmat = matrix(P)
            qmat = matrix(q)
            Amat = matrix(A) if (A is not None) else None
            bmat = matrix(b) if (b is not None) else None
            Gmat = matrix(G) if (G is not None) else None
            hmat = matrix(h) if (h is not None) else None

            solvers.options["show_progress"] = False
            try:
                if Gmat is None:
                    sol = solvers.qp(Pmat, qmat, None, None, Amat, bmat)
                else:
                    sol = solvers.qp(Pmat, qmat, Gmat, hmat, Amat, bmat)
                x = np.array(sol["x"]).flatten()
                success = (('status' not in sol) or sol['status'] == 'optimal')
                return x, success
            except Exception as e:
                # fall back to scipy if cvxopt call fails
                print('Error with cvxopt, falling back to scipy.', e)
                pass

        # SciPy SLSQP fallback formulation: minimize 0.5 x' P x + q' x
        x0 = np.ones(n) / n

        cons = []
        if A is not None and b is not None:
            # equality constraints
            A = np.atleast_2d(A)
            b = np.asarray(b).flatten()
            # each row of A is equality: sum A_i * x = b_i
            for row, val in zip(A, b):
                cons.append(
                    {
                        'type': 'eq',
                        'fun': lambda x,
                        row=row,
                        val=val: float(np.dot(row, x) - val)
                    }
                )

        if G is not None and h is not None:
            G = np.atleast_2d(G)
            h = np.asarray(h).flatten()
            for row, val in zip(G, h):
                # inequality: row @ x <= val  =>  val - row @ x >= 0
                cons.append(
                    {
                        'type': 'ineq',
                        'fun': lambda x,
                        row=row,
                        val=val: float(val - np.dot(row, x))
                    }
                )

        # bounds as provided or None
        # objective
        def obj(x):
            return 0.5 * float(x @ P @ x) + float(np.dot(q, x))

        # scipy minimize
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"ftol": 1e-12, "maxiter": 2000})
        if not res.success:
            return (res.x if res.x is not None else x0), False
        return res.x, True

    def set_ridge(self, reg: float|str):
        """
        Setter function to set a small ridge to diagonal of
        covariance to stabilize inversion.
        
        Args:
            reg (float | str): small ridge added to diagonal of covariance to stabilize inversion.
        """
        self.reg = float(reg)

# ---------- Global Minimum Variance Portfolio ---------- #
@TradModelLibrary.register()
class GlobalMinimumVariance(BaseQuadraticOptimizer):
    """
    Global Minimum-Variance Portfolio estimator.
    """

    def __init__(
            self, allow_short: bool = False, solver: str = 'auto'
        ):
        """
        Initializer for GlobalMinimumVariance Portfolio. 
        Estimates portfolio allocation weights based on minimum varince of covariance of returns.

        Args:
            allow_short (bool):
                - If True, allow negative weights and use analytic formula w ∝ Σ^{-1} 1.
                - If False (default), enforce long-only and solve a QP.
            solver (str): 'auto', 'cvxopt' or 'scipy'. Python module to be used for optimization.
                Default = 'auto', auto detects is cvxopt is avaiable or uses SciPy as a fallback.
        """
        super().__init__(solver=solver)
        self.allow_short = bool(allow_short)

        # fitted attrs
        self.cov = None
        self.weights_ = None
        self.success_ = False

    def calculate_weights(self, cov: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Fit Global Minimum Variance to calculate portfolio allocation weights 
        using covariance matrix of returns.

        Args:
            cov (np.ndarray | pd.DataFrame): Covariance matrix of returns.

        Returns:
            weights (np.ndarray): Allocation weights of the portfolio, which sum to 1.
        """
        cov_mat = self._to_numpy(cov)
        self.cov = cov_mat

        n = cov_mat.shape[0]
        ones = np.ones(n)

        # analytic unconstrained solution
        if self.allow_short:
            inv = self._safe_inv(cov_mat)
            raw = inv @ ones
            w = raw / (ones @ raw)
            self.weights_ = w
            self.success_ = True
            return self.weights_

        # long-only: QP solve: minimize 0.5 x' Σ x  s.t. 1^T x = 1, x >= 0
        P = cov_mat
        q = np.zeros(n)
        A = ones.reshape(1, -1)
        b = np.array([1.0])
        G = -np.eye(n)  # -I x <= 0  =>  x >= 0
        h = np.zeros(n)
        bounds = tuple((0.0, 1.0) for _ in range(n))

        x, success = self._qp_solve(P=P, q=q, A=A, b=b, G=G, h=h, bounds=bounds)
        x = np.asarray(x, dtype=float)
        if not np.isclose(x.sum(), 1.0):
            if np.isclose(x.sum(), 0.0):
                x = np.ones_like(x) / n
            else:
                x = x / x.sum()
        self.weights_ = x
        self.success_ = bool(success)
        return self.weights_

    def get_weights(self) -> np.ndarray:
        """
        Getter function to get weights for a portfolio that have been 
        estimated by running `calculate_weights(...)`

        Returns:
            weights (np.ndarray): Allocation weights of the portfolio, which sum to 1.
        
        Raises:
            ValueError: Exception is raises since portfolio allocation weights 
                have not been calculated yet.
        """
        if self.weights_ is None:
            raise ValueError('Estimator not fit -  call `calculate_weights(...) first.`')
        return self.weights_


# ---------- Mean-Variance Portfolio (with internal expected-returns calc) ---------- #
@TradModelLibrary.register()
class MeanVariancePortfolio(BaseQuadraticOptimizer):
    """
    Mean-Variance Portfolio (Markowitz) that optionally computes expected returns.

    Solves:
    minimize  0.5 w^T Σ w  -  risk_aversion * μ^T w
    subject to: 1^T w = 1, w >= 0 (if allow_short=False)
    """
    def __init__(
            self,
            expected_returns_method: str|None = None,
            risk_aversion: float = 1.0,
            allow_short: bool = False,
            solver: str = 'auto',
        ):
        """
        Args:
            expected_returns_method (str | None): 'arithmetic' or 'geometric' can be used to 
                calculate the expected returns.

                - If None -> caller must pass expected_returns to calculate_weights().
                - If 'arithmetic' or 'geometric' -> caller must pass `returns` (obs x assets)
                    to calculate_weights() and μ will be computed from those returns.
                    
            risk_aversion (float): Risk aversion value for the estimation.
            allow_short (bool): Allow short strategy allocation weights. (-1 to 1)
            solver (str): 'auto', 'cvxopt' or 'scipy'. Python module to be used for optimization.
                Default = 'auto', auto detects is cvxopt is avaiable or uses SciPy as a fallback.
        
        Raises:
            ValueError: expected_returns_method must be None, 'arithmetic' or 'geometric'
        """
        super().__init__(solver=solver)
        if expected_returns_method is not None:
            method = expected_returns_method.lower()
            if method not in ('arithmetic', 'geometric'):
                raise ValueError("expected_returns_method must be None, 'arithmetic' or 'geometric'")
            self.expected_returns_method = method
        else:
            self.expected_returns_method = None

        self.risk_aversion = risk_aversion
        self.allow_short = allow_short

        # fitted attributes
        self.cov = None
        self.expected_returns_ = None
        self.weights_ = None
        self.success_ = False

    # expected returns calculators
    def _arith_mean_from_returns(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Per-period arithmetic mean (returns: DataFrame or 2D ndarray).
        
        Args:
            returns (pd.DataFrame): Daily returns of all stocks.

        Returns:
            np.ndarray: Array of arithmetic mean returns for all stocks.
        
        Raises:
            ValueError: Dimensions of the returns matric must be 2D.
        """
        if returns.ndim != 2:
            raise ValueError('returns must be 2D (obs x assets)')
        return np.nanmean(returns, axis=0)

    def _geom_mean_from_returns(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Geometric mean per-period using log1p to avoid overflow and handle NaNs:
        gm = exp(mean(log1p(returns))) - 1.

        Args:
            returns (pd.DataFrame): Daily returns of all stocks.

        Returns:
            gm (np.ndarray): Array of geometric mean returns for all stocks
        
        Raises:
            ValueError: Dimensions of the returns matric must be 2D.
        """
        if returns.ndim != 2:
            raise ValueError('returns must be 2-D (obs x assets)')
        # compute mean of logs ignoring NaNs
        with np.errstate(divide='ignore', invalid='ignore'):
            log1p = np.log1p(returns)
            mean_log = np.nanmean(log1p, axis=0)
            gm = np.expm1(mean_log)  # exp(mean_log)-1
        return gm

    def calculate_weights(
            self,
            cov: np.ndarray | pd.DataFrame,
            returns: np.ndarray|pd.DataFrame|None = None,
            expected_returns: np.ndarray|None = None
        ) -> np.ndarray:
        """
        Compute mean-variance portfolio allocation weights. 
        Either `returns` or `expected_returns` is required

        Args:
            cov (np.ndarray | pd.Dataframe): (n,n) Covariance matrix of returns.
            returns (np.ndarray | pd.DataFrame | None): 
                (obs, n) - used to compute expected_returns if the constructor 
                set expected_returns_method to 'arithmetic' or 'geometric'
            expected_returns (np.ndarray | None): (n,) if provided it will be used directly

        Returns:
            weights (np.ndarray): Allocation weights of the portfolio, which sum to 1.
        """
        cov_mat = self._to_numpy(cov)
        self.cov = cov_mat
        
        n = cov_mat.shape[0]
        ones = np.ones(n)

        # Determine expected returns vector mu
        if expected_returns is not None:
            mu = np.asarray(expected_returns, dtype=float).flatten()
        else:
            if self.expected_returns_method is None:
                raise ValueError(
                    'expected_returns not provided and expected_returns_method is None.'
                    'Either pass expected_returns or set expected_returns_method in constructor.'
                )
            
            # expected_returns_method is set -> returns must be provided
            if returns is None:
                raise ValueError(
                    'returns must be provided to compute expected returns when expected_returns_method is set.'
                )
            # allow pandas DataFrame or ndarray
            if not isinstance(returns, np.ndarray):
                try:
                    # If it's a DataFrame-like with .values
                    returns_arr = np.asarray(returns)
                except Exception:
                    raise ValueError('returns must be array-like (obs x assets)')
            else:
                returns_arr = returns

            if self.expected_returns_method == 'arithmetic':
                mu = self._arith_mean_from_returns(returns_arr)
            elif self.expected_returns_method == 'geometric':  # geometric
                mu = self._geom_mean_from_returns(returns_arr)

        mu = np.asarray(mu, dtype=float).flatten()
        if mu.shape[0] != n:
            raise ValueError('expected_returns length does not match covariance dimension')

        self.expected_returns_ = mu

        # build linear term q such that the QP is 0.5 w^T Σ w + q^T w
        # we want min 0.5 w'Σw - gamma * mu' w  =>  q = - gamma * mu
        q = - self.risk_aversion * mu

        # If shorting allowed -> analytic closed-form (use Lagrange multiplier to enforce sum=1)
        if self.allow_short:
            Sigma_inv = self._safe_inv(self.cov)
            a = ones @ (Sigma_inv @ ones)
            b = ones @ (Sigma_inv @ mu)
            gamma = float(self.risk_aversion)
            lam = (gamma * b - 1.0) / a
            x = Sigma_inv @ (gamma * mu - lam * ones)
            x = np.asarray(x, dtype=float)
            # numerical normalization just in case
            if not np.isclose(x.sum(), 1.0):
                if np.isclose(x.sum(), 0.0):
                    x = np.ones(n) / n
                else:
                    x = x / x.sum()
            self.weights_ = x
            self.success_ = True
            return self.weights_

        # Otherwise enforce long-only by QP: 1^T x = 1, x >= 0
        P = self.cov
        A = ones.reshape(1, -1)
        b = np.array([1.0])
        G = -np.eye(n)
        h = np.zeros(n)
        bounds = tuple((0.0, 1.0) for _ in range(n))

        x, success = self._qp_solve(P=P, q=q, A=A, b=b, G=G, h=h, bounds=bounds)
        x = np.asarray(x, dtype=float)

        # numerical normalization
        if not np.isclose(x.sum(), 1.0):
            if np.isclose(x.sum(), 0.0):
                x = np.ones_like(x) / n
            else:
                x = x / x.sum()

        self.weights_ = x
        self.success_ = bool(success)
        return self.weights_

    def get_weights(self) -> np.ndarray:
        """
        Getter function to get weights for a portfolio that have been 
        estimated by running `calculate_weights(...)`

        Returns:
            weights (np.ndarray): Allocation weights of the portfolio, which sum to 1.
        """
        if self.weights_ is None:
            raise ValueError('Estimator not fit - call `calculate_weights(...)` first.')
        return self.weights_

    def get_expected_returns(self) -> np.ndarray:
        """
        Getter function to get expected returns for a portfolio that have been 
        estimated during running of `calculate_weights(...)`.

        Returns:
            expected_returns (np.ndarray): Array of expected returns for all stocks.
        """
        if self.expected_returns_ is None:
            raise ValueError('Estimator not fit -  call `calculate_weights(...) first.`')
        return self.expected_returns_.copy()


# ---------- Hierarchial Risk Parity Clustering ---------- #
@TradModelLibrary.register()
class HierarchialRiskParity:
    """
    Implementation of Hierarchial Risk Parity Clustering.
    """
    def __init__(self, linkage: str = 'single'):
        """
        Initialize Hierarchial Risk Parity Clustering using given hyperparameters.

        Args:
            linkage (str): Linkage method to be used for hierarchial clustering. 
                'single', 'average', 'complete', 'ward', 'centroid','mean' or 'median'.
                Default = 'single'. 
        """
        self.linkage = linkage

        self.weights = None
    
    def _correlDist(self, corr: pd.DataFrame) -> pd.DataFrame:
        """
        Compute correlation distance of correlation matrix.

        Args:
            corr (pd.DataFrame): Correlation matrix of returns.

        Returns:
            pd.DataFrame: Correlation distance matrix.
        """
        # A distance matrix based on correlation, where 0<=d[i,j]<=1
        # This is a proper distance metric
        dist = ((1 - corr) / 2.) ** .5

        # Force symmetry
        dist = (dist + dist.T) / 2  
        np.fill_diagonal(dist.values, 0)  # ensure diagonal is 0
        return dist

    def _getQuasiDiag(self, link: np.ndarray) -> list:
        """
        Compute Quasi Diagonal from clustered items and sort them.

        Args:
            link (np.ndarray): Custered link from a hierarchial custering method.

        Returns:
            list: List of Sorted indexes of the clustered items.
        """
        # Sort clustered items by distance
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3]  # number of original items
        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
            df0 = sortIx[sortIx >= numItems]  # find clusters
            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sortIx =pd.concat([sortIx,df0])  # item 2
            sortIx = sortIx.sort_index()  # re-sort
            sortIx.index = range(sortIx.shape[0])  # re-index
        return sortIx.tolist()

    def _getIVP(self, cov: pd.DataFrame, **kargs) -> np.ndarray:
        """
        Compute the inverse-variance portfolio
        
        Args:
            cov (pd.DataFrame): Covariance matrix
        
        Returns:
            ivp (np.ndarray): Inverse variance portfolio.
        """
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def _getClusterVar(self, cov: pd.DataFrame, cItems: list) -> np.ndarray:
        """
        Compute intra cluster variance.
        Cluster is idenfied from the entire cov matrix using the procided indexes.

        Args:
            cov (pd.DataFrame): Covariance matrix
            cItems (list): Items belonging a particular cluster

        Returns:
            cVar (np.ndarray): Cariance of a particular cluster.
        """
        # Compute variance per cluster
        cov_=cov.loc[cItems,cItems]
        w_= self._getIVP(cov_).reshape(-1,1)
        cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
        return cVar

    def _getRecBipart(self, cov: pd.DataFrame, sortIx: list) -> pd.Series:
        """
        Compute HRP allocation using Risk Parity using intra cluster variance.

        Args:
            cov (pd.DataFrame): Covariance matrix.
            sortIx (list): Sorted indexes of clustered items.

        Returns:
            w (pd.Series): Portfolio allocation weights based on Risk Parity
        """
        # Compute HRP alloc
        w = pd.Series(1.0, index=sortIx)
        cItems = [sortIx]  # initialize all items in one cluster
        while len(cItems) > 0:
            cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), 
                (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
            for i in range(0, len(cItems), 2):  # parse in pairs
                cItems0 = cItems[i]  # cluster 1
                cItems1 = cItems[i + 1]  # cluster 2
                cVar0 = self._getClusterVar(cov, cItems0)
                cVar1 = self._getClusterVar(cov, cItems1)
                alpha = 1 - cVar0 / (cVar0 + cVar1)
                w[cItems0] *= alpha  # weight 1
                w[cItems1] *= 1 - alpha  # weight 2
        return w

    def calculate_weights(
            self, cov: pd.DataFrame, corr: pd.DataFrame
        ) -> pd.Series:
        """
        Hierachial Risk Parity Clustering for portfolio optimization using covariance 
        and correlation matrices of returns.
        
        Args:
            cov (pd.DataFrame): Covariance matrix of returns.
            corr (pd.DataFrame): Correlation matrix of returns.
        
        Returns 
            weights (pd.Series): optimized portfolio allocation weights, which sum to 1.
        """
        # Construct a hierarchical portfolio
        if len(cov) > 1:
            dist = self._correlDist(corr)
            condensed_dist = squareform(dist)
            link = linkage(condensed_dist, self.linkage)

            sortIx = self._getQuasiDiag(link)
            sortIx = corr.index[sortIx].tolist()
            hrp = self._getRecBipart(cov, sortIx)
        else:
            hrp = pd.Series(1.0, index=cov.index)
        
        self.weights = hrp.sort_index()
        return self.weights
    
    def get_weights(self) -> pd.Series:
        """
        Getter function to get weights for a portfolio that have been 
        estimated by running `calculate_weights(...)`.

        Returns:
            weights (pd.Series): Allocation weights of the portfolio, which sum to 1.
        
        Raises:
            ValueError: Exception is raises since portfolio allocation weights 
                have not been calculated yet.
        """
        if self.weights is None:
            raise ValueError('Estimator not fit -  call `calculate_weights(...) first.`')
        return self.weights

@TradModelLibrary.register()
class NestedClusteredOptimization():
    """Implementation of Nested Clustered Optimization"""
    def __init__(self, de_noise: bool = True):
        """
        Initialize Implmentation of Nested Clustered Optimization which uses 
        Global Minimum Variance Optimization for inter cluster and intra cluster
        optimization.
        
        Args:
            de_noise (bool): Apply de noising to covariance matrix. Default = True.
        """
        self.optimizer = GlobalMinimumVariance() # To use different algo, change here and 
                                                # returns are available in self.calculate_weights.
        self.de_noise = de_noise

        self.weights = None
    
    def _cov2corr(self, cov: np.ndarray) -> np.ndarray:
        """
        Derive correlation matrix from covariance matrix.

        Args:
            cov (np.ndarray): Covariance matrix.
        
        Returns:
            corr (np.ndarray): Correlation Matrix.
        """
        std = np.sqrt(np.diag(cov))
        corr = cov/np.outer(std,std)
        corr[corr<-1], corr[corr>1] = -1, 1 # numerical error
        return corr

    def _getPCA(self, matrix) -> tuple[np.ndarray, np.ndarray]:
        """
        Get eVal, eVec from a Hermitian matrix.
        
        Args:
            matrix (np.ndarray): Hermitian matrix to do PCA on.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: eVal & eVec.
        """
        eVal, eVec = np.linalg.eigh(matrix)
        indices = eVal.argsort()[::-1] # args for sorting eVal desc
        eVal, eVec = eVal[indices], eVec[:,indices]
        eVal = np.diagflat(eVal)
        return eVal, eVec

    def _mpPDF(self, var, q, pts):
        """
        Calculate Marcenko-Pastur pdf, q=T/N, to determine signal to 
        noise ratio to shrink covariance matrix for denoising.
        """
        eMin, eMax = var * (1-(1./q)**.5)**2, var*(1+(1./q)**.5)**2
        eVal = np.linspace(eMin, eMax, pts)
        pdf = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
        pdf2 = pd.Series(
            pdf.reshape(pdf.shape[0],),
            index=eVal.reshape(eVal.shape[0],)
        )
        return pdf2

    def _fitKDE(self, obs, bWidth=.25, kernel='gaussian', x=None):
        """
        Fit kernel density to a series of obs, and derive the prob of obs
        # x is the arraymof values on which the fit KDE will be evaluated
        """
        if len(obs.shape)==1: obs=obs.reshape(-1,1)
        kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
        if x is None: x=np.unique(obs).reshape(-1,1)
        if len(x.shape)==1: x=x.reshape(-1,1)
        logProb = kde.score_samples(x) # log (density)
        pdf = pd.Series(np.exp(logProb), index=x.flatten())
        return pdf

    def _errPDFs(self, var, eVal, q, bWidth, pts=1000):
        """Fit error"""
        pdf0 = self._mpPDF(var, q, pts) # theoretical pdf
        pdf1 = self._fitKDE(eVal, bWidth, x=pdf0.index.values) # empirical pdf
        sse = np.sum((pdf1-pdf0) ** 2)
        return sse 

    def _findMaxEval(self, eVal, q, bWidth):
        """Find max random eVal by fitting Marcenko's dist"""
        out = minimize(
            lambda *x: self._errPDFs(*x), .5,args=(eVal, q, bWidth),
            bounds=((1E-5, 1-1E-5),)
        )
        if out['success']: var = out['x'][0]
        else: var = 1
        eMax = var * (1+(1./q)) ** 2
        return eMax, var

    def _denoisedCorr(self, eVal, eVec, nFacts):
        """Remove noise from corr by fixing random eigenvalues"""
        eVal_ = np.diag(eVal).copy()
        eVal_[nFacts:]=eVal_[nFacts].sum()/float(eVal_.shape[0]-nFacts)
        eVal_ = np.diag(eVal_)
        corr1 = np.dot(eVec, eVal_).dot(eVec.T)
        corr1 = self._cov2corr(corr1)
        return corr1
    
    def _corr2cov(self, corr, std):
        cov = corr * np.outer(std, std)
        return cov
    
    def _deNoiseCov(self, cov0, q, bWidth):
        """
        Denoise covariance matrix using signal to noise ratio from Marcenko-Pastur pdf. 
        Shrink and reconstructs matrix after shrinkage. 
        """
        corr0=self._cov2corr(cov0)
        eVal0, eVec0 = self._getPCA(corr0)
        eMax0, var0 = self._findMaxEval(np.diag(eVal0), q, bWidth)
        nFacts0 = eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
        corr1 = self._denoisedCorr(eVal0, eVec0, nFacts0)
        cov1 = self._corr2cov(corr1, np.diag(cov0) ** .5)
        return cov1

    def _de_noise(self, cov: pd.DataFrame, T: int, N: int) -> pd.DataFrame:
        """
        De Noising of covariance matrix of returns of all stocks.

        Args:
            cov (pd.DataFrame): Covaraince matrix of returns.
            T (int): Number of time steps or rows.
            N (int): Number of assets (stocks) or columns.
        
        Returns:
            cov (pd.DataFrame): Denoised covariance matrix.
        """
        cols = cov.columns
        q = T/N
        cov = self._deNoiseCov(cov, q, bWidth=0.1)
        cov = pd.DataFrame(cov, index=cols, columns=cols)
        return cov

    def _clusterKMeansBase(self, corr0, maxNumClusters=10, n_init=10):
        x = ((1-corr0.fillna(0))/2.)**.5
        silh = pd.Series(dtype=float) # observation matrix
        kmeans = None
        
        if maxNumClusters > 1 and corr0.shape[0] > 2:
            for init in range(n_init):
                for i in range(2, maxNumClusters+1):
                    kmeans_= KMeans(n_clusters=i, n_init=1)
                    kmeans_ = kmeans_.fit(x)
                    silh_ = silhouette_samples(x, kmeans_.labels_)
                    stat = (silh_.mean()/silh_.std(), silh.mean()/silh.std())
                    if np.isnan(stat[1]) or stat[0]>stat[1]:
                        silh, kmeans = silh_, kmeans_
        
        # FALLBACK: If no clustering happened, treat everyone as one cluster
        if kmeans is None:
            clstrs = {0: corr0.columns.tolist()}
            silh = pd.Series(0.0, index=x.index)
            return corr0, clstrs, silh
        
        newIdx = np.argsort(kmeans.labels_)
        corr1 = corr0.iloc[newIdx] # reorder rows

        corr1=corr1.iloc[:,newIdx] # reorder columns
        clstrs = {
            i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist() 
            for i in np.unique(kmeans.labels_)
                
        } # cluster members
        silh = pd.Series(silh, index=x.index)
        return corr1, clstrs, silh
    
    def _calc_nco(self, cov, mu=None, maxNumClusters=None):
        # cov = pd.DataFrame(cov)
        # if mu is not None: mu = pd.Series(mu[:,0])
        if mu is not None:
            mu = pd.Series(mu.flatten(), index=cov.index)
        
        corr1 = self._cov2corr(cov)
        corr1, clstrs, _ = self._clusterKMeansBase(corr1, maxNumClusters, n_init=10)
        
        wIntra = pd.DataFrame(0.0, index = cov.index, columns=clstrs.keys())
        for i in clstrs:
            cov_ = cov.loc[clstrs[i], clstrs[i]]
            if mu is None: mu_=None
            else: mu_ = mu.loc[clstrs[i]].values.reshape(-1,1)
            wIntra.loc[clstrs[i], i] = self.optimizer.calculate_weights(cov_)
        cov_ = wIntra.T.dot(np.dot(cov, wIntra)) # reduce covariance matrix
        mu_ = (None if mu is None else wIntra.T.dot(mu))
        wInter = pd.Series(self.optimizer.calculate_weights(cov_), index=cov_.index)
        nco = wIntra.mul(wInter, axis=1).sum(axis=1)
        
        return nco
    
    def calculate_weights(
            self, cov: pd.DataFrame, returns: pd.DataFrame
        ) -> pd.Series:
        """
        Fit NCO model to given covariance matrix.

        Args:
            cov (pd.DataFrame): covariance matrix of returns.
            returns (pd.DataFrame): asset (stocks) returns.
        
        Returns:
            weights (pd.Series): Optimized portfolio allocation weights. 
        """
        if self.de_noise:
            cov = self._de_noise(cov, T=returns.shape[0], N=returns.shape[1])
        
        self.weights = self._calc_nco(
            cov=cov,
            maxNumClusters=int(cov.shape[0]/2)
        )

        return self.weights