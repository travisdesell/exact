#### -------------------- All Covariance based Models (Classical) -------------------- ####

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from src.models.registry import TradModelLibrary

# Try cvxopt; fallback to scipy
try:
    import cvxopt as opt
    from cvxopt import matrix, solvers
    CVXOPT_AVAILABLE = True
except Exception:
    CVXOPT_AVAILABLE = False

from scipy.optimize import minimize


# ---------- Naive Minimum Variance Portfolio ---------- #
# Uses vectors of zeros for q_vec (expected returns)
@TradModelLibrary.register()
class NaiveMVP:
    @staticmethod
    def calculate_weights(cov: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Naive Implmentation of Minimum Variance Portfolio

        @param cov (pd.DataFrame | np.array) Covariance matrix of the returns
        
        @return np.ndarray Weights of the portfolio
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
        @param solver str 'auto' | 'cvxopt' | 'scipy'

        @param reg float | 'auto'
            small ridge added to diagonal of covariance to stabilize inversion.
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
        and its transpose by a constant. constant * (mat + mat.T)

        @param mat np.ndarray Matrix
        @param constant float Constant value to be multiplied. Default = 0.5

        @return np.ndarray Symmetric matrix
        """
        mat = np.asarray(mat, dtype=float)
        return constant * (mat + mat.T)

    @staticmethod
    def _to_numpy(mat: np.ndarray) -> np.ndarray:
        """
        Converts matrix from dataframe to numpy array or enforces float if 
        already numpy array.

        @param mat np.ndarray Matrix

        @return np.ndarray
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

        @param mat np.ndarray Matrix to be inverted safely

        @retun np.ndarray Inverted matrix 
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

        @param P np.ndarray Matrix used to calculate the ridge value
        @return float Ridge value
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

        @param P np.ndarray (n,n) symmetric positive semidef
        @param q np.ndarray (n,) vector
        @param A (np.ndarray | None) (m_eq, n) equality matrix
        @param b (np.ndarray | None) (m_eq,) equality RHS
        @param G (np.ndarray | None) (m_ineq, n) inequality matrix (G x <= h)
        @param h (np.ndarray | None) (m_ineq,) inequality RHS
        @param bounds: (Tuple[Tuple[float, float], ...] | None) tuple of (low, high) per variable or None

        @return x Tuple[np.ndarray, bool] (n,), success (bool)
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
        covariance to stabilize inversion
        
        @param reg: (float|str) small ridge added to diagonal of covariance to stabilize inversion
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
        @param allow_short bool
            If True, allow negative weights and use analytic formula w ∝ Σ^{-1} 1.
            If False (default), enforce long-only and solve a QP.
        @param solver str ('auto'|'cvxopt'|'scipy') 
            Solver library to use. Checks if cvxopt is available by default
            (passed to BaseQuadraticOptimizer).
        """
        super().__init__(solver=solver)
        self.allow_short = bool(allow_short)

        # fitted attrs
        self.cov = None
        self.weights_ = None
        self.success_ = False

    def calculate_weights(self, cov: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Fit Global Minimum Variance to calculate portfolio allocation weights.

        @param cov (np.ndarray | pd.DataFrame) Covariance matrix

        @return np.ndarray Calculated allocation weights. Shape = (n,)
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
        return self.weights_.copy()

    def get_weights(self) -> np.ndarray:
        """
        Getter function to get weights for a portfolio that have been 
        estimated by running `calculate_weights(...)`

        @return np.ndarray Array of allocation weights
        """
        if self.weights_ is None:
            raise ValueError('Estimator not fit -  call `calculate_weights(...) first.`')
        return self.weights_.copy()


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
        @param expected_returns_method (None | 'arithmetic' | 'geometric')
            If None -> caller must pass expected_returns to calculate_weights().
            If 'arithmetic' or 'geometric' -> caller must pass `returns` (obs x assets)
               to calculate_weights() and μ will be computed from those returns.
        @param risk_aversion float
        @param allow_short bool
        @param solver : str ('auto'|'cvxopt'|'scipy')
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
        
        @param returns pd.DataFrame Returns of all stocks

        @return np.ndarray Array of arithmetic mean returns for all stocks
        """
        if returns.ndim != 2:
            raise ValueError("returns must be 2-D (obs x assets)")
        return np.nanmean(returns, axis=0)

    def _geom_mean_from_returns(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Geometric mean per-period using log1p to avoid overflow and handle NaNs:
        gm = exp(mean(log1p(returns))) - 1

        @param returns pd.DataFrame Returns of all stocks

        @return np.ndarray Array of geometric mean returns for all stocks
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
            cov: np.ndarray,
            returns: np.ndarray|None = None,
            expected_returns: np.ndarray|None = None
        ) -> np.ndarray:
        """
        Compute mean-variance weights. Either returns or expected_returns is required

        @param cov (n,n) covariance matrix
        @param returns (obs, n) optional - used to compute expected_returns if the constructor
                  set expected_returns_method to 'arithmetic' or 'geometric'
        @param expected_returns (n,) optional - if provided it will be used directly

        @return np.ndarray Calculated allocation weights. Shape = (n,)
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
        return self.weights_.copy()

    def get_weights(self) -> np.ndarray:
        """
        Getter function to get weights for a portfolio that have been 
        estimated by running `calculate_weights(...)`

        @return np.ndarray Array of allocation weights
        """
        if self.weights_ is None:
            raise ValueError('Estimator not fit - call `calculate_weights(...)` first.')
        return self.weights_.copy()

    def get_expected_returns(self) -> np.ndarray:
        """
        Getter function to get expected returns for a portfolio that have been 
        estimated during running of `calculate_weights(...)`

        @return np.ndarray Array of expected returns
        """
        if self.expected_returns_ is None:
            raise ValueError('Estimator not fit -  call `calculate_weights(...) first.`')
        return self.expected_returns_.copy()


# ---------- Hierarchial Risk Parity Clustering ---------- #
@TradModelLibrary.register()
class HierarchialRiskParity:
    """
    Implementation of Hierarchial Risk Parity Clustering
    """
    def __init__(self, linkage: str = 'single'):
        """
        Initialize Hierarchial Risk Parity Clustering using given hyperparameters.

        @param linkage str
            Linkage method to be used for hierarchial clustering. 'single', 'average',
            'complete', 'ward', 'centroid','mean' or 'median'. 
        """
        self.linkage = linkage

        self.weights = None
    
    def _correlDist(self, corr: pd.DataFrame) -> pd.DataFrame:
        """
        Compute correlation distance of correlation matrix.

        @param corr pd.DataFrame Correlation matrix

        @return pd.DataFrame Correlation distance matrix
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

        @param link np.ndarray cCustered link from a hierarchial custering method

        @return list Sorted index of the clustered items
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
        @param cov pd.DataFrame Covariance matrix
        
        @return np.ndarray Inverse variance portfolio
        """
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def _getClusterVar(self, cov: pd.DataFrame, cItems: list) -> np.ndarray:
        """
        Compute intra cluster variance.
        Cluster is idenfied from the entire cov matrix using the procided indexes.

        @param cov pd.DataFrame Covariance matrix
        @param cItems list Items belonging a particular cluster

        @return np.ndarray variance of a particular cluster

        """
        # Compute variance per cluster
        cov_=cov.loc[cItems,cItems]
        w_= self._getIVP(cov_).reshape(-1,1)
        cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
        return cVar

    def _getRecBipart(self, cov: pd.DataFrame, sortIx: list) -> pd.Series:
        """
        Compute HRP allocation using Risk Parity using intra cluster variance.

        @param cov pd.DataFrame Covariance matrix
        @param sortIx list Sorted indexes of clustered items

        @return pd.Series Portfolio allocation weights based on Risk Parity
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
        Hierachial Risk Parity Clustering for portfolio optimization.
        
        @param cov pd.DataFrame
                covariance matrix of returns
        @param corr pd.DataFrame
                correlation matrix of returnsx
        
        @return weights pd.Series
                optimized weights for the portfolio out of 1 (not 100)
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
        return self.weights.copy()
    
    def get_weights(self):
        """
        Getter function to get weights for a portfolio that have been 
        estimated by running `calculate_weights(...)`
        """
        if self.weights is None:
            raise ValueError('Estimator not fit -  call `calculate_weights(...) first.`')
        return self.weights.copy()