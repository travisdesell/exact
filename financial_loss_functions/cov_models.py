import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from pandas.tseries.offsets import BDay
from scipy.cluster.hierarchy import linkage
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples


import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers

import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers

def getMVP(cov):
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


class HRP:
    """
    Implementation of Hierarchial Risk Parity Clustering
    """
    def __init__(self, linkage: str = 'single'):
        """
        Initialize Hierarchial Risk Parity Clustering using given hyperparameters.

        Paramaters
        ----------
        linkage : str
            Linkage method to be used for hierarchial clustering. 'single', 'average', 'complete',
            'ward', 'centroid','mean' or 'median'. 
        """
        self.linkage = linkage
    
    def _correlDist(self, corr):
        # A distance matrix based on correlation, where 0<=d[i,j]<=1
        # This is a proper distance metric
        dist = ((1 - corr) / 2.) ** .5

        # Force symmetry
        dist = (dist + dist.T) / 2  
        np.fill_diagonal(dist.values, 0)  # ensure diagonal is 0
        return dist

    def _getQuasiDiag(self, link):
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

    def _getIVP(self, cov, **kargs):
        # Compute the inverse-variance portfolio
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def _getClusterVar(self, cov, cItems):
        # Compute variance per cluster
        cov_=cov.loc[cItems,cItems]
        w_= self._getIVP(cov_).reshape(-1,1)
        cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
        return cVar

    def _getRecBipart(self, cov, sortIx):
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

    def _get_hrp(self, cov, corr):
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
        return hrp.sort_index()
    
    def fit(self, cov: pd.DataFrame, corr: pd.DataFrame) -> pd.Series:
        """
        Hierachial Risk Parity Clustering for portfolio optimization.
        
        Parameters 
        ----------
            cov : pd.DataFrame
                covariance matrix of returns
            corr : pd.DataFrame
                correlation matrix of returnsx
        Returns
        -------
            weights : pd.Series
                optimized weights for the portfolio out of 1 (not 100)
        """
        return self._get_hrp(cov, corr)