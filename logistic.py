# logistic.py
# This file implements the position-dependent start probability from logistic regression:
#     p_start[t] = sigmoid(beta0 + beta1 * N_asin[t]
import numpy as np


class StartLogitModel:
    """
    Position-dependent start probability from logistic regression:
        p_start[t] = sigmoid(beta0 + beta1 * N_asin[t] + beta2 * range_score[t])
    """
    def __init__(self, beta0,beta1,beta2,eps=1e-12):
        self.beta0 = float(beta0)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = eps

    
    @staticmethod
    def logistic(x):
        return 1.0/(1.0+np.exp(-x))

    @staticmethod
    def compute_start_prob(triplet_df, beta0, beta1, beta2,eps=1e-12):
        """
        Compute start probability from a logistic regression trained in R:

            P_t = sigmoid(beta0 + beta1 * N_asin_t + beta2 * range_score_t)

        Parameters
        ----------
        triplet_df : pandas.DataFrame
            Must contain columns ['N_asin', 'range_score'].
        beta0, beta1, beta2 : float
            Logistic regression coefficients from R.
        eps : float
            Numerical clipping to avoid exact 0 or 1.

        Returns
        -------
        p_start : np.ndarray of shape (T,)
        """
        #n_asin = triplet_df["N_asin"].to_numpy(dtype=float)
        n_asin = triplet_df["x1_asin"].to_numpy(dtype=float)
        range_score = triplet_df["x1_range_score"].to_numpy(dtype=float)

        n_asin = np.nan_to_num(n_asin, nan=0.0, posinf=0.0, neginf=0.0)
        range_score = np.nan_to_num(range_score, nan=0.0, posinf=0.0, neginf=0.0)

        eta = beta0 + beta1*n_asin + beta2*range_score
        p_start = StartLogitModel.logistic(eta)
        p_start = np.clip(p_start,eps,1-eps)
        return p_start
    
    
    @staticmethod
    def compute_ustart_prob(p_start,delta = 1, eps=1e-12):
        p_ustart = p_start*delta
        return p_ustart


    @staticmethod
    def compute_dstart_prob(p_start,delta = 1, eps=1e-12):
        p_dstart = p_start*delta
        return p_dstart
    


