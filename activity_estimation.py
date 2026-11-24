
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# 首先用poisson 分布假设beta_k，假设b_k = 1.0，之后再用hmm或者别的估计方法替代！！
def fit_activity_poisson(Y, A, background=1.0):
    """
    给定一组ORF,联合估计beta. mini beta
    Estimate ORF activities by Poisson MLE.

    Y: observed counts, shape (K,)
    A: design matrix, shape (K, J)
    background: fixed background mean
    """

    Y = np.asarray(Y, dtype=float)
    A = np.asarray(A, dtype=float)

    K, J = A.shape

    if Y.shape != (K,):
        raise ValueError("Y and A have incompatible shapes")

    b = np.broadcast_to(
        np.asarray(background, dtype=float), (K,)
    )

    if np.any(Y < 0) or np.any(A < 0) or np.any(b < 0):
        raise ValueError("Y, A and background must be nonnegative")

    if J == 0:
        return np.array([])

    def objective(beta):
        mu = np.maximum(b + A @ beta, 1e-12)

        # Negative Poisson log-likelihood (up to a constant)
        return np.sum(mu - Y * np.log(mu))

    def gradient(beta):
        mu = np.maximum(b + A @ beta, 1e-12)

        return A.T @ (1 - Y / mu)

    beta_init = np.full(J, max(Y.mean(), 1.0))

    result = minimize(
        objective,
        x0=beta_init,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(0, None)] * J,
        options={"maxiter": 2000}
    )

    if not result.success:
        raise RuntimeError(result.message)

    # 返回最大似然beta_j
    return result.x


def poisson_loglik(Y, mu):
    """Full Poisson log-likelihood."""
    Y = np.asarray(Y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-12)

    return np.sum(Y * np.log(mu) - mu - gammaln(Y + 1))


def select_orfs_poisson(Y, A, background=1.0):
    """
    Select ORFs using forward selection and BIC. 选择最佳J

    Returns
    -------
    selected : list
        Selected column indices in the original A.
    beta : ndarray
        Estimated activities of selected ORFs.
    history : DataFrame
        BIC and log-likelihood at every accepted step.
    
        假设hmm预测出5个ORF,从J=0开始，每轮加入一个ORF;
        iteration 1: 分别测试5个ORF,选择bic最低的
        iteration 2： 固定已选择的ORF, 分别尝试加入其他ORF, 每次都联合，重新估计beta --》 fit_activity_poisson
        如果加入新的ORF不能使BIC减小则停止
    """

    Y = np.asarray(Y, dtype=float)
    A = np.asarray(A, dtype=float)

    K, J_max = A.shape
    b = np.broadcast_to(
        np.asarray(background, dtype=float), (K,)
    ).copy()

    selected = []
    remaining = list(range(J_max))
    beta = np.array([])

    # Initial model: background only
    loglik = poisson_loglik(Y, b)
    bic = -2 * loglik

    history = [{
        "J": 0,
        "added_orf": None,
        "loglik": loglik,
        "BIC": bic,
    }]

    while remaining:

        best_bic = np.inf
        best_orf = None
        best_beta = None
        best_loglik = None

        # Try adding each remaining ORF
        for j in remaining:

            trial_selected = selected + [j]
            A_trial = A[:, trial_selected]

            # Jointly estimate all activities
            beta_trial = fit_activity_poisson(
                Y, A_trial, background=b
            )

            mu_trial = b + A_trial @ beta_trial

            ll_trial = poisson_loglik(Y, mu_trial)

            J_trial = len(trial_selected)
            bic_trial = -2 * ll_trial + J_trial * np.log(K)

            if bic_trial < best_bic:
                best_bic = bic_trial
                best_orf = j
                best_beta = beta_trial
                best_loglik = ll_trial

        # Stop if adding an ORF does not improve BIC
        if best_bic >= bic:
            break

        # Accept the best ORF
        selected.append(best_orf)
        remaining.remove(best_orf)

        beta = best_beta
        bic = best_bic

        history.append({
            "J": len(selected),
            "added_orf": best_orf,
            "loglik": best_loglik,
            "BIC": bic,
        })

    return selected, beta, pd.DataFrame(history)

