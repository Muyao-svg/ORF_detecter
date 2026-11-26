"""
kmer index: 1 based
frame/phase: 0,1,2

The columns of A represent candidate ORFs; rows match sorted 1-based k-mer
indices. For exploratory candidate discovery only, the residual is rounded and
clipped before running an existing NB/Multinomial HMM. Statistical model
comparison is always performed on original counts.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from constants_frame import *

EPS = 1e-12

def background_vector(background, K):
    """
    ❓ The background has not yet been implemented in detail; here, `background=1` is used for testing!!!!
    """
    b = np.broadcast_to(np.asarray(background, dtype=float), (K,)).copy()
    if not np.all(np.isfinite(b)) or np.any(b <= 0):
        raise ValueError("background must be finite and strictly positive")
    return b


def poisson_loglik(Y, mu):
    Y, mu = np.asarray(Y, float), np.asarray(mu, float)
    if Y.shape != mu.shape or not np.all(np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError("Y/mu shape mismatch or nonpositive mu")
    return float(np.sum(Y * np.log(mu) - mu - gammaln(Y + 1)))



def fit_activity_poisson(Y, A, background=1.0):
    """Joint nonnegative Poisson MLE for all candidate ORF activities.
    Y: raw counts
    A: design matrix
    """
    Y, A = np.asarray(Y, float), np.asarray(A, float)
    if A.ndim != 2 or Y.shape != (A.shape[0],):
        raise ValueError("Expected Y (K,) and A (K,J)")
    if not np.all(np.isfinite(Y)) or not np.all(np.isfinite(A)) or np.any(Y < 0) or np.any(A < 0):
        raise ValueError("Y and A must be finite and nonnegative")
    K, J = A.shape
    b = background_vector(background, K)
    if J == 0:
        return np.zeros(0)
    x0 = np.full(J, max(float(Y.mean() - b.mean()), 0) / max(J, 1))
    x0 /= np.maximum(A.mean(axis=0), EPS)
    def fun(beta): # Negative Poisson Log-Likelihood
        mu = b + A @ beta
        return float(np.sum(mu - Y * np.log(mu)))
    def jac(beta): # gradient
        mu = b + A @ beta
        return A.T @ (1 - Y / mu)
    result = minimize(fun, x0, jac=jac, method="L-BFGS-B",
                      bounds=[(0, None)] * J,
                      options={"maxiter": 2000, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"Poisson MLE failed: {result.message}")
    return np.maximum(result.x, 0)


def build_A_matrix(k_indices, candidate_orfs, pi_coding_by_frame):
    """Construct (K,J) ORF profiles, phase-aligned at each ORF's k_start.

    frame in the current run_single_gene.py is a sliding-triplet *row offset*,
    not necessarily the absolute P-site phase. This assumes contiguous k_idx
    across each predicted ORF (as the HMM triplet builder does).

    k_indice:1,2,3...k
    candidate_orfs: ORFs predicted by HMM
    pi_coding_by_frame: pi estimated by HMM
    """
    k = np.asarray(k_indices, dtype=int)
    if k.ndim != 1 or len(np.unique(k)) != len(k):
        raise ValueError("k_indices must be unique and one-dimensional")
    A = np.zeros((len(k), len(candidate_orfs)), float)
    for j, (_, orf) in enumerate(candidate_orfs.iterrows()):
        start, end, frame = int(orf["k_start"]), int(orf["k_end"]), int(orf["frame"])
        pi = np.asarray(pi_coding_by_frame[frame], float)
        if pi.shape != (3,) or np.any(pi < 0) or not np.all(np.isfinite(pi)) or not np.isclose(pi.sum(), 1):
            raise ValueError(f"Invalid pi for frame {frame}: {pi}")
        mask = (k >= start) & (k <= end)
        segment = k[mask]
        if not len(segment) or len(segment) != end - start + 1 or segment[0] != start:
            raise ValueError(f"Missing k-mer indices in ORF [{start}, {end}]")
        A[mask, j] = pi[(segment - start) % 3]
        # for example:
        # segment = [4,5,6,7,8,9], so start = 4
        # segment - start = [0,1,2,3,4,5]
        # (segment - start) % 3 = [0,1,2,0,1,2]
        # pi[(segment - start) % 3] corresponding to the pi_0, pi_1, pi_2 estimated by HMM
    return A



def make_residual_dataframe(df_gene, k_indices, Y, A_selected, beta):
    """
    Return a copy with 'sum' replaced by clipped residual counts.
    df_gene MUST already be sorted with k_idx matching k_indices

    df_gene: ["kmer_id", "sequence","sum_counts".....]
    Y = df_gene["sum_counts"]
    """
    k = np.asarray(k_indices, dtype=int)
    Y = np.asarray(Y, float)
    A_selected = np.asarray(A_selected, float)
    beta = np.asarray(beta, float)
    if "k_idx" not in df_gene or not np.array_equal(df_gene["k_idx"].to_numpy(dtype=int), k):
        raise ValueError("df_gene must be sorted and aligned by k_idx")
    if A_selected.shape != (len(k), len(beta)) or Y.shape != (len(k),):
        raise ValueError("Residual dimensions do not agree")
    residual_raw = Y - A_selected @ beta
    df = df_gene.copy()
    df["sum"] = np.rint(np.maximum(residual_raw, 0)).astype(np.int64)
    return df


def orf_key(orf):
    return (int(orf["frame"]), int(orf["k_start"]), int(orf["k_end"]))



def evaluate_candidate_orfs(Y, k_indices, candidates, selected_orfs,
                            A_selected, beta_selected, pi_fixed,
                            background=1.0, min_activity=1e-6):
    """Try each new candidate with *joint* refit; return best BIC candidate.

    Returns (best_or_none, trials_df), where best includes A, beta, mu,
    loglik, bic. Does not decide whether the BIC improvement is sufficient.
    ❓pi_fix uses the parameters estimated by the first HMM. These parameters are fixed to simplify the logic, but will be modified later!!!!!!!!!!!!!!!!!!!!!!!!!!
    ❓pi_fix --> pi_coding_by_frame???????
    """
    Y = np.asarray(Y, float)
    K = len(Y)
    b = background_vector(background, K)
    # Remove duplicate ORFs
    selected_keys = {orf_key(row) for _, row in selected_orfs.iterrows()}

    # If the ORF predicted by the current HMM has already been accepted in a previous iteration, skip it;
    # If the same ORF appears multiple times among the candidates, keep the entry with the higher orf_score
    unique = {}
    for _, row in candidates.iterrows():
        key = orf_key(row)
        if key in selected_keys:
            continue
        if key not in unique or float(row.get("orf_score", -np.inf)) > float(unique[key].get("orf_score", -np.inf)):
            unique[key] = row
    best, records = None, []
    for key, row in unique.items():
        trial_info = {"frame": key[0], "k_start": key[1], "k_end": key[2]}
        if key[0] not in pi_fixed:
            trial_info["status"] = "missing_pi"
            records.append(trial_info)
            continue
        try:
            #Construct a new A_matrix that includes both previously evaluated ORFs and newly added ORFs
            a_new = build_A_matrix(k_indices, pd.DataFrame([row]), pi_fixed)
        except ValueError:
            trial_info["status"] = "invalid_A"
            records.append(trial_info)
            continue
        # Combine the ORFs newly predicted by the HMM with the ORFs that have already passed testing to form a new A_matrix.
        A_trial = np.column_stack((A_selected, a_new))  
        beta_trial = fit_activity_poisson(Y, A_trial, b) # Estimate beta based on the new A
        mu = b + A_trial @ beta_trial
        ll = poisson_loglik(Y, mu)
        bic = -2 * ll + A_trial.shape[1] * np.log(K)   # Compute BIC
        trial_info.update({"beta_new": beta_trial[-1], "loglik": ll,
                           "BIC": bic, "status": "evaluated"})
        records.append(trial_info)
        # If the ORF beta is less than the threshold, it is considered to make no contribution.
        if beta_trial[-1] <= min_activity:
            trial_info["status"] = "negligible_activity"
            continue
        # ❓ Select the candidate ORF with the lowest BIC. Need improve: Perhaps can select multiple ORF at once?
        if best is None or bic < best["BIC"]:
            best = {"row": row, "A": A_trial, "beta": beta_trial,
                    "mu": mu, "loglik": ll, "BIC": bic, "key": key}
    return best, pd.DataFrame(records, columns=["frame", "k_start", "k_end", "beta_new", "loglik", "BIC", "status"])


def signal_table(k_indices, Y, A_selected, beta, background):
    Y = np.asarray(Y, float)
    b = background_vector(background, len(Y))
    explained = A_selected @ beta
    residual = Y - explained
    return pd.DataFrame({"k_idx": k_indices, "Y_original": Y,
                         "background": b, "explained_orfs": explained,
                         "mu_poisson": b + explained,
                         "residual_raw": residual,
                         "residual_for_hmm": np.rint(np.maximum(residual, 0)).astype(int)})








