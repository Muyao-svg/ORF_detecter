# frame_em.py

import numpy as np
import pandas as pd

# ==========================================================
# 1. Non-HMM EM for unified-frame mixture model
# Non used in main pipeline, but may be useful for comparison and ablation.
# ==========================================================
class FrameMixtureModel:
    """
    Non-HMM EM on unified big triplet dataframe.

    States:
        B
        Start0, C0, Stop0
        Start1, C1, Stop1
        Start2, C2, Stop2

    Responsibilities:
        gamma[t, s] = posterior probability that row t belongs to state s

    This is a mixture model, not an HMM:
        P(z_t = s) = mix_weights[s]
        P(x_t | z_t = s) = given by EmissionModelFrame
    """

    def __init__(self, states, emission_model, mix_weights=None, eps=1e-12):
        self.states = list(states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.S = len(self.states)
        self.emission_model = emission_model
        self.eps = eps

        if mix_weights is None:
            # B + 3*(Start, C, Stop)
            # sums to 1.00
            mix_weights = np.array([
                0.70,   # B
                0.03, 0.07, 0.01,   # Start0, C0, Stop0
                0.03, 0.07, 0.01,   # Start1, C1, Stop1
                0.03, 0.07, 0.01    # Start2, C2, Stop2
            ], dtype=float)

        mix_weights = np.asarray(mix_weights, dtype=float)
        if mix_weights.shape != (self.S,):
            raise ValueError(f"mix_weights must have shape ({self.S},), got {mix_weights.shape}")

        mix_weights = np.clip(mix_weights, 0.0, None)
        if mix_weights.sum() <= 0:
            raise ValueError("mix_weights sum to zero.")

        self.mix_weights = mix_weights / mix_weights.sum()

    def get_log_mix_weights(self):
        return np.log(np.maximum(self.mix_weights, self.eps))

    def update_mix_weights(self, gamma, pseudocount=1e-6):
        w = gamma.sum(axis=0) + pseudocount
        self.mix_weights = w / w.sum()

    def summary(self):
        print("\nCurrent mixture weights:")
        for s, w in zip(self.states, self.mix_weights):
            print(f"  {s:>6s}: {w:.4f}")


def logsumexp(x, axis=None, keepdims=False):
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def run_e_step_mixture(triplet_df, mixture_model):
    """
    E-step for non-HMM mixture model.

    Returns
    -------
    emission_logp : np.ndarray of shape (T, S)
    gamma : np.ndarray of shape (T, S)
    loglik : float
    """
    emission_logp = mixture_model.emission_model.build_emission_matrix(triplet_df)  # Shape (T, S): triplet * state -> logP(x_t | z_t = s)
    logw = mixture_model.get_log_mix_weights()[None, :]   # shape (1, S)

    log_joint = emission_logp + logw                      # shape (T, S)
    log_norm = logsumexp(log_joint, axis=1, keepdims=True)
    gamma = np.exp(log_joint - log_norm)

    loglik = float(np.sum(log_norm))
    return emission_logp, gamma, loglik


def run_frame_em(
    triplet_df,
    emission_model,
    states,
    n_iter=20,
    mix_weights_init=None,
    update_mix_weights=True,
    update_mu=True,
    update_pi=True,
    update_r=True,
    verbose=True,
):
    """
    Run non-HMM EM for unified-frame mixture model.

    Parameters
    ----------
    triplet_df : pd.DataFrame
        Must contain columns x1, x2, x3, frame_run, has_ATG_triplet, has_STOP_triplet
    emission_model : EmissionModelFrame
    states : list[str]
    n_iter : int
    mix_weights_init : array-like of shape (S,), optional
    update_mix_weights : bool
    update_mu : bool
    update_pi : bool
    verbose : bool

    Returns
    -------
    mixture_model : FrameMixtureModel
    gamma : np.ndarray of shape (T, S)
    loglik_trace : list[float]
    out_df : pd.DataFrame
        Original dataframe + posterior supports + best_state summaries
    """
    required_cols = {"x1", "x2", "x3", "frame_run", "has_ATG_triplet", "has_STOP_triplet"}
    missing = required_cols - set(triplet_df.columns)
    if missing:
        raise ValueError(f"triplet_df missing columns: {sorted(missing)}")

    mixture_model = FrameMixtureModel(
        states=states,
        emission_model=emission_model,
        mix_weights=mix_weights_init
    )

    loglik_trace = []
    gamma = None

    for it in range(n_iter):
        # E-step
        emission_logp, gamma, loglik = run_e_step_mixture(triplet_df, mixture_model)
        loglik_trace.append(loglik)

        if verbose:
            print(f"\nEM iter {it+1:02d}, loglik = {loglik:.6f}")

        # M-step
        if update_mix_weights:
            mixture_model.update_mix_weights(gamma)

        if update_mu:
            emission_model.update_mu(triplet_df, gamma)

        if update_pi:
            emission_model.update_multinomial_pi(triplet_df, gamma)

        if verbose:
            mixture_model.summary()
            emission_model.summary()

    out_df = attach_frame_em_results(triplet_df, gamma, states)
    return mixture_model, gamma, loglik_trace, out_df


def attach_frame_em_results(triplet_df, gamma, states):
    """
    Convert posterior gamma into convenient result columns.

    We keep:
      - fine-grained state supports
      - aggregated global-frame supports
      - lambda_hat = 1 - support_B
      - best_state = argmax over all 10 states
      - best_frame_only = argmax over aggregated F0/F1/F2
      - best_group_state = argmax over B/F0/F1/F2
    """
    state_to_idx = {s: i for i, s in enumerate(states)}
    out = triplet_df.copy().reset_index(drop=True)

    # -------- background --------
    out["support_B"] = gamma[:, state_to_idx["B"]]

    # -------- fine-grained state supports --------
    for s in states:
        out[f"support_{s}"] = gamma[:, state_to_idx[s]]

    # -------- aggregated frame supports --------
    out["support_corrige_F0"] = (
        out["support_Start0"] + out["support_C0"] + out["support_Stop0"]
    )
    out["support_corrige_F1"] = (
        out["support_Start1"] + out["support_C1"] + out["support_Stop1"]
    )
    out["support_corrige_F2"] = (
        out["support_Start2"] + out["support_C2"] + out["support_Stop2"]
    )

    # -------- coding probability --------
    out["lambda_hat"] = (
        out["support_corrige_F0"] +
        out["support_corrige_F1"] +
        out["support_corrige_F2"]
    )

    # -------- best fine-grained state --------
    best_idx = np.argmax(gamma, axis=1)
    out["best_state"] = [states[i] for i in best_idx]

    # -------- best global frame only --------
    agg_frame = np.vstack([
        out["support_corrige_F0"].to_numpy(),
        out["support_corrige_F1"].to_numpy(),
        out["support_corrige_F2"].to_numpy()
    ]).T
    best_frame_idx = np.argmax(agg_frame, axis=1)
    out["best_frame_only"] = [f"F{i}" for i in best_frame_idx]

    # -------- best among B / F0 / F1 / F2 --------
    agg_group = np.vstack([
        out["support_B"].to_numpy(),
        out["support_corrige_F0"].to_numpy(),
        out["support_corrige_F1"].to_numpy(),
        out["support_corrige_F2"].to_numpy()
    ]).T
    group_names = ["B", "F0", "F1", "F2"]
    best_group_idx = np.argmax(agg_group, axis=1)
    out["best_group_state"] = [group_names[i] for i in best_group_idx]

    return out



#==========================================================
# 2. HMM EM for unified-frame model
# Implemented in frame_hmm.py, which is more complex and has more details.  
#==========================================================
# EM + HMM
from frame_hmm import (
    forward_log_dynamic_frame,backward_log_dynamic_frame,compute_gamma_frame,compute_xi_dynamic_frame,viterbi_decode_dynamic_frame,)


def run_hmm_em_frame(
    triplet_df,
    emission_model,
    transition_model,
    p_start,
    frame_run,
    n_iter=20,
    tol=1e-4,
    update_mu=True,
    update_pi=True,
    update_r=True,
    update_transition=True,
    pooled_transition=True,
    pseudocount=1e-6,
    verbose=True,
):
    loglik_trace = []
    hmm_state = None

    for it in range(n_iter):
        # -------------------------
        # E-step
        # -------------------------
        emission_logp = emission_model.build_emission_matrix(triplet_df)

        hmm_state = forward_log_dynamic_frame(
            emission_logp=emission_logp,
            transition_model=transition_model,
            p_start=p_start,
            frame_run=frame_run,
            hmm_state=hmm_state
        )

        hmm_state = backward_log_dynamic_frame(
            emission_logp=emission_logp,
            transition_model=transition_model,
            p_start=p_start,
            frame_run=frame_run,
            hmm_state=hmm_state
        )

        gamma = compute_gamma_frame(hmm_state)
        xi = compute_xi_dynamic_frame(
            emission_logp=emission_logp,
            transition_model=transition_model,
            p_start=p_start,
            frame_run=frame_run,
            hmm_state=hmm_state
        )

        loglik = float(hmm_state.loglik)
        loglik_trace.append(loglik)

        if verbose:
            print(f"HMM-EM iter {it+1:02d}, loglik = {loglik:.6f}")

        # -------------------------
        # M-step
        # -------------------------
        if update_mu:
            emission_model.update_mu(
                triplet_df=triplet_df,
                gamma=gamma,
                pseudocount=pseudocount
            )

        if update_r:
            emission_model.update_r(
                triplet_df=triplet_df,
                gamma=gamma,
                pseudocount=pseudocount
            )

        if update_pi:
            emission_model.update_multinomial_pi(
                triplet_df=triplet_df,
                gamma=gamma,
                pseudocount=pseudocount
            )

        if update_transition:
            transition_model.update_coding_transitions(
                xi=xi,
                pseudocount=pseudocount,
                pooled=pooled_transition
            )

        # -------------------------
        # convergence
        # -------------------------
        if it >= 1:
            delta = loglik_trace[-1] - loglik_trace[-2]
            if verbose:
                print(f"  delta loglik = {delta:.6f}")

            if abs(delta) < tol:
                if verbose:
                    print("HMM-EM converged.")
                break

    # final decode with updated params
    emission_logp_final = emission_model.build_emission_matrix(triplet_df)
    hmm_state = viterbi_decode_dynamic_frame(
        emission_logp=emission_logp_final,
        transition_model=transition_model,
        p_start=p_start,
        frame_run=frame_run,
        hmm_state=hmm_state
    )

    hmm_state = forward_log_dynamic_frame(
        emission_logp=emission_logp_final,
        transition_model=transition_model,
        p_start=p_start,
        frame_run=frame_run,
        hmm_state=hmm_state
    )
    hmm_state = backward_log_dynamic_frame(
        emission_logp=emission_logp_final,
        transition_model=transition_model,
        p_start=p_start,
        frame_run=frame_run,
        hmm_state=hmm_state
    )
    gamma = compute_gamma_frame(hmm_state)
    xi = compute_xi_dynamic_frame(
        emission_logp=emission_logp_final,
        transition_model=transition_model,
        p_start=p_start,
        frame_run=frame_run,
        hmm_state=hmm_state
    )

    hmm_state.gamma = gamma
    hmm_state.xi = xi

    return hmm_state, loglik_trace