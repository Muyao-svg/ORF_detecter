# frame_hmm.py
# This file implements the HMM inference algorithms (forward, backward, gamma, xi, Viterbi) for the unified-frame HMM with explicit Start/C/Stop states.
import numpy as np
from utils import logsumexp
from hmm_core import HMMState


def forward_log_dynamic_frame(emission_logp, transition_model, p_start, frame_run, hmm_state=None):
    emission_logp = np.asarray(emission_logp, dtype=float)
    p_start = np.asarray(p_start, dtype=float)
    frame_run = np.asarray(frame_run, dtype=int)

    T, S = emission_logp.shape

    if hmm_state is None:
        hmm_state = HMMState(T=T, S=S)
    else:
        hmm_state.reset_forward()

    logpi0_t0 = transition_model.build_logpi0_t0(
        p_start_0=p_start[0],
        frame_run_0=frame_run[0]
    )
    hmm_state.alpha[0, :] = logpi0_t0 + emission_logp[0, :]

    for t in range(1, T):
        logA_t = transition_model.build_logA_t(
            p_start_t=p_start[t],
            frame_run_t=frame_run[t]
        )
        for s in range(S):
            prev_terms = hmm_state.alpha[t - 1, :] + logA_t[:, s]
            hmm_state.alpha[t, s] = logsumexp(prev_terms) + emission_logp[t, s]

    hmm_state.loglik = logsumexp(hmm_state.alpha[T - 1, :])
    return hmm_state


def backward_log_dynamic_frame(emission_logp, transition_model, p_start, frame_run, hmm_state=None):
    emission_logp = np.asarray(emission_logp, dtype=float)
    p_start = np.asarray(p_start, dtype=float)
    frame_run = np.asarray(frame_run, dtype=int)

    T, S = emission_logp.shape

    if hmm_state is None:
        hmm_state = HMMState(T=T, S=S)

    hmm_state.beta = np.full((T, S), -np.inf, dtype=float)
    hmm_state.beta[T - 1, :] = 0.0

    for t in range(T - 2, -1, -1):
        logA_t1 = transition_model.build_logA_t(
            p_start_t=p_start[t + 1],
            frame_run_t=frame_run[t + 1]
        )
        for i in range(S):
            terms = logA_t1[i, :] + emission_logp[t + 1, :] + hmm_state.beta[t + 1, :]
            hmm_state.beta[t, i] = logsumexp(terms)

    return hmm_state


def compute_gamma_frame(hmm_state):
    if hmm_state.alpha is None or hmm_state.beta is None:
        raise ValueError("alpha and beta must be computed before gamma.")
    if hmm_state.loglik is None:
        raise ValueError("loglik must be available before gamma.")

    log_gamma = hmm_state.alpha + hmm_state.beta - hmm_state.loglik
    gamma = np.exp(log_gamma)
    gamma /= np.clip(gamma.sum(axis=1, keepdims=True), 1e-300, None)

    hmm_state.gamma = gamma
    return gamma


def compute_xi_dynamic_frame(emission_logp, transition_model, p_start, frame_run, hmm_state):
    emission_logp = np.asarray(emission_logp, dtype=float)
    p_start = np.asarray(p_start, dtype=float)
    frame_run = np.asarray(frame_run, dtype=int)

    T, S = emission_logp.shape
    xi = np.zeros((T - 1, S, S), dtype=float)

    for t in range(T - 1):
        logA_t1 = transition_model.build_logA_t(
            p_start_t=p_start[t + 1],
            frame_run_t=frame_run[t + 1]
        )

        log_xi_t = (
            hmm_state.alpha[t, :, None]
            + logA_t1
            + emission_logp[t + 1, None, :]
            + hmm_state.beta[t + 1, None, :]
            - hmm_state.loglik
        )

        xi_t = np.exp(log_xi_t)
        xi_t /= np.clip(xi_t.sum(), 1e-300, None)
        xi[t] = xi_t

    hmm_state.xi = xi
    return xi


def viterbi_decode_dynamic_frame(emission_logp, transition_model, p_start, frame_run, hmm_state=None):
    emission_logp = np.asarray(emission_logp, dtype=float)
    p_start = np.asarray(p_start, dtype=float)
    frame_run = np.asarray(frame_run, dtype=int)

    T, S = emission_logp.shape

    if hmm_state is None:
        hmm_state = HMMState(T=T, S=S)

    delta = np.full((T, S), -np.inf, dtype=float)
    psi = np.full((T, S), -1, dtype=int)

    logpi0_t0 = transition_model.build_logpi0_t0(
        p_start_0=p_start[0],
        frame_run_0=frame_run[0]
    )
    delta[0, :] = logpi0_t0 + emission_logp[0, :]
    psi[0, :] = -1

    for t in range(1, T):
        logA_t = transition_model.build_logA_t(
            p_start_t=p_start[t],
            frame_run_t=frame_run[t]
        )
        for s in range(S):
            prev_scores = delta[t - 1, :] + logA_t[:, s]
            psi[t, s] = np.argmax(prev_scores)
            delta[t, s] = prev_scores[psi[t, s]] + emission_logp[t, s]

    path = np.full(T, -1, dtype=int)
    path[T - 1] = np.argmax(delta[T - 1, :])

    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    hmm_state.delta = delta
    hmm_state.psi = psi
    hmm_state.viterbi_path = path
    return hmm_state