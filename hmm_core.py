# hmm_core.py
# The main pipeline only use the definition of class HMMState
import numpy as np

from utils import logsumexp
from transition_frame import TransitionModelFrame


class HMMState:
    """
    Container for HMM inference results on one triplet sequence.

    Attributes
    ----------
    T : int
        Number of observations (triplets).
    S : int
        Number of hidden states.
    alpha : np.ndarray of shape (T, S)
        Forward log probabilities.
    beta : np.ndarray of shape (T, S) or None
        Backward log probabilities.
    gamma : np.ndarray of shape (T, S) or None
        Posterior state probabilities.
    xi : np.ndarray of shape (T-1, S, S) or None
        Posterior transition probabilities.
    loglik : float or None
        Total log-likelihood of the sequence.
    """

    def __init__(self, T, S):
        self.T = T
        self.S = S

        self.alpha = np.full((T, S), -np.inf, dtype=float)
        self.beta = None
        self.gamma = None
        self.xi = None
        self.loglik = None

        self.delta = None
        self.psi = None
        self.viterbi_path = None

    def reset_forward(self):
        self.alpha[:] = -np.inf
        self.loglik = None

    def summary(self):
        print(f"HMMState(T={self.T}, S={self.S}, loglik={self.loglik})")
        print("alpha shape:", None if self.alpha is None else self.alpha.shape)
        print("beta shape:", None if self.beta is None else self.beta.shape)
        print("gamma shape:", None if self.gamma is None else self.gamma.shape)
        print("xi shape:", None if self.xi is None else self.xi.shape)


def forward_log(emission_logp, transition_model, hmm_state=None):
    """
    Run the forward algorithm in log space.

    Parameters
    ----------
    emission_logp : np.ndarray of shape (T, S)
        Log emission matrix. emission_logp[t, s] = log P(x_t | z_t = s)
    transition_model : TransitionModel
        Holds A and pi0.
    hmm_state : HMMState or None
        If provided, store results in this object.
        If None, create a new one.

    Returns
    -------
    hmm_state : HMMState
        Object containing alpha and loglik.
    """
    emission_logp = np.asarray(emission_logp, dtype=float)
    T, S = emission_logp.shape

    if S != transition_model.S:
        raise ValueError(
            f"State number mismatch: emission has {S} states, "
            f"but transition model has {transition_model.S}."
        )

    if hmm_state is None:
        hmm_state = HMMState(T=T, S=S)
    else:
        if hmm_state.T != T or hmm_state.S != S:
            raise ValueError(
                f"HMMState shape mismatch: got (T={hmm_state.T}, S={hmm_state.S}), "
                f"expected (T={T}, S={S})."
            )
        hmm_state.reset_forward()

    logA = transition_model.get_logA()
    logpi0 = transition_model.get_logpi0()

    # Initialization
    hmm_state.alpha[0, :] = logpi0 + emission_logp[0, :]

    # Recursion
    for t in range(1, T):
        for s in range(S):
            prev_terms = hmm_state.alpha[t - 1, :] + logA[:, s]
            hmm_state.alpha[t, s] = logsumexp(prev_terms) + emission_logp[t, s]

    # Termination
    hmm_state.loglik = logsumexp(hmm_state.alpha[T - 1, :])

    return hmm_state


def forward_log_dynamic(emission_logp, transition_model, p_start, p_ustart, p_dstart, hmm_state=None):
    emission_logp = np.asarray(emission_logp, dtype=float)
    T, S = emission_logp.shape

    if hmm_state is None:
        hmm_state = HMMState(T=T, S=S)
    else:
        hmm_state.reset_forward()

    # dynamic initialization for t = 0
    logpi0_t0 = transition_model.build_logpi0_t0(
        p_start_0=p_start[0],
        p_ustart_0=p_ustart[0]
    )
    hmm_state.alpha[0, :] = logpi0_t0 + emission_logp[0, :]

    for t in range(1, T):
        logA_t = transition_model.build_logA_t(
            p_start[t],
            p_ustart[t],
            p_dstart[t]
        )
        for s in range(S):
            prev_terms = hmm_state.alpha[t - 1, :] + logA_t[:, s]
            hmm_state.alpha[t, s] = logsumexp(prev_terms) + emission_logp[t, s]

    hmm_state.loglik = logsumexp(hmm_state.alpha[T - 1, :])
    return hmm_state




def backward_log(emission_logp, transition_model, hmm_state=None):
    """
    Run the backward algorithm in log space.

    Parameters
    ----------
    emission_logp : np.ndarray of shape (T, S)
        Log emission matrix. emission_logp[t, s] = log P(x_t | z_t = s)
    transition_model : TransitionModel
        Holds A and pi0.
    hmm_state : HMMState or None
        If provided, store results in this object.
        If None, create a new one.

    Returns
    -------
    hmm_state : HMMState
        Object containing beta.
    """
    emission_logp = np.asarray(emission_logp, dtype=float)
    T, S = emission_logp.shape

    if S != transition_model.S:
        raise ValueError(
            f"State number mismatch: emission has {S} states, "
            f"but transition model has {transition_model.S}."
        )

    if hmm_state is None:
        hmm_state = HMMState(T=T, S=S)

    hmm_state.beta = np.full((T, S), -np.inf, dtype=float)

    logA = transition_model.get_logA()

    # Initialization
    hmm_state.beta[T - 1, :] = 0.0

    # Recursion backward
    for t in range(T - 2, -1, -1):
        for i in range(S):
            terms = logA[i, :] + emission_logp[t + 1, :] + hmm_state.beta[t + 1, :]
            hmm_state.beta[t, i] = logsumexp(terms)

    return hmm_state

def backward_log_dynamic(emission_logp, transition_model, p_start,p_ustart,p_dstart,hmm_state=None):
    emission_logp = np.asarray(emission_logp, dtype=float)
    T, S = emission_logp.shape

    if hmm_state is None:
        hmm_state = HMMState(T=T, S=S)

    hmm_state.beta = np.full((T, S), -np.inf, dtype=float)
    hmm_state.beta[T - 1, :] = 0.0

    for t in range(T - 2, -1, -1):
        logA_t1 = transition_model.build_logA_t(p_start[t + 1], p_ustart[t+1],p_dstart[t+1])
        for i in range(S):
            terms = logA_t1[i, :] + emission_logp[t + 1, :] + hmm_state.beta[t + 1, :]
            hmm_state.beta[t, i] = logsumexp(terms)

    return hmm_state



def compute_gamma(hmm_state):
    """
    Compute posterior state probabilities gamma_t(s) = P(z_t=s | x_1:T).

    Parameters
    ----------
    hmm_state : HMMState
        Must already contain alpha, beta, and loglik.

    Returns
    -------
    gamma : np.ndarray of shape (T, S)
        Posterior state probabilities in normal probability space.
    """
    if hmm_state.alpha is None or hmm_state.beta is None:
        raise ValueError("alpha and beta must be computed before gamma.")
    if hmm_state.loglik is None:
        raise ValueError("loglik must be available before gamma.")

    log_gamma = hmm_state.alpha + hmm_state.beta - hmm_state.loglik
    gamma = np.exp(log_gamma)

    # Numerical cleanup: renormalize each row
    row_sums = gamma.sum(axis=1, keepdims=True)
    gamma = gamma / np.clip(row_sums, 1e-300, None)

    hmm_state.gamma = gamma
    return gamma


def compute_xi(emission_logp, transition_model, hmm_state):
    """
    Compute posterior transition probabilities
    xi_t(i,j) = P(z_t=i, z_{t+1}=j | x_1:T).

    Parameters
    ----------
    emission_logp : np.ndarray of shape (T, S)
        Log emission matrix.
    transition_model : TransitionModel
        Holds A and pi0.
    hmm_state : HMMState
        Must already contain alpha, beta, and loglik.

    Returns
    -------
    xi : np.ndarray of shape (T-1, S, S)
        Posterior transition probabilities in normal probability space.
    """
    if hmm_state.alpha is None or hmm_state.beta is None:
        raise ValueError("alpha and beta must be computed before xi.")
    if hmm_state.loglik is None:
        raise ValueError("loglik must be available before xi.")

    emission_logp = np.asarray(emission_logp, dtype=float)
    T, S = emission_logp.shape

    if S != transition_model.S:
        raise ValueError(
            f"State number mismatch: emission has {S} states, "
            f"but transition model has {transition_model.S}."
        )

    logA = transition_model.get_logA()
    xi = np.zeros((T - 1, S, S), dtype=float)

    for t in range(T - 1):
        log_xi_t = (
            hmm_state.alpha[t, :, None]
            + logA
            + emission_logp[t + 1, None, :]
            + hmm_state.beta[t + 1, None, :]
            - hmm_state.loglik
        )

        xi_t = np.exp(log_xi_t)
        denom = xi_t.sum()
        if denom > 0:
            xi_t /= denom
        xi[t] = xi_t

    hmm_state.xi = xi
    return xi

def compute_xi_dynamic(emission_logp, transition_model, p_start, p_ustart,p_dstart, hmm_state):
    emission_logp = np.asarray(emission_logp, dtype=float)
    T, S = emission_logp.shape
    xi = np.zeros((T - 1, S, S), dtype=float)

    for t in range(T - 1):
        logA_t1 = transition_model.build_logA_t(p_start[t + 1],p_ustart[t+1], p_dstart[t+1])
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