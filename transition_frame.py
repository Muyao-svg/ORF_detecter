# transition_frame.py
# This file implements the dynamic transition model for the unified-frame HMM, where the transition probabilities from B to Start{g} depend on the position-specific start probability p_start[t] and frame_run[t].
import numpy as np


class TransitionModelFrame:
    """
    Dynamic transition model for unified-frame 10-state HMM.

    States:
        B
        Start0, C0, Stop0
        Start1, C1, Stop1
        Start2, C2, Stop2
    """

    def __init__(self, states, A, pi0, allowed_mask=None, eps=1e-12):
        self.states = list(states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.S = len(self.states)
        self.eps = eps

        self.A = np.asarray(A, dtype=float)
        self.pi0 = np.asarray(pi0, dtype=float)

        if self.A.shape != (self.S, self.S):
            raise ValueError(f"A must have shape ({self.S}, {self.S}), got {self.A.shape}")
        if self.pi0.shape != (self.S,):
            raise ValueError(f"pi0 must have shape ({self.S},), got {self.pi0.shape}")

        if allowed_mask is None:
            self.allowed_mask = np.ones((self.S, self.S), dtype=float)
        else:
            self.allowed_mask = np.asarray(allowed_mask, dtype=float)
            if self.allowed_mask.shape != (self.S, self.S):
                raise ValueError(
                    f"allowed_mask must have shape ({self.S}, {self.S}), got {self.allowed_mask.shape}"
                )

        self._apply_constraints()
        self._validate()

    def _apply_constraints(self):
        self.A = np.clip(self.A, 0.0, None)
        self.A = self.A * self.allowed_mask

        row_sums = self.A.sum(axis=1, keepdims=True)
        for i in range(self.S):
            if row_sums[i, 0] <= 0:
                allowed = self.allowed_mask[i] > 0
                n_allowed = allowed.sum()
                if n_allowed == 0:
                    raise ValueError(f"State {self.states[i]} has no allowed outgoing transitions.")
                self.A[i, allowed] = 1.0 / n_allowed
            else:
                self.A[i, :] /= row_sums[i, 0]

        pi0 = np.asarray(self.pi0, dtype=float).copy()
        pi0 = np.clip(pi0, 0.0, None)
        if pi0.sum() <= 0:
            raise ValueError("pi0 sums to zero.")
        self.pi0 = pi0 / pi0.sum()

    def _validate(self):
        if np.any(self.A < 0):
            raise ValueError("Transition matrix A contains negative values.")
        if np.any(self.pi0 < 0):
            raise ValueError("Initial distribution pi0 contains negative values.")

        row_sums = self.A.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError(f"Each row of A must sum to 1. Row sums: {row_sums}")

        if not np.isclose(self.pi0.sum(), 1.0, atol=1e-8):
            raise ValueError(f"pi0 must sum to 1. Got {self.pi0.sum()}")

    def get_logA(self):
        return np.log(np.maximum(self.A, self.eps))

    def get_logpi0(self):
        return np.log(np.maximum(self.pi0, self.eps))

    def build_logpi0_t0(self, p_start_0, frame_run_0, gamma=3, min_start_prob=0.7):
        out = np.zeros(self.S, dtype=float)

        iB = self.state_to_idx["B"]
        iS = self.state_to_idx[f"Start{int(frame_run_0)}"]

        p = p_start_0 ** gamma if p_start_0 >= min_start_prob else 0.0
        p = min(p, 0.95)

        out[iB] = 1.0 - p
        out[iS] = p

        logpi0_t0 = np.full(self.S, -np.inf, dtype=float)
        positive = out > 0
        logpi0_t0[positive] = np.log(out[positive])
        return logpi0_t0

    def build_logA_t(self, p_start_t, frame_run_t, gamma=3, min_start_prob=0.7):
        """
        Dynamic transition at row t.

        Only B row is rewritten:
            B -> B
            B -> Start{frame_run_t}
        """
        A_t = np.array(self.A, dtype=float, copy=True)

        iB = self.state_to_idx["B"]
        iS = self.state_to_idx[f"Start{int(frame_run_t)}"]

        p = p_start_t ** gamma if p_start_t >= min_start_prob else 0.0
        if p > 0.95:
            p = 0.95

        A_t[iB, :] = 0.0
        A_t[iB, iB] = 1.0 - p
        A_t[iB, iS] = p

        A_t = A_t * self.allowed_mask
        row_sums = A_t.sum(axis=1, keepdims=True)
        A_t = A_t / np.clip(row_sums, 1e-300, None)

        logA_t = np.full_like(A_t, -np.inf, dtype=float)
        positive = A_t > 0
        logA_t[positive] = np.log(A_t[positive])
        return logA_t
    

    def update_coding_transitions(self, xi, pseudocount=1e-6, pooled=True):
        idx = self.state_to_idx

        if pooled:
            n_cc = (
                xi[:, idx["C0"], idx["C0"]].sum() +
                xi[:, idx["C1"], idx["C1"]].sum() +
                xi[:, idx["C2"], idx["C2"]].sum()
            )
            n_cs = (
                xi[:, idx["C0"], idx["Stop0"]].sum() +
                xi[:, idx["C1"], idx["Stop1"]].sum() +
                xi[:, idx["C2"], idx["Stop2"]].sum()
            )

            denom = n_cc + n_cs + 2 * pseudocount
            p_cc = (n_cc + pseudocount) / denom
            p_cs = (n_cs + pseudocount) / denom

            for g in [0, 1, 2]:
                self.A[idx[f"C{g}"], :] = 0.0
                self.A[idx[f"C{g}"], idx[f"C{g}"]] = p_cc
                self.A[idx[f"C{g}"], idx[f"Stop{g}"]] = p_cs

        else:
            for g in [0, 1, 2]:
                n_cc = xi[:, idx[f"C{g}"], idx[f"C{g}"]].sum()
                n_cs = xi[:, idx[f"C{g}"], idx[f"Stop{g}"]].sum()

                denom = n_cc + n_cs + 2 * pseudocount
                p_cc = (n_cc + pseudocount) / denom
                p_cs = (n_cs + pseudocount) / denom

                self.A[idx[f"C{g}"], :] = 0.0
                self.A[idx[f"C{g}"], idx[f"C{g}"]] = p_cc
                self.A[idx[f"C{g}"], idx[f"Stop{g}"]] = p_cs

        self.A = self.A * self.allowed_mask
        row_sums = self.A.sum(axis=1, keepdims=True)
        self.A = self.A / np.clip(row_sums, 1e-300, None)