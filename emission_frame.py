# emission_frame.py

import numpy as np
from scipy.special import gammaln
from scipy.stats import nbinom
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom


class EmissionModelFrame:
    """
    Unified-frame emission model with explicit Start/C/Stop states.

    States:
        B
        Start0, C0, Stop0
        Start1, C1, Stop1
        Start2, C2, Stop2

    Observation for each row:
        x = [x1, x2, x3]
        N = x1 + x2 + x3

    Key idea:
        - B uses its own multinomial template pi
        - Start/C/Stop states use their own pi_base
        - pi_base is rotated according to:
              local_frame = (global_frame - frame_run) % 3

    Important:
        - Startg / Stopg are only meaningful on rows with frame_run == g
        - we gate them here by a strong negative penalty if frame_run != global_frame
        - p_start is NOT used here yet; it will come later in transition logic
    """

    def __init__(
        self,
        state_params,
        states,
        lambda_canonical_start=200.0,
        lambda_non_canonical_start=100.0,
        lambda_no_start=500.0,
        lambda_stop=100.0,
        lambda_no_stop=500.0,
        frame_mismatch_penalty=1e6,
        eps=1e-12
    ):
        self.state_params = state_params
        self.states = list(states)

        self.lambda_canonical_start = float(lambda_canonical_start)
        self.lambda_non_canonical_start = float(lambda_non_canonical_start)
        self.lambda_no_start = float(lambda_no_start)
        self.lambda_stop = float(lambda_stop)
        self.lambda_no_stop = float(lambda_no_stop)

        self.frame_mismatch_penalty = float(frame_mismatch_penalty)
        self.eps = eps

    # =====================================================
    # Basic probability pieces
    # =====================================================
    @staticmethod
    def log_nb_pmf(n, mu, r):
        if mu <= 0 or r <= 0 or n < 0:
            raise ValueError("mu, r must be > 0 and n must be >= 0.")
        p = r / (r + mu)
        return float(nbinom.logpmf(n, n=r, p=p))

    @staticmethod
    def log_multinomial_pmf(x, pi, eps=1e-12):
        x = np.asarray(x, dtype=float)
        pi = np.asarray(pi, dtype=float)

        if x.shape != (3,):
            raise ValueError(f"x must have shape (3,), got {x.shape}")
        if pi.shape != (3,):
            raise ValueError(f"pi must have shape (3,), got {pi.shape}")
        if np.any(x < 0):
            raise ValueError("x must be non-negative.")
        if np.any(pi < 0):
            raise ValueError("pi must be non-negative.")

        pi = np.maximum(pi, eps)
        pi = pi / pi.sum()
        n = x.sum()

        log_coeff = gammaln(n + 1.0) - np.sum(gammaln(x + 1.0))
        log_prob = np.sum(x * np.log(pi))
        return float(log_coeff + log_prob)

    # =====================================================
    # Rotation logic
    # =====================================================
    @staticmethod
    def rotate_pi_base(pi_base, global_frame, frame_run):
        """
        Convert canonical base template into row-specific local template.

        local_frame = (global_frame - frame_run) % 3

        local_frame = 0 -> x1-dominant
        local_frame = 1 -> x2-dominant
        local_frame = 2 -> x3-dominant
        """
        pi_base = np.asarray(pi_base, dtype=float)
        local_frame = (int(global_frame) - int(frame_run)) % 3
        return np.roll(pi_base, local_frame)

    @staticmethod
    def inverse_rotate_x_to_base(x, global_frame, frame_run):
        """
        Rotate observed x back to the canonical base coordinate.
        Used in M-step.
        """
        x = np.asarray(x, dtype=float)
        local_frame = (int(global_frame) - int(frame_run)) % 3
        return np.roll(x, -local_frame)

    # =====================================================
    # State-specific multinomial template
    # =====================================================
    def get_state_pi(self, state_name, frame_run):
        params = self.state_params[state_name]
        state_type = params["type"]

        if state_type == "background":
            pi = np.asarray(params["pi"], dtype=float)
            pi = np.maximum(pi, self.eps)
            return pi / pi.sum()

        if state_type in {"start", "coding", "stop"}:
            pi_base = np.asarray(params["pi_base"], dtype=float)
            global_frame = int(params["global_frame"])
            pi = self.rotate_pi_base(pi_base, global_frame, frame_run)
            pi = np.maximum(pi, self.eps)
            return pi / pi.sum()

        raise ValueError(f"Unknown state type for state {state_name}")

    # =====================================================
    # Emission for one row / one state
    # =====================================================
    def log_emission_one_state(
        self,
        x,
        state_name,
        frame_run,
        has_canonical_start=0,
        has_non_canonical_start=0,
        has_STOP_triplet=0
    ):
        params = self.state_params[state_name]
        state_type = params["type"]

        mu = float(params["mu"])
        r = float(params["r"])
        pi = self.get_state_pi(state_name, frame_run)

        n = float(np.sum(x))
        nb_part = self.log_nb_pmf(n, mu, r)
        multi_part = self.log_multinomial_pmf(x, pi, eps=self.eps)
        score = nb_part + multi_part

        # Each start/stop state only "active" on rows with matching frame_run; apply strong penalty if not
        if state_type in {"start", "stop"}:
            global_frame = int(params["global_frame"])

            # Only allow Startg / Stopg on rows whose frame_run == g
            if int(frame_run) != global_frame:
                score -= self.frame_mismatch_penalty
                return score

        # Start motif bonus / penalty
        if state_type == "start":
            if int(has_canonical_start) == 1:
                score += self.lambda_canonical_start
            elif int(has_non_canonical_start) == 1:
                score += self.lambda_non_canonical_start
            else:
                score -= self.lambda_no_start

        if state_type == "stop":
            if int(has_STOP_triplet) == 1:
                score += self.lambda_stop
            else:
                score -= self.lambda_no_stop

        return score

    # =====================================================
    # Emission matrix
    # =====================================================

    def build_emission_matrix(self, triplet_df):
        required_cols = {
            "x1", "x2", "x3",
            "frame_run",
            "has_canonical_start",
            "has_non_canonical_start",
            "has_STOP_triplet"
        }

        missing = required_cols - set(triplet_df.columns)
        if missing:
            raise ValueError(
                f"triplet_df missing columns: {sorted(missing)}"
            )

        x_mat = triplet_df[["x1", "x2", "x3"]].to_numpy(dtype=float)
        frame_run_vec = triplet_df["frame_run"].to_numpy(dtype=int)

        canonical_start_vec = triplet_df["has_canonical_start"].to_numpy(dtype=int)
        non_canonical_start_vec = triplet_df["has_non_canonical_start"].to_numpy(dtype=int)
        stop_vec = triplet_df["has_STOP_triplet"].to_numpy(dtype=int)

        T = len(triplet_df)
        S = len(self.states)
        emission_logp = np.zeros((T, S), dtype=float)

        for t in range(T):
            x = x_mat[t]
            frame_run = frame_run_vec[t]

            has_canonical_start = canonical_start_vec[t]
            has_non_canonical_start = non_canonical_start_vec[t]
            has_stop = stop_vec[t]

            for s, state_name in enumerate(self.states):
                emission_logp[t, s] = self.log_emission_one_state(
                    x=x,
                    state_name=state_name,
                    frame_run=frame_run,
                    has_canonical_start=has_canonical_start,
                    has_non_canonical_start=has_non_canonical_start,
                    has_STOP_triplet=has_stop
                )

        return emission_logp


    # =====================================================
    # M-step: update mu
    #   - background separately
    #   - Start pooled across frames
    #   - C pooled across frames
    #   - Stop pooled across frames
    # =====================================================
    def update_mu(self, triplet_df, gamma, pseudocount=1e-6):
        n_vec = triplet_df[["x1", "x2", "x3"]].sum(axis=1).to_numpy(dtype=float)
        state_to_idx = {s: i for i, s in enumerate(self.states)}

        # Background
        iB = state_to_idx["B"]
        wB = gamma[:, iB]
        mu_B = (wB * n_vec).sum() / (wB.sum() + pseudocount)
        self.state_params["B"]["mu"] = max(mu_B, 1e-6)

        # Start pooled
        start_states = ["Start0", "Start1", "Start2"]
        start_idx = [state_to_idx[s] for s in start_states]
        wStart = gamma[:, start_idx].sum(axis=1)
        mu_Start = (wStart * n_vec).sum() / (wStart.sum() + pseudocount)
        mu_Start = max(mu_Start, 1e-6)
        for s in start_states:
            self.state_params[s]["mu"] = mu_Start

        # Coding pooled
        coding_states = ["C0", "C1", "C2"]
        coding_idx = [state_to_idx[s] for s in coding_states]
        wC = gamma[:, coding_idx].sum(axis=1)
        mu_C = (wC * n_vec).sum() / (wC.sum() + pseudocount)
        mu_C = max(mu_C, 1e-6)
        for s in coding_states:
            self.state_params[s]["mu"] = mu_C

        # Stop pooled
        stop_states = ["Stop0", "Stop1", "Stop2"]
        stop_idx = [state_to_idx[s] for s in stop_states]
        wStop = gamma[:, stop_idx].sum(axis=1)
        mu_Stop = (wStop * n_vec).sum() / (wStop.sum() + pseudocount)
        mu_Stop = max(mu_Stop, 1e-6)
        for s in stop_states:
            self.state_params[s]["mu"] = mu_Stop

    # =====================================================
    # M-step: update multinomial templates
    #   - background pi updated directly
    #   - Start pi_base pooled after inverse rotation
    #   - Coding pi_base pooled after inverse rotation
    #   - Stop pi_base pooled after inverse rotation
    # =====================================================
    def update_multinomial_pi(self, triplet_df, gamma, pseudocount=1e-6):
        x_mat = triplet_df[["x1", "x2", "x3"]].to_numpy(dtype=float)
        frame_run_vec = triplet_df["frame_run"].to_numpy(dtype=int)

        state_to_idx = {s: i for i, s in enumerate(self.states)}

        # ---- background pi ----
        iB = state_to_idx["B"]
        weighted_B = (gamma[:, iB][:, None] * x_mat).sum(axis=0) + pseudocount
        pi_B = weighted_B / weighted_B.sum()
        self.state_params["B"]["pi"] = pi_B

        def pooled_pi_base(state_names):
            pooled_counts = np.zeros(3, dtype=float)

            for state_name in state_names:
                s_idx = state_to_idx[state_name]
                global_frame = int(self.state_params[state_name]["global_frame"])

                for t in range(len(triplet_df)):
                    x = x_mat[t]
                    frame_run = frame_run_vec[t]
                    x_aligned = self.inverse_rotate_x_to_base(
                        x=x,
                        global_frame=global_frame,
                        frame_run=frame_run
                    )
                    pooled_counts += gamma[t, s_idx] * x_aligned

            pooled_counts += pseudocount
            return pooled_counts / pooled_counts.sum()

        # Start pooled pi_base
        start_states = ["Start0", "Start1", "Start2"]
        pi_start_base = pooled_pi_base(start_states)
        for s in start_states:
            self.state_params[s]["pi_base"] = pi_start_base.copy()

        # Coding pooled pi_base
        coding_states = ["C0", "C1", "C2"]
        pi_c_base = pooled_pi_base(coding_states)
        for s in coding_states:
            self.state_params[s]["pi_base"] = pi_c_base.copy()

        # Stop pooled pi_base
        stop_states = ["Stop0", "Stop1", "Stop2"]
        pi_stop_base = pooled_pi_base(stop_states)
        for s in stop_states:
            self.state_params[s]["pi_base"] = pi_stop_base.copy()


    def update_r(self, triplet_df, gamma, r_min = 0.01, r_max = 1e5, pseudocount=1e-6):
        n_vec = triplet_df[["x1", "x2", "x3"]].sum(axis=1).to_numpy(dtype=float)  # Nt
        state_to_idx = {s: i for i, s in enumerate(self.states)}

        groups = [["B"],
                  ["Start0", "Start1", "Start2"],
                  ["C0", "C1", "C2"],
                  ["Stop0", "Stop1", "Stop2"]]

        for states in groups:
            idx = [state_to_idx[s] for s in states]
            weight = gamma[:, idx].sum(axis=1)  # sum over states in the group

            mu = float(self.state_params[states[0]]["mu"])  
            mu = max(mu, 1e-8)   # Ensure mu is positive
            
            def objective(log_r):
                r = np.exp(log_r) 
                p = r/(r + mu)
                log_prob = nbinom.logpmf(n_vec, n=r, p=p)
                return -np.dot(weight, log_prob)  # Negative log-likelihood

            result = minimize_scalar(objective, bounds=(np.log(r_min), np.log(r_max)), method='bounded')

            if result.success:
                r_hat = float(np.exp(result.x))
                for state in states:
                    self.state_params[state]["r"] = r_hat

                 

    # =====================================================
    # Debug helper
    # =====================================================
    def emission_parameter_summary(self):
        """
        Print the current emission parameters for each HMM state:
        - Negative Binomial: mu, r
        - Multinomial: pi or pi_base
        """

        print("\n" + "=" * 65)
        print("HMM EMISSION PARAMETER SUMMARY")
        print("=" * 65)

        for state_name in self.states:
            params = self.state_params[state_name]
            state_type = params["type"]

            mu = params["mu"]
            r = params["r"]

            if state_type == "background":
                pi = params["pi"]

                print(
                    f"{state_name:>6s} | "
                    f"type={state_type:<10s} | "
                    f"mu={mu:>10.3f} | "
                    f"r={r:>10.3f} | "
                    f"pi={np.round(pi, 4)}"
                )

            else:
                pi_base = params["pi_base"]
                global_frame = params["global_frame"]

                print(
                    f"{state_name:>6s} | "
                    f"type={state_type:<10s} | "
                    f"frame={global_frame} | "
                    f"mu={mu:>10.3f} | "
                    f"r={r:>10.3f} | "
                    f"pi_base={np.round(pi_base, 4)}"
                )

        print("=" * 65)
