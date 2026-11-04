import numpy as np
import pandas as pd


def max_consecutive_true(arr):
    """
    Return the maximum number of consecutive True values.
    """
    max_run = 0
    current = 0

    for v in arr:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0

    return max_run


def get_internal_gap_stats(
    triplet_df,
    start_i,
    end_i,
    zero_threshold=0.0
):
    """
    Compute zero-gap statistics inside the internal coding region.

    Start and Stop rows are excluded:
        Start | C C C ... C | Stop
                ^^^^^^^^^^^
                checked part

    A gap is defined as a triplet whose total signal x1+x2+x3 <= zero_threshold.
    """
    if end_i - start_i + 1 <= 2:
        return {
            "internal_n_triplets": 0,
            "internal_sum_N": 0.0,
            "internal_nonzero_fraction": 0.0,
            "max_zero_gap": 5,
        }

    internal = triplet_df.iloc[start_i + 1:end_i].copy()

    N_internal = internal[["x1", "x2", "x3"]].sum(axis=1).to_numpy(dtype=float)

    is_zero_gap = N_internal <= zero_threshold
    max_zero_gap = max_consecutive_true(is_zero_gap)

    internal_n_triplets = len(N_internal)
    internal_sum_N = float(N_internal.sum())

    if internal_n_triplets > 0:
        internal_nonzero_fraction = float((N_internal > zero_threshold).mean())
    else:
        internal_nonzero_fraction = 0.0

    return {
        "internal_n_triplets": internal_n_triplets,
        "internal_sum_N": internal_sum_N,
        "internal_nonzero_fraction": internal_nonzero_fraction,
        "max_zero_gap": max_zero_gap,
    }


def has_long_zero_gap(
    triplet_df,
    start_i,
    end_i,
    zero_threshold=0.0,
    max_zero_gap_allowed=9
):
    """
    Return True if the predicted ORF contains a long zero-coverage gap.

    Here:
        max_zero_gap_allowed = 2

    Therefore:
        max_zero_gap >= 3  -> reject this ORF.
    """
    stats = get_internal_gap_stats(
        triplet_df=triplet_df,
        start_i=start_i,
        end_i=end_i,
        zero_threshold=zero_threshold
    )

    return stats["max_zero_gap"] > max_zero_gap_allowed


def extract_orfs_from_viterbi(
    triplet_df,
    hmm_state,
    state_names,
    emission_logp,
    transition_model,
    p_start,
    frame_run,
    gene_name,
    zero_threshold=0.0,
    max_zero_gap_allowed=100,
):
    path = hmm_state.viterbi_path
    states = [state_names[s] for s in path]
    records = []

    i = 0
    T = len(states)

    frame_defs = [
        ("Start0", "C0", "Stop0", "0"),
        ("Start1", "C1", "Stop1", "1"),
        ("Start2", "C2", "Stop2", "2"),
    ]

    while i < T:
        matched = False

        for start_state, coding_state, stop_state, frame_label in frame_defs:
            if states[i] != start_state:
                continue

            start_i = i
            j = i + 1

            while j < T and states[j] == coding_state:
                j += 1

            if j < T and states[j] == stop_state:
                end_i = j

                gap_stats = get_internal_gap_stats(
                    triplet_df=triplet_df,
                    start_i=start_i,
                    end_i=end_i,
                    zero_threshold=zero_threshold
                )

                # Reject ORF if internal coding region contains >= 3 consecutive zero triplets
                if gap_stats["max_zero_gap"] > max_zero_gap_allowed:
                    i = end_i + 1
                    matched = True
                    break

                seg_score = compute_orf_score(
                    start_i=start_i,
                    end_i=end_i,
                    path=path,
                    emission_logp=emission_logp,
                    transition_model=transition_model,
                    p_start=p_start,
                    frame_run=frame_run,
                )

                records.append({
                    "Gene": gene_name,
                    "frame": frame_label,
                    "triplet_start_idx": start_i,
                    "triplet_end_idx": end_i,
                    "k_start": int(triplet_df.iloc[start_i]["start_kidx"]),
                    "k_end": int(triplet_df.iloc[end_i]["end_kidx"]),
                    "n_triplets": end_i - start_i + 1,
                    "orf_score": seg_score,
                    "mean_orf_score": seg_score / (end_i - start_i + 1),
                    "has_canonical_start": int(triplet_df.iloc[start_i]["has_canonical_start"]),
                    "has_non_canonical_start": int(triplet_df.iloc[start_i]["has_non_canonical_start"]),
                    "has_stop": int(triplet_df.iloc[end_i]["has_STOP_triplet"]),

                    # Useful diagnostic columns
                    #"internal_n_triplets": gap_stats["internal_n_triplets"],
                    #"internal_sum_N": gap_stats["internal_sum_N"],
                    #"internal_nonzero_fraction": gap_stats["internal_nonzero_fraction"],
                    #"max_zero_gap": gap_stats["max_zero_gap"],
                })

                i = end_i + 1
                matched = True
                break

        if not matched:
            i += 1

    if len(records) == 0:
        return pd.DataFrame(columns=[
            "Gene", "frame",
            "triplet_start_idx", "triplet_end_idx",
            "k_start", "k_end",
            "n_triplets", "orf_score", "mean_orf_score",
            "has_canonical_start", "has_non_canonical_start", "has_stop",
            "internal_n_triplets", "internal_sum_N",
            "internal_nonzero_fraction", "max_zero_gap"
        ])

    return pd.DataFrame(records)



def compute_orf_score(start_i, end_i, path, emission_logp, transition_model, p_start, frame_run):
    """
    Sum local log-scores on the chosen path for this ORF segment.
    This is NOT the cumulative Viterbi delta sum.
    """
    score = 0.0

    for t in range(start_i, end_i + 1):
        s = path[t]
        score += float(emission_logp[t, s])

        if t == 0:
            logpi0_t0 = transition_model.build_logpi0_t0(
                p_start_0=p_start[0],
                frame_run_0=frame_run[0]
            )
            score += float(logpi0_t0[s])
        else:
            logA_t = transition_model.build_logA_t(
                p_start_t=p_start[t],
                frame_run_t=frame_run[t]
            )
            score += float(logA_t[path[t - 1], s])

    return score