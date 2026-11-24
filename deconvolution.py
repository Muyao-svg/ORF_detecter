"""
kmer index: 1 based
frame/phase: 0,1,2
"""
import numpy as np

from constants_frame import *


# 先只用coding state作为测试
def build_A_matrix(k_indices, candidate_orfs, pi_coding):
    """
    Build ORF activity design matrix.

    k_indices: ordered global k-mer indices
    candidate_orfs: DataFrame with k_start, k_end, frame
    pi_coding: coding-state multinomial template
    """
    k_indices = np.asarray(k_indices, dtype=int)
    pi_coding = np.asarray(pi_coding, dtype=float)

    if pi_coding.shape != (3,):
        raise ValueError("pi_coding must have length 3")

    A = np.zeros((len(k_indices), len(candidate_orfs)))

    for j, (_, orf) in enumerate(candidate_orfs.iterrows()):
        start = int(orf["k_start"])
        end = int(orf["k_end"])
        frame = int(orf["frame"])

        mask = (k_indices >= start) & (k_indices <= end)

        # Existing code defines frame using OFFSET.
        local_frame = (k_indices[mask] - 1 + OFFSET - frame) % 3

        A[mask, j] = pi_coding[local_frame]

    return A




