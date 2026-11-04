"""Fit independent HMMs to frames 0, 1, 2 of a single gene."""
import copy
from pathlib import Path

import pandas as pd

from constants_frame import *
from data import TripletData
from emission_frame import EmissionModelFrame
from frame_em import run_hmm_em_frame
from logistic import StartLogitModel
from transition_frame import TransitionModelFrame
from viterbi import extract_orfs_from_viterbi

# =====================================================
# Test configuration: edit these three values, then run the script.
# MATRIC_DIR points to the input matrix file (not its parent folder).
# =====================================================
GENE = "RPS20"
MATRIC_DIR = f"{DATA_DIR}/KmersFromContigsQuerySum"
OUT_DIR = OUTPUT_DIR


def process_gene(gene_name, df_gene):
    """Return separate fitted tables and ORF predictions for each frame."""
    df_kmer = TripletData.add_k_idx_and_frame(
        df_gene, gene_name=gene_name, offset_nt=OFFSET
    )
    df_kmer = TripletData.add_start_type(df_kmer)
    df_kmer = TripletData.add_has_stop(df_kmer)
    fit_parts, orf_parts = [], []

    for frame in (0, 1, 2):
        print(f"Gene {gene_name}: running frame {frame}...", flush=True)
        triplets = TripletData.build_sliding_triplets(df_kmer, offset=frame)
        if triplets.empty:
            print(f"  Frame {frame}: no complete triplets")
            continue
        triplets = TripletData.add_x1_range_score(triplets)
        triplets = TripletData.add_x1_asin(triplets)
        triplets["frame_run"] = frame
        triplets = triplets.reset_index(drop=True)

        # Initialize independent parameters for each reading frame.
        emission_model = EmissionModelFrame(
            state_params=copy.deepcopy(STATE_PARAMS),
            states=STATE_NAMES,
            lambda_canonical_start=LAMBDA_CANONICAL_START,
            lambda_non_canonical_start=LAMBDA_NON_CANONICAL_START,
            lambda_no_start=LAMBDA_NO_START,
            lambda_stop=LAMBDA_STOP,
            lambda_no_stop=LAMBDA_NO_STOP,
        )
        transition_model = TransitionModelFrame(
            states=STATE_NAMES,
            A=A_INIT_HMM.copy(),
            pi0=PI0_INIT_HMM.copy(),
            allowed_mask=ALLOWED_MASK.copy(),
        )
        p_start = StartLogitModel.compute_start_prob(
            triplet_df=triplets, beta0=BETA0, beta1=BETA1, beta2=BETA2
        )
        frame_run = triplets["frame_run"].to_numpy(dtype=int)
        hmm_state, trace = run_hmm_em_frame(
            triplet_df=triplets,
            emission_model=emission_model,
            transition_model=transition_model,
            p_start=p_start,
            frame_run=frame_run,
            n_iter=20,
            tol=1e-4,
            update_mu=True,
            update_r=True,
            update_pi=True,
            update_transition=True,
            pooled_transition=True,
            pseudocount=1e-6,
            verbose=True,
        )
        print(f"Frame {frame} log-likelihood trace: {trace}")
        fitted = triplets.copy()
        fitted["Gene"] = gene_name
        fitted["p_start"] = p_start
        fitted["state_hmm"] = [STATE_NAMES[s] for s in hmm_state.viterbi_path]
        for i, state_name in enumerate(STATE_NAMES):
            fitted[f"gamma_{state_name}"] = hmm_state.gamma[:, i]
        fit_parts.append(fitted)

        emission_logp_final = emission_model.build_emission_matrix(triplets)

        print(f"\nGene {gene_name} | Frame {frame}")
        # Check the final emission parameters after EM convergence.
        emission_model.emission_parameter_summary()

        orfs = extract_orfs_from_viterbi(
            triplet_df=fitted,
            hmm_state=hmm_state,
            state_names=STATE_NAMES,
            emission_logp=emission_logp_final,
            transition_model=transition_model,
            p_start=p_start,
            frame_run=frame_run,
            gene_name=gene_name,
        )
        if not orfs.empty:
            orf_parts.append(orfs)
        print(f"  Frame {frame}: {len(orfs)} candidate ORFs")

    return fit_parts, orf_parts


def main():
    df = TripletData.load_kmer_table(str(MATRIC_DIR))
    df_gene = TripletData.subset_gene(df, GENE)
    fit_parts, orf_parts = process_gene(GENE, df_gene)
    if not fit_parts:
        raise ValueError(f"No complete triplets available for gene {GENE}")

    fitted = pd.concat(fit_parts, ignore_index=True)
    fitted = fitted.sort_values(["start_kidx", "frame_run"], kind="stable").reset_index(drop=True)
    if orf_parts:
        orfs = pd.concat(orf_parts, ignore_index=True)
        orfs = orfs.sort_values(["k_start", "frame"], kind="stable").reset_index(drop=True)
    else:
        orfs = pd.DataFrame(columns=[
            "Gene", "frame", "triplet_start_idx", "triplet_end_idx",
            "k_start", "k_end", "n_triplets", "orf_score", "mean_orf_score",
            "has_canonical_start", "has_non_canonical_start", "has_stop",
        ])

    # Note: triplet_start_idx and triplet_end_idx are frame-local indices.
    output_dir = Path(OUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted.to_csv(output_dir / f"df_fit_hmm_{GENE}.txt", index=False, sep="\t")
    orfs.to_csv(output_dir / f"candidate_orfs_{GENE}.txt", index=False, sep="\t")
    print(f"Finished {GENE}: {len(orfs)} candidate ORFs from three frames")


if __name__ == "__main__":
    main()
