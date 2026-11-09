# run_frame_demo.py
# This is a demo script to run the unified-frame Ribohmm on one example gene, and extract candidate ORFs from the Viterbi path.
import numpy as np
import pandas as pd
import copy
import os
from concurrent.futures import ProcessPoolExecutor

from data import TripletData
from constants_frame import *
from emission_frame import EmissionModelFrame
from frame_em import *
from logistic import StartLogitModel
from transition_frame import TransitionModelFrame
from frame_hmm import (
    viterbi_decode_dynamic_frame,
    forward_log_dynamic_frame,
    backward_log_dynamic_frame,
    compute_gamma_frame
)
from viterbi import extract_orfs_from_viterbi, compute_orf_score


def process_gene(task):
    """Fit all three reading frames independently for one gene."""
    gene_name, df_gene = task
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
            print(f"  Gene {gene_name}, frame {frame}: no complete triplets", flush=True)
            continue
        triplets = TripletData.add_x1_range_score(triplets)
        triplets = TripletData.add_x1_asin(triplets)
        triplets["frame_run"] = frame
        triplets = triplets.reset_index(drop=True)

        # Each frame starts with independent emission/transition parameters.
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
        print(f"Gene {gene_name}, frame {frame} log-likelihood trace: {trace}", flush=True)
        print(f"\nGene {gene_name} | Frame {frame}", flush=True)
        emission_model.emission_parameter_summary()

        fitted = triplets.copy()
        fitted["Gene"] = gene_name
        fitted["p_start"] = p_start
        fitted["state_hmm"] = [STATE_NAMES[s] for s in hmm_state.viterbi_path]
        for i, state_name in enumerate(STATE_NAMES):
            fitted[f"gamma_{state_name}"] = hmm_state.gamma[:, i]
        fit_parts.append(fitted)

        emission_logp_final = emission_model.build_emission_matrix(triplets)
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
        print(f"  Gene {gene_name}, frame {frame}: {len(orfs)} candidate ORFs", flush=True)

    return gene_name, fit_parts, orf_parts



def main():
    print("Loading kmer table...")
    df = TripletData.load_kmer_table(
        f"{DATA_DIR}/benchmark_10000_sum_count_matrix.tsv"
    )

    id_columns = [
    col for col in df.columns
    if str(col).casefold() == "id"
]

    if len(id_columns) != 1:
        raise ValueError(
            f"Expected exactly one ID column (case-insensitive), "
            f"found: {id_columns}"
        )

    if id_columns[0] != "id":
        df = df.rename(columns={id_columns[0]: "id"})

    gene_list = (
        df["id"].astype(str).str.replace(r"_kmer_\d+$", "", regex=True).dropna().unique()
        .tolist()
    )

    tasks = []
    for gene_name in gene_list:
        try:
            df_gene = TripletData.subset_gene(df, gene_name)
        except ValueError as e:
            print(f"Warning: {e} Skipping gene {gene_name}.")
            continue

        tasks.append((gene_name, df_gene))

    all_df_fit_hmm = []
    all_candidate_orfs = []

    # 可按机器资源调整；先从 2–4 个进程开始测试
    max_workers = min(4, os.cpu_count() or 1, len(tasks)) if tasks else 1
    print(f"Processing {len(tasks)} genes with {max_workers} processes.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for gene_name,gene_hmm_results, gene_orf_results in executor.map(process_gene, tasks):
            all_df_fit_hmm.extend(gene_hmm_results)
            all_candidate_orfs.extend(gene_orf_results)
            print("Finished gene:", gene_name)

    if all_df_fit_hmm:
        df_fit_hmm = pd.concat(all_df_fit_hmm, ignore_index=True)
    else:
        df_fit_hmm = pd.DataFrame(columns=["Gene"])

    if all_candidate_orfs:
        candidate_orfs = pd.concat(all_candidate_orfs, ignore_index=True)
    else:
        candidate_orfs = pd.DataFrame(columns=[
            "Gene", "frame",
            "triplet_start_idx", "triplet_end_idx",
            "k_start", "k_end",
            "n_triplets", "orf_score", "mean_orf_score",
            "has_canonical_start", "has_non_canonical_start", 
            "has_stop",
        ])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_fit_hmm.to_csv(
        f"{OUTPUT_DIR}/hmm_fit_results_v2.txt",
        index=False,
        sep="\t"
    )

    print("\nCandidate ORFs from Viterbi path:")
    print(candidate_orfs)

    candidate_orfs.to_csv(
        f"{OUTPUT_DIR}/candidate_orfs_v2.txt",
        index=False,
        sep="\t"
    )


if __name__ == "__main__":
    main()