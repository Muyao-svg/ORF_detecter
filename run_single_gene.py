"""Fit independent HMMs to frames 0, 1, 2 of a single gene.
run_single_gene.py
    |
    |-- process_gene()
    |      |-- run_hmm_em_frame()
    |      |-- extract_orfs_from_viterbi()
    |      |-- collect fitted HMM parameters
    |
    |-- iterative_orf_discovery()
           |
           |-- HMM(original counts)
           |-- build_A_matrix()
           |-- fit_activity_poisson()
           |-- calculate residual
           |-- HMM(residual counts)
           |-- jointly re-estimate beta
           |-- evaluate_candidate_orfs()
           |-- BIC model selection
           |-- accept / stop

"""
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
from deconvolution import *

# =====================================================
# Test configuration: edit these three values, then run the script.
# MATRIC_DIR points to the input matrix file (not its parent folder).
# =====================================================
GENE = "RPS20"
MATRIC_DIR = DATA_DIR
OUT_DIR = OUTPUT_DIR
BACKGROUND = 1.0
MAX_ITER = 5
MIN_ACTIVITY = 1e-6
MIN_BIC_IMPROVEMENT = 0.0
PRINT_FULL_A = True


def process_gene(gene_name, df_gene):
    """Run existing independent three-frame HMM; return fitted params as well."""
    df_kmer = TripletData.add_k_idx_and_frame(
        df_gene, gene_name=gene_name, offset_nt=OFFSET)
    df_kmer = TripletData.add_start_type(df_kmer)
    df_kmer = TripletData.add_has_stop(df_kmer)
    fit_parts, orf_parts, pi_coding_by_frame, params_records = [], [], {}, []

    for frame in (0, 1, 2):
        print(f"\n[HMM] Gene={gene_name}, triplet offset={frame}", flush=True)
        triplets = TripletData.build_sliding_triplets(df_kmer, offset=frame)
        if triplets.empty:
            continue
        triplets = TripletData.add_x1_range_score(triplets)
        triplets = TripletData.add_x1_asin(triplets)
        triplets["frame_run"] = frame
        triplets = triplets.reset_index(drop=True)

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
            A=A_INIT_HMM.copy(), pi0=PI0_INIT_HMM.copy(),
            allowed_mask=ALLOWED_MASK.copy())
        p_start = StartLogitModel.compute_start_prob(
            triplet_df=triplets, beta0=BETA0, beta1=BETA1, beta2=BETA2)
        frame_run = triplets["frame_run"].to_numpy(dtype=int)
        hmm_state, trace = run_hmm_em_frame(
            triplet_df=triplets, emission_model=emission_model,
            transition_model=transition_model, p_start=p_start,
            frame_run=frame_run, n_iter=20, tol=1e-4,
            update_mu=True, update_r=True, update_pi=True,
            update_transition=True, pooled_transition=True,
            pseudocount=1e-6, verbose=True)

        pi_coding_by_frame[frame] = np.asarray(
            emission_model.state_params[f"C{frame}"]["pi_base"], dtype=float).copy()
        for state_name, param in emission_model.state_params.items():
            pi = np.asarray(param.get("pi_base", param.get("pi")), dtype=float)
            params_records.append({"frame_run": frame, "state": state_name,
                                   "mu_hmm": float(param["mu"]),
                                   "r_hmm": float(param["r"]),
                                   "pi1": pi[0], "pi2": pi[1], "pi3": pi[2]})
        print(f"[HMM] pi_coding(frame={frame}): {pi_coding_by_frame[frame]}")
        print(f"[HMM] final log-likelihood: {trace[-1] if len(trace) else 'NA'}")
        emission_model.emission_parameter_summary()

        fitted = triplets.copy()
        fitted["Gene"] = gene_name
        fitted["p_start"] = p_start
        fitted["state_hmm"] = [STATE_NAMES[s] for s in hmm_state.viterbi_path]
        for i, state_name in enumerate(STATE_NAMES):
            fitted[f"gamma_{state_name}"] = hmm_state.gamma[:, i]
        fit_parts.append(fitted)

        orfs = extract_orfs_from_viterbi(
            triplet_df=fitted, hmm_state=hmm_state, state_names=STATE_NAMES,
            emission_logp=emission_model.build_emission_matrix(triplets),
            transition_model=transition_model, p_start=p_start,
            frame_run=frame_run, gene_name=gene_name)
        if not orfs.empty:
            orf_parts.append(orfs)
        print(f"[HMM] candidate ORFs in offset {frame}: {len(orfs)}")

    params_df = pd.DataFrame(params_records)
    return fit_parts, orf_parts, pi_coding_by_frame, params_df



def detect_orfs(gene_name, df_counts, round_index, output_dir):
    fit_parts, orf_parts, pi, hmm_params = process_gene(gene_name, df_counts)
    candidates = pd.concat(orf_parts, ignore_index=True) if orf_parts else pd.DataFrame(
        columns=["Gene", "frame", "k_start", "k_end", "orf_score"])
    fitted = pd.concat(fit_parts, ignore_index=True) if fit_parts else pd.DataFrame()
    #candidates.to_csv(output_dir / f"round_{round_index}_candidates_{gene_name}.tsv", sep="\t", index=False)
    #fitted.to_csv(output_dir / f"round_{round_index}_hmm_fit_{gene_name}.tsv", sep="\t", index=False)
    #hmm_params.to_csv(output_dir / f"round_{round_index}_hmm_params_{gene_name}.tsv", sep="\t", index=False)
    print(f"[HMM] all candidates for round {round_index}:\n{candidates.to_string(index=False)}")
    print(f"[HMM] all state parameters (mu_hmm, r_hmm, pi):\n{hmm_params.to_string(index=False)}")
    return candidates, pi


def iterative_orf_discovery(gene_name, df_gene, output_dir,
                            background=BACKGROUND, max_iter=MAX_ITER):
    """Run original/residual HMM rounds; accept best new ORF if BIC improves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_original = TripletData.add_k_idx_and_frame(
        df_gene.copy(), gene_name=gene_name, offset_nt=OFFSET)
    k_indices = df_original["k_idx"].to_numpy(dtype=int)
    if len(np.unique(k_indices)) != len(k_indices):
        raise ValueError("Duplicated k_idx in gene")
    Y = df_original["sum"].to_numpy(dtype=float)
    if not len(Y) or np.any(Y < 0) or not np.all(np.isfinite(Y)):
        raise ValueError("Y must be nonempty, finite and nonnegative")
    K = len(Y)
    b = background_vector(background, K)
    selected = pd.DataFrame(columns=["Gene", "frame", "k_start", "k_end"])
    A_selected = np.zeros((K, 0), dtype=float)
    beta = np.empty(0, dtype=float)
    previous_loglik = poisson_loglik(Y, b)
    previous_bic = -2 * previous_loglik
    fixed_pi = None
    history = [{"round": 0, "J": 0, "loglik": previous_loglik,
                "BIC": previous_bic, "status": "background_only"}]
    stop_reason = "max_iter"


    #这里待修改，循环次数不应该由人为指定，应该是BIC 提升小于某个值时停止！！！！！！！！
    for rnd in range(1, max_iter + 1):
        print(f"\n{'='*65}\nROUND {rnd}: {'ORIGINAL' if rnd == 1 else 'RESIDUAL'} counts\n{'='*65}")
        df_input = (df_original.copy() if rnd == 1 else
                    make_residual_dataframe(df_original, k_indices, Y, A_selected, beta))
        print(f"[Input] observed original Y (K={K}):\n{Y}")
        print(f"[Input] current HMM signal:\n{df_input['sum'].to_numpy()}")
        candidates, pi_this_round = detect_orfs(gene_name, df_input, rnd, output_dir)
        if fixed_pi is None:
            fixed_pi = {int(f): np.asarray(p, float).copy() for f, p in pi_this_round.items()}
            print(f"[Deconvolution] fixed original-data coding pi: {fixed_pi}")
        # 这一轮进入BIC测试的候选ORF(hmm预测的ORF)
        # 选择BIC最低的
        # ！！！！！一开始hmm是没有overlapping的，可以直接一次性保留三个！！！ 要修改！！！
        best, trials = evaluate_candidate_orfs(
            Y, k_indices, candidates, selected, A_selected, beta,
            fixed_pi, background=b, min_activity=MIN_ACTIVITY)
        trials.to_csv(output_dir / f"round_{rnd}_trials_{gene_name}.tsv", sep="\t", index=False)
        print(f"[Model selection] all trials:\n{trials.to_string(index=False)}")
        # 如果BIC提升小于阈值，则停止循环
        if best is None or previous_bic - best["BIC"] <= MIN_BIC_IMPROVEMENT:
            stop_reason = "no_candidates" if candidates.empty else "no_bic_improvement"
            print(f"[STOP] {stop_reason}; J={len(beta)}, BIC={previous_bic:.6f}")
            history.append({"round": rnd, "J": len(beta), "loglik": previous_loglik,
                            "BIC": previous_bic, "status": stop_reason})
            break

        # 更新参数
        old_bic = previous_bic
        selected = pd.concat((selected, pd.DataFrame([best["row"]])), ignore_index=True)
        A_selected, beta = best["A"], best["beta"]
        previous_loglik, previous_bic = best["loglik"], best["BIC"]
        J = len(beta)
        history.append({"round": rnd, "J": J,
                        "added_frame": best["key"][0],
                        "added_k_start": best["key"][1],
                        "added_k_end": best["key"][2],
                        "loglik": previous_loglik, "BIC": previous_bic,
                        "delta_BIC": previous_bic - old_bic, "status": "accepted"})
        print(f"[Accepted ORF] frame,start,end: {best['key']}")
        print(f"[Activity] J={J}; beta={beta}")
        print(f"[Poisson] mu=b+A@beta:\n{best['mu']}")
        print(f"[BIC] old={old_bic:.6f}, new={previous_bic:.6f}, delta={previous_bic-old_bic:.6f}")
        if PRINT_FULL_A:
            with np.printoptions(threshold=np.inf, linewidth=180, precision=6, suppress=True):
                print(f"[Design matrix] A shape={A_selected.shape}:\n{A_selected}")
        A_df = pd.DataFrame(A_selected, columns=[f"ORF_{j+1}" for j in range(J)])
        A_df.insert(0, "k_idx", k_indices)
        A_df.to_csv(output_dir / f"round_{rnd}_A_{gene_name}.tsv", sep="\t", index=False)
        signal_table(k_indices, Y, A_selected, beta, b).to_csv(
            output_dir / f"round_{rnd}_signal_{gene_name}.tsv", sep="\t", index=False)
        selected.assign(beta=beta, selected_order=np.arange(1, J+1)).to_csv(
            output_dir / f"round_{rnd}_selected_orfs_{gene_name}.tsv", sep="\t", index=False)

    selected = selected.copy()
    selected["beta"] = beta
    selected["selected_order"] = np.arange(1, len(selected)+1)
    history_df = pd.DataFrame(history)
    final_signal = signal_table(k_indices, Y, A_selected, beta, b)
    selected.to_csv(output_dir / f"iterative_orfs_{gene_name}.tsv", sep="\t", index=False)
    history_df.to_csv(output_dir / f"iteration_history_{gene_name}.tsv", sep="\t", index=False)
    final_signal.to_csv(output_dir / f"final_signal_{gene_name}.tsv", sep="\t", index=False)
    print(f"\n[FINAL] gene={gene_name}, J={len(beta)}, beta={beta}, stop={stop_reason}")
    print(f"[FINAL] mu:\n{final_signal['mu_poisson'].to_numpy()}")
    print(f"[FINAL] selected ORFs:\n{selected.to_string(index=False)}")
    return selected, history_df, final_signal


def main():
    df = TripletData.load_kmer_table(str(MATRIC_DIR))
    df_gene = TripletData.subset_gene(df, GENE)
    iterative_orf_discovery(GENE, df_gene, OUT_DIR)


if __name__ == "__main__":
    main()