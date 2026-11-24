# Pseudocode

from 

Y_original = Y.copy()
selected_orfs = []

while True:

    # 1. Detect new ORF candidates from residual signal
    candidates = run_hmm(residual_for_hmm)

    # 2. Build candidate A columns
    A_candidates = build_A(candidates)

    # 3. Jointly refit activities on ORIGINAL counts
    beta = fit_activity_poisson(
        Y_original, A_trial, background
    )

    # 4. Compare model selection criterion
    if new_model_is_better:
        accept_candidate()
    else:
        break

    # 5. Update residual for next discovery
    residual = Y_original - A_selected @ beta
    residual_for_hmm = np.rint(
        np.maximum(residual, 0)
    ).astype(int)