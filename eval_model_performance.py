# Category == "TIS", whether the model find exact start: Exact accuracy, mean absolute error, detection rate
# positive = "TIS" + "CDS", negative = "5/3UTR". Accuracy, precision, recall, f1 score
# How many ORFs are detected in 4 regions: 5'UTR, CDS, 3'UTR, TIS

import pandas as pd
import numpy as np

# -----------------------------
# 1. Load data
# -----------------------------
benchmark = pd.read_csv("/stockage/plateformes/I2BC/CBNG_Muyao-thesis/Ribokast_hmm/Data/Benchmark/benchmark_100nt.tsv", sep="\t")
pred = pd.read_csv("/stockage/plateformes/I2BC/CBNG_Muyao-thesis/Ribokast_hmm/Segmentation_signal/Outputs/Benchmark/candidate_orfs_benchmarkset.txt", sep="\t")

# -----------------------------
# 2. Parse IDs and truth
# -----------------------------
pred["fragment_id"] = pred["Gene"].str.split("|", regex=False).str[0] # ex. F00001
pred["pred_region"] = pred["Gene"].str.split("|", regex=False).str[-1] # ex. 5UTR, CDS, 3UTR, TIS

# true TIS position inside fragment (1-based nt)
benchmark["true_tis_nt"] = benchmark["cds_start"] - benchmark["start"] + 1

# true label for binary classification; 1 = positive = TIS or CDS, 0 = negative = 5UTR or 3UTR
benchmark["true_label"] = benchmark["category"].map(
    lambda x: 1 if x in ("TIS", "CDS") else 0
)

# predicted start nucleotide from k_start
pred["pred_start_nt"] = pred["k_start"] + 12

# merge predictions with benchmark
merged = pred.merge(
    benchmark,
    on="fragment_id",
    how="left",
    validate="many_to_one"
)

# -----------------------------
# 3. Region-level detection rate
# -----------------------------
detected_frags = set(pred["fragment_id"].unique())

benchmark["detected"] = benchmark["fragment_id"].isin(detected_frags)

region_summary = (
    benchmark.groupby("category")
    .agg(
        n_total=("fragment_id", "size"),
        n_detected=("detected", "sum")
    )
    .reset_index()
)

region_summary["detection_rate"] = (
    region_summary["n_detected"] / region_summary["n_total"]
)

print("\n=== Detection rate by region ===")
print(region_summary)

# -----------------------------
# 4. Fragment-level binary classification
#    predicted positive = at least one candidate
# -----------------------------
frag_pred = (
    benchmark[["fragment_id", "category", "true_label"]]
    .copy()
)

frag_pred["pred_label"] = frag_pred["fragment_id"].isin(detected_frags).astype(int)

TP = ((frag_pred["true_label"] == 1) & (frag_pred["pred_label"] == 1)).sum()
TN = ((frag_pred["true_label"] == 0) & (frag_pred["pred_label"] == 0)).sum()
FP = ((frag_pred["true_label"] == 0) & (frag_pred["pred_label"] == 1)).sum()
FN = ((frag_pred["true_label"] == 1) & (frag_pred["pred_label"] == 0)).sum()

accuracy = (TP + TN) / len(frag_pred)
precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
balanced_acc = (recall + specificity) / 2

print("\n=== Fragment-level binary classification ===")
print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
print(f"Accuracy          : {accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity       : {specificity:.4f}")
print(f"F1-score          : {f1:.4f}")
print(f"Balanced accuracy : {balanced_acc:.4f}")

# -----------------------------
# 5. TIS localization: ANY-HIT rule
# -----------------------------
tis_truth = benchmark[benchmark["category"] == "TIS"].copy()

tis_pred = merged[merged["category"] == "TIS"].copy()

# Calculate nucleotide position error
tis_pred["start_error"] = (
    tis_pred["pred_start_nt"] - tis_pred["true_tis_nt"]
)

tis_pred["abs_error"] = tis_pred["start_error"].abs()

# For each fragment, choose the candidate closest to true TIS
anyhit = (
    tis_pred.groupby("fragment_id")
    .agg(
        exact_hit=("abs_error", lambda x: (x == 0).any()),
        best_abs_error=("abs_error", "min"),
        n_candidates=("fragment_id", "size")
    )
    .reset_index()
)

# Include fragments without predictions
anyhit = tis_truth[["fragment_id"]].merge(
    anyhit,
    on="fragment_id",
    how="left"
)

anyhit["exact_hit"] = anyhit["exact_hit"].fillna(False)
anyhit["n_candidates"] = anyhit["n_candidates"].fillna(0)
anyhit["detected"] = anyhit["n_candidates"] > 0

# Mean absolute nucleotide error
mean_nt_error = anyhit["best_abs_error"].mean()

print("\n=== TIS localization (ANY-HIT) ===")
print(f"Detection rate       : {anyhit['detected'].mean():.4f}")
print(f"Exact TIS accuracy   : {anyhit['exact_hit'].mean():.4f}")
print(f"Mean nt error (MAE)  : {mean_nt_error:.2f} nt")

# -----------------------------
# 6. TIS localization: TOP-1 rule
#    Keep highest-scoring candidate per fragment
# -----------------------------

top1 = (
    tis_pred.sort_values(
        ["fragment_id", "orf_score"],
        ascending=[True, False]
    )
    .drop_duplicates("fragment_id")
    .copy()
)

# Calculate nucleotide position error
top1["start_error"] = (
    top1["pred_start_nt"] - top1["true_tis_nt"]
)

top1["abs_error"] = top1["start_error"].abs()
top1["exact_hit"] = top1["abs_error"] == 0

# Include undetected TIS fragments
top1 = tis_truth[["fragment_id"]].merge(
    top1[
        ["fragment_id", "pred_start_nt",
         "start_error", "abs_error", "exact_hit"]
    ],
    on="fragment_id",
    how="left"
)

top1["detected"] = top1["abs_error"].notna()
top1["exact_hit"] = top1["exact_hit"].fillna(False)

# Mean absolute nucleotide error
mean_nt_error = top1["abs_error"].mean()

print("\n=== TIS localization (TOP-1 by orf_score) ===")
print(f"Detection rate      : {top1['detected'].mean():.4f}")
print(f"Exact TIS accuracy  : {top1['exact_hit'].mean():.4f}")
print(f"Mean nt error (MAE) : {mean_nt_error:.2f} nt")