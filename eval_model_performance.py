"""Run evaluation for the 110 annotated genes; edit paths here as needed."""
from pathlib import Path
import pandas as pd
import numpy as np
from evaluate_metrics import (
    normalize_orfs, evaluate_orfs, frame_classification_metrics, format_report
)


PRED_FILE = Path('/stockage/plateformes/I2BC/CBNG_Muyao-thesis/Ribokast_hmm/Segmentation_signal/Outputs/Benchmark/candidate_orfs_benchmarkset.txt')
REF_FILE = Path('/stockage/plateformes/I2BC/CBNG_Muyao-thesis/Ribokast_hmm/Data/Benchmark/benchmark_100nt_ref.tsv')
OUT = Path('/stockage/plateformes/I2BC/CBNG_Muyao-thesis/Ribokast_hmm/Segmentation_signal/Outputs/Benchmark/Evaluation')

OFFSET_NT = 12

print('Loading data...')
pred_ORF = pd.read_csv(PRED_FILE, sep='\t')
ref_ORF = pd.read_csv(REF_FILE, sep=r'\s+')

# Check that the prediction file has a frame column
if 'frame' not in pred_ORF.columns:
    raise ValueError('Prediction file must include frame (0, 1, 2)')
print('Prediction ORFs:\n', pred_ORF.head())
print('Reference ORFs:\n', ref_ORF.head())

# 1. Normalize the ORFs to have the same column names and formats
ref = normalize_orfs(ref_ORF, 'gene_name', 'cds_start_kmer', 'cds_end_kmer', 'ref')
pred = normalize_orfs(pred_ORF, 'Gene', 'k_start', 'k_end', 'pred', frame_col='frame')

print('Unknown predicted genes:', sorted(set(pred.gene_name) - set(ref.gene_name)))
print('Reference genes without predictions:', sorted(set(ref.gene_name) - set(pred.gene_name)))

# 2. Evaluate the predictions against the reference
all_candidates, selected, metrics = evaluate_orfs(ref, pred, offset_nt=OFFSET_NT)
frame_table, _ = frame_classification_metrics(selected)


# 3. Save the results
OUT.mkdir(parents=True, exist_ok=True)
all_candidates.to_csv(OUT / 'all_candidate_scores.tsv', sep='\t', index=False)
selected.to_csv(OUT / 'selected_orfs.tsv', sep='\t', index=False)
if not frame_table.empty:
    frame_table.to_csv(OUT / 'frame_metrics.tsv', sep='\t', index=False)
pd.DataFrame(list(metrics.items()), columns=['metric', 'value']).to_csv(
    OUT / 'evaluation_metrics.tsv', sep='\t', index=False
)
report = format_report(metrics)
(OUT / 'evaluation_report.txt').write_text(report, encoding='utf-8')
print(report)
print('Saved to:', OUT)


# ============================================================
# Positive vs Negative benchmark evaluation
# ============================================================

print("\n========== Positive / Negative Evaluation ==========")

# 1. All reference fragments
group_eval = ref_ORF[["gene_name", "category", "label"]].copy()    # ex. f000001  TIS Positive
group_eval = group_eval.drop_duplicates(subset="gene_name")

# 2. Whether the model predicted at least one ORF
detected_ids = set(pred_ORF["Gene"].dropna())
group_eval["detected"] = group_eval["gene_name"].isin(detected_ids)

# 3. Add evaluation metrics of selected ORFs
metric_cols = [
    "gene_name",
    "frame_correct",
    "start_exact",
    "stop_exact",
    "exact_match",
    "precision",
    "recall",
    "iou",
    "f1_score",
    "start_abs_error",
    "stop_abs_error"
]

available_cols = [
    col for col in metric_cols
    if col in selected.columns
]

group_eval = group_eval.merge(
    selected[available_cols],
    on="gene_name",
    how="left",
    validate="one_to_one"
)

# 4. Positive and negative groups
positive = group_eval[group_eval["label"] == "positive"].copy()
negative = group_eval[group_eval["label"] == "negative"].copy()

summary = group_eval.groupby("label").agg(
    n_total=("gene_name", "size"),
    n_detected=("detected", "sum"))

summary["n_undetected"] = (summary["n_total"] - summary["n_detected"])
summary["detection_rate"] = (summary["n_detected"] / summary["n_total"])

print("\n--- Detection by group ---")
print(summary)

# ============================================================
# Binary classification
# ============================================================

#TP = int(positive["detected"].sum())
#FN = int((~positive["detected"]).sum())
#
#FP = int(negative["detected"].sum())
#TN = int((~negative["detected"]).sum())
#
#accuracy = (TP + TN) / (TP + TN + FP + FN)
#precision = (TP / (TP + FP) if TP + FP > 0 else 0)
#recall = ( TP / (TP + FN) if TP + FN > 0 else 0)
#specificity = (TN / (TN + FP) if TN + FP > 0 else 0)
#f1 = (2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN > 0 else 0)
#
#print("\n--- Binary classification ---")
#print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
#print(f"Accuracy:    {accuracy:.4f}")
#print(f"Precision:   {precision:.4f}")
#print(f"Recall:      {recall:.4f}")
#print(f"Specificity: {specificity:.4f}")
#print(f"F1-score:   {f1:.4f}")

# ============================================================
# Positive group: ORF prediction quality
# ============================================================

print("\n--- Positive ORF quality ---")
# Keep only positive fragments with selected ORFs
positive_detected = positive[positive["detected"]].copy()

n_positive = len(positive)
n_detected = len(positive_detected)

positive_metrics = {
    "n_total": n_positive,
    "n_detected": n_detected,
    "detection_rate": (
        n_detected / n_positive if n_positive else np.nan
    ),

    # Frame
    "n_frame_correct": int(
        positive_detected["frame_correct"].fillna(False).sum()
    ),

    # Boundary accuracy
    "n_start_exact": int(
        positive_detected["start_exact"].fillna(False).sum()
    ),

    "n_stop_exact": int(
        positive_detected["stop_exact"].fillna(False).sum()
    ),

    "n_exact_cds": int(
        positive_detected["exact_match"].fillna(False).sum()
    ),

    # Interval-level metrics: selected ORFs only
    "mean_precision": positive_detected["precision"].mean(),
    "mean_recall": positive_detected["recall"].mean(),
    "mean_iou": positive_detected["iou"].mean(),
    "mean_f1": positive_detected["f1_score"].mean(),

    # Boundary errors: selected ORFs only
    "start_mae_detected": positive_detected["start_abs_error"].mean(),
    "stop_mae_detected": positive_detected["stop_abs_error"].mean()
}

# Frame accuracy among detected positive fragments
positive_metrics["frame_accuracy"] = (
    positive_metrics["n_frame_correct"] / n_detected
    if n_detected else np.nan
)

# Exact boundary rates among detected positive fragments
positive_metrics["start_exact_rate"] = (
    positive_metrics["n_start_exact"] / n_detected
    if n_detected else np.nan
)

positive_metrics["stop_exact_rate"] = (
    positive_metrics["n_stop_exact"] / n_detected
    if n_detected else np.nan
)

positive_metrics["exact_cds_recovery"] = (
    positive_metrics["n_exact_cds"] / n_detected
    if n_detected else np.nan
)

for key, value in positive_metrics.items():
    print(f"{key:28s}: {value:.4f}")

# ============================================================
# Negative group: False-positive detection
# ============================================================

print("\n--- Negative ORF detection ---")

negative_metrics = {
    "n_total": len(negative),
    "n_false_positive": int(negative["detected"].sum()),
    "n_true_negative": int((~negative["detected"]).sum()),
    "false_positive_rate": negative["detected"].mean(),
    "specificity": (~negative["detected"]).mean()
}

for key, value in negative_metrics.items():
    print(f"{key:25s}: {value:.4f}")

# ============================================================
# Save Positive / Negative evaluation report
# ============================================================

OUT.mkdir(parents=True, exist_ok=True)

# 1. Save fragment-level results
group_eval.to_csv(OUT / "positive_negative_fragment_results.tsv",sep="\t",index=False)

# 2. Save group detection summary
summary.to_csv(OUT / "positive_negative_detection_summary.tsv",sep="\t")

# 3. Combine all metrics into one report
report_lines = [
    "========== Positive / Negative Evaluation ==========",
    "",
    "========== Detection by Group ==========",
    summary.to_string(),
    "",
  # "========== Binary Classification ==========",
  # f"TP: {TP}",
  # f"FP: {FP}",
  # f"TN: {TN}",
  # f"FN: {FN}",
  # f"Accuracy:    {accuracy:.4f}",
  # f"Precision:   {precision:.4f}",
  # f"Recall:      {recall:.4f}",
  # f"Specificity: {specificity:.4f}",
  # f"F1-score:   {f1:.4f}",
  # "",
    "========== Positive ORF Quality ==========",
]

for key, value in positive_metrics.items():
    report_lines.append(f"{key:25s}: {value:.4f}")

report_lines.extend([
    "",
    "========== Negative ORF Detection =========="
])

for key, value in negative_metrics.items():
    report_lines.append(f"{key:25s}: {value:.4f}")

report_text = "\n".join(report_lines) + "\n"

# 4. Save one combined report
with open(OUT / "evaluation_report.txt", "a", encoding="utf-8") as f:
    f.write("\n\n")
    f.write(report_text)

print(report_text)
print("Report saved to:", OUT / "evaluation_report.txt")
