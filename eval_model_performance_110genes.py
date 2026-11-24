
import pandas as pd
from pathlib import Path

from evaluate_metrics import (
    normalize_orfs,
    evaluate_orfs,
    frame_classification_metrics,
    format_report
)

# ============================================================
# 0. File paths
# ============================================================

BASE = Path(
    "/stockage/plateformes/I2BC/CBNG_Muyao-thesis/"
    "Ribokast_hmm/Segmentation_signal"
)

PRED_FILE = BASE / "Outputs/Human_gene_110/candidate_orfs_v2.txt"

REF_FILE = (
    BASE.parent / "Data/data_i2bc/110_gene_position_ref.txt"
)

OUT = BASE / "Outputs/Human_gene_110/Evaluation"
OUT.mkdir(parents=True, exist_ok=True)

OFFSET_NT = 12

# ============================================================
# 1. Load data
# ============================================================

print("Loading data...")

pred_ORF = pd.read_csv(PRED_FILE, sep="\t")
ref_ORF = pd.read_csv(REF_FILE, sep=r"\s+")

print("\nPrediction ORFs:")
print(pred_ORF.head())

print("\nReference ORFs:")
print(ref_ORF.head())

# ============================================================
# 2. Normalize ORF coordinates
# ============================================================

ref = normalize_orfs(
    ref_ORF,
    gene_col="gene_name",
    start_col="cds_start_kmer",
    end_col="cds_end_kmer",
    prefix="ref"
)

pred = normalize_orfs(
    pred_ORF,
    gene_col="Gene",
    start_col="k_start",
    end_col="k_end",
    prefix="pred",
    frame_col="frame"
)

# ============================================================
# 3. Check reference and predicted genes
# ============================================================

ref_genes = set(ref["gene_name"])
pred_genes = set(pred["gene_name"])

unknown_genes = pred_genes - ref_genes
missing_genes = ref_genes - pred_genes

print("\n========== Gene Validation ==========")

print(f"Reference genes: {len(ref_genes)}")
print(f"Predicted genes: {len(pred_genes)}")
print(f"Unknown predicted genes: {len(unknown_genes)}")
print(f"Genes without predictions: {len(missing_genes)}")

if unknown_genes:
    print("Unknown genes:", sorted(unknown_genes))

if missing_genes:
    print("Missing genes:", sorted(missing_genes))

# evaluate_orfs requires predictions to exist in reference
pred = pred.loc[
    pred["gene_name"].isin(ref_genes)
].copy()

# ============================================================
# 4. Evaluate ORFs
# ============================================================

all_candidates, selected, metrics = evaluate_orfs(
    ref,
    pred,
    offset_nt=OFFSET_NT
)

# ============================================================
# 5. Frame classification
# ============================================================

frame_table = pd.DataFrame()

if "pred_frame" in selected.columns:
    frame_table, frame_metrics = (
        frame_classification_metrics(selected)
    )

# ============================================================
# 6. Print results
# ============================================================

print("\n========== ORF Detection ==========")

print(f"Reference genes: {metrics['n_reference']}")
print(f"Detected genes: {metrics['n_detected']}")
print(f"Undetected genes: {metrics['n_undetected']}")
print(f"Detection rate: {metrics['detection_rate']:.4f}")

print("\n========== ORF Quality ==========")

print(
    f"Mean precision: "
    f"{metrics['mean_precision_detected']:.4f}"
)

print(
    f"Mean recall: "
    f"{metrics['mean_recall_detected']:.4f}"
)

print(
    f"Mean F1: "
    f"{metrics['mean_f1_detected']:.4f}"
)

print(
    f"Mean IoU: "
    f"{metrics['mean_iou_detected']:.4f}"
)

print("\n========== Boundary Accuracy ==========")

print(
    f"Start exact rate: "
    f"{metrics['start_exact_rate_detected']:.4f}"
)

print(
    f"Stop exact rate: "
    f"{metrics['stop_exact_rate_detected']:.4f}"
)

print(
    f"Exact CDS rate: "
    f"{metrics['exact_cds_rate_detected']:.4f}"
)

print(
    f"Start MAE: "
    f"{metrics['start_mae_detected']:.4f}"
)

print(
    f"Stop MAE: "
    f"{metrics['stop_mae_detected']:.4f}"
)

print("\n========== Frame Accuracy ==========")

if "frame_accuracy" in metrics:
    print(
        f"Frame accuracy: "
        f"{metrics['frame_accuracy']:.4f}"
    )

    print("\nPer-frame classification:")
    print(frame_table)

# ============================================================
# 7. Save results
# ============================================================

# All candidate ORFs with evaluation scores
all_candidates.to_csv(
    OUT / "all_candidate_scores.tsv",
    sep="\t",
    index=False
)

# Best matching ORF per gene
selected.to_csv(
    OUT / "selected_orfs.tsv",
    sep="\t",
    index=False
)

# All summary metrics
pd.DataFrame(
    list(metrics.items()),
    columns=["metric", "value"]
).to_csv(
    OUT / "evaluation_metrics.tsv",
    sep="\t",
    index=False
)

# Frame classification
if not frame_table.empty:
    frame_table.to_csv(
        OUT / "frame_metrics.tsv",
        sep="\t",
        index=False
    )

# Text report
report = format_report(metrics)

(OUT / "evaluation_report.txt").write_text(
    report,
    encoding="utf-8"
)

print("\n========== Evaluation Completed ==========")
print(report)
print("Results saved to:", OUT)
