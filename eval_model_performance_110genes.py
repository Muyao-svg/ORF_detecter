# 打印前几行数据
# 检查pred_ORF的基因是否在ref_ORF中
# 检查是否所有基因都预测出了ORF
# 对于拥有多个ORF的基因，采取距离ref_ORF最近的ORF作为预测结果
# 计算CDS exact match的数量, accuracy, recall
# 计算IoU, precision, F1 score
# 计算start exact match的数量
# 计算stop exact match的数量
# 计算平均绝对误差（MAE）对于start和stop位置
# 输出报告

import pandas as pd
import numpy as np
import os
from pathlib import Path

from evaluate_metrics import *


# 0. Load data
print("Loading data...")
pred_ORF = pd.read_csv("/stockage/plateformes/I2BC/CBNG_Muyao-thesis/Ribokast_hmm/Segmentation_signal/Outputs/Human_gene_110/candidate_orfs_v2.txt", sep="\t")
ref_ORF = pd.read_csv("/stockage/plateformes/I2BC/CBNG_Muyao-thesis/Ribokast_hmm/Data/data_i2bc/110_gene_position_ref.txt", sep="\s+")

# 打印前几行数据
print("Prediction ORF data:")
print(pred_ORF.head())
print("\nReference ORF data:")
print(ref_ORF.head())


# =====================================================
# 1. Normalize ORF data
# =====================================================

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
    prefix="pred"
)

# =====================================================
# 2. Check genes
# =====================================================

pred_genes = set(pred["gene_name"])
ref_genes = set(ref["gene_name"])

unknown_genes = pred_genes - ref_genes
missing_genes = ref_genes - pred_genes

print("\n=== Gene validation ===")
print("Reference genes:", len(ref_genes))
print("Predicted genes:", len(pred_genes))

print("Predicted genes absent from reference:", unknown_genes)
print("Reference genes without predictions:", missing_genes)

# =====================================================
# 3. Calculate evaluation metrics
# =====================================================

all_candidates, selected_orfs, metrics = evaluate_orfs(ref,pred)

# evaluate_orfs automatically:
# 1. Scores all candidate ORFs
# 2. Selects closest ORF to reference per gene
# 3. Calculates exact matches, Recall, F1, IoU, MAE
# 4. Includes genes without predictions

# =====================================================
# 4. Print results
# =====================================================

print("\n=== ORF evaluation ===")

print("Exact CDS matches:", metrics["n_exact_cds"])
print("Exact CDS recovery:", metrics["exact_cds_recovery"])

print("Mean Recall:", metrics["mean_recall"])
print("Mean Precision:", metrics["mean_precision_detected"])
print("Mean F1-score:", metrics["mean_f1"])
print("Mean IoU:", metrics["mean_iou"])

print("Start exact matches:", metrics["n_start_exact"])
print("Stop exact matches:", metrics["n_stop_exact"])

print("Start MAE (nt):", metrics["start_mae_detected"])
print("Stop MAE (nt):", metrics["stop_mae_detected"])

# =====================================================
# 5. Save results
# =====================================================

OUT = Path(
    "/stockage/plateformes/I2BC/CBNG_Muyao-thesis/"
    "Ribokast_hmm/Segmentation_signal/Outputs/"
    "Human_gene_110/Evaluation"
)

OUT.mkdir(parents=True, exist_ok=True)

# 所有预测的ORF及其评分, recall, precision, F1, IoU, MAE
all_candidates.to_csv(
    OUT / "all_candidate_scores.tsv",
    sep="\t",
    index=False
)

# 最接近参考ORF的预测ORF
selected_orfs.to_csv(
    OUT / "selected_orfs.tsv",
    sep="\t",
    index=False
)

# 评估指标
pd.DataFrame(
    list(metrics.items()),
    columns=["metric", "value"]
).to_csv(
    OUT / "evaluation_metrics.tsv",
    sep="\t",
    index=False
)

report = format_report(metrics)

with open(OUT / "evaluation_report.txt", "w") as f:
    f.write(report)

print("\nEvaluation completed.")
print("Results saved to:", OUT)