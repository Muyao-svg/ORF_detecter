"""Reusable annotated-CDS versus predicted-ORF evaluation utilities.

Coordinates are inclusive, and reference and prediction must use the same
coordinate system. If using k-mer starts at stride one, boundary differences
have units of nucleotides; segment coverage remains k-mer-position-level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_orfs(df: pd.DataFrame, gene_col: str, start_col: str,
                   end_col: str, prefix: str) -> pd.DataFrame:
    """Normalize endpoints to inclusive [start, end] integer intervals."""
    cols = [gene_col, start_col, end_col]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    result = pd.DataFrame({
        "gene_name": df[gene_col].astype("string").str.strip(),
        f"{prefix}_start": pd.to_numeric(df[start_col], errors="raise"),
        f"{prefix}_end": pd.to_numeric(df[end_col], errors="raise"),
    })
    if result.isna().any().any() or result["gene_name"].eq("").any():
        raise ValueError("Null or empty gene names/coordinates are not allowed")
    for field in [f"{prefix}_start", f"{prefix}_end"]:
        if not (result[field] % 1 == 0).all():
            raise ValueError(f"{field} must contain integer coordinates")
        result[field] = result[field].astype('int64')
    a = result[f"{prefix}_start"].copy()
    b = result[f"{prefix}_end"].copy()
    result[f"{prefix}_start"] = np.minimum(a, b)
    result[f"{prefix}_end"] = np.maximum(a, b)
    return result


def score_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Score one predicted interval against one reference interval per row."""
    result = df.copy()
    p0, p1 = result["pred_start"], result["pred_end"]
    r0, r1 = result["ref_start"], result["ref_end"]
    overlap = np.maximum(0, np.minimum(p1, r1) - np.maximum(p0, r0) + 1)
    pred_len, ref_len = p1 - p0 + 1, r1 - r0 + 1
    result["overlap_len"] = overlap
    result["pred_len"] = pred_len
    result["ref_len"] = ref_len
    result["precision"] = overlap / pred_len
    result["recall"] = overlap / ref_len
    result["f1_score"] = 2 * overlap / (pred_len + ref_len)
    result["iou"] = overlap / (pred_len + ref_len - overlap)
    result["start_abs_error"] = (p0 - r0).abs()
    result["stop_abs_error"] = (p1 - r1).abs()
    result["boundary_distance"] = result["start_abs_error"] + result["stop_abs_error"]
    result["start_exact"] = p0.eq(r0)
    result["stop_exact"] = p1.eq(r1)
    result["exact_match"] = result["start_exact"] & result["stop_exact"]
    return result


def evaluate_orfs(ref: pd.DataFrame, pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select closest ORF per gene and evaluate across every reference gene.

    Inputs must contain gene_name/ref_start/ref_end and
    gene_name/pred_start/pred_end, respectively. A unique reference per gene
    is required. Selection minimizes |start error| + |end error|, breaking ties
    by higher IoU, then earlier original prediction order.
    """
    if ref['gene_name'].duplicated().any():
        raise ValueError("Multiple reference ORFs for a gene: use transcript/ORF ID as key")
    unknown = sorted(set(pred['gene_name']) - set(ref['gene_name']))
    missing = sorted(set(ref['gene_name']) - set(pred['gene_name']))
    if unknown:
        raise ValueError(f"Predicted genes absent from reference: {unknown[:10]}" +
                         (f" (and {len(unknown)-10} more)" if len(unknown)>10 else ""))
    scored = score_intervals(pred.merge(ref, on='gene_name', how='left', validate='many_to_one'))
    scored['_original_order'] = np.arange(len(scored))
    chosen = (scored.sort_values(
        ['gene_name', 'boundary_distance', 'iou', '_original_order'],
        ascending=[True, True, False, True], kind='stable'
    ).drop_duplicates('gene_name').drop(columns='_original_order'))
    result = ref.merge(chosen.drop(columns=['ref_start', 'ref_end']),
                       on='gene_name', how='left', validate='one_to_one')
    result['detected'] = result['pred_start'].notna()
    for col in ['start_exact','stop_exact','exact_match']:
        result[col] = result[col].fillna(False).astype(bool)
    # Missing predictions have zero region overlap, hence F1/IoU/Recall = 0.
    for col in ['recall','f1_score','iou']:
        result[col] = result[col].fillna(0.0)
    n = len(ref)
    detected = int(result['detected'].sum())
    exact = int(result['exact_match'].sum())
    metrics = {
        'n_reference': n,
        'n_predicted_genes': pred['gene_name'].nunique(),
        'n_predicted_orfs': len(pred),
        'n_missing_predictions': len(missing),
        'n_detected': detected,
        'detection_rate': detected/n if n else np.nan,
        'n_exact_cds': exact,
        'exact_cds_recovery': exact/n if n else np.nan,
        'n_start_exact': int(result['start_exact'].sum()),
        'start_exact_rate': result['start_exact'].mean(),
        'n_stop_exact': int(result['stop_exact'].sum()),
        'stop_exact_rate': result['stop_exact'].mean(),
        # Mean across all reference genes, missing calls have zero overlap.
        'mean_recall': result['recall'].mean(),
        'mean_f1': result['f1_score'].mean(),
        'mean_iou': result['iou'].mean(),
        # Precision not defined for a gene with no predicted interval.
        'mean_precision_detected': result['precision'].mean(),
        # Location error is only defined for detected genes.
        'start_mae_detected': result['start_abs_error'].mean(),
        'stop_mae_detected': result['stop_abs_error'].mean(),
    }
    return scored.drop(columns='_original_order'), result, metrics


def format_report(metrics: dict) -> str:
    lines = ["ORF evaluation (closest candidate per reference gene)",
             "Selection: min(|pred_start-ref_start| + |pred_end-ref_end|); tie-break: max IoU", ""]
    for k, v in metrics.items():
        lines.append(f"{k:26s} : {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k:26s} : {v}")
    lines += ["", "Notes:",
              "* exact_cds_recovery = exact matches / all reference genes (positive-set recall, NOT binary accuracy).",
              "* mean_recall / mean_f1 / mean_iou include undetected genes as 0.",
              "* mean_precision_detected and start/stop MAE exclude undetected genes.",
              "* Coordinates are inclusive. Boundary MAE has nt units only if coordinate stride is 1 nt.",
              "* Closest-candidate selection uses reference labels (oracle); not deployable Top-1 performance."]
    return '\n'.join(lines) + '\n'
