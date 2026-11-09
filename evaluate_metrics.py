"""Reusable ORF evaluation with frame classification and boundary metrics.

Intervals are inclusive k-mer positions. Coordinates must share an origin and
stride. Boundary errors are nt only when k-mer step is 1 nucleotide.

Selection is reference-guided (min absolute start + end error), an oracle
benchmark rather than a deployed candidate-ranking metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_orfs(df: pd.DataFrame, gene_col: str, start_col: str,
                   end_col: str, prefix: str, frame_col: str = "frame") -> pd.DataFrame:
    """Normalize endpoint columns; retain candidate frame for prediction data."""
    needed = {gene_col, start_col, end_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result = pd.DataFrame({
        "gene_name": df[gene_col].astype("string").str.strip(),
        f"{prefix}_start": pd.to_numeric(df[start_col], errors="raise"),
        f"{prefix}_end": pd.to_numeric(df[end_col], errors="raise"),
    })
    if result.isna().any().any() or result["gene_name"].eq("").any():
        raise ValueError("Null or empty gene names/coordinates are not allowed")
    for col in (f"{prefix}_start", f"{prefix}_end"):
        if not (result[col] % 1 == 0).all():
            raise ValueError(f"{col} must contain integer coordinates")
        result[col] = result[col].astype("int64")
    a = result[f"{prefix}_start"].copy()
    b = result[f"{prefix}_end"].copy()
    result[f"{prefix}_start"] = np.minimum(a, b)
    result[f"{prefix}_end"] = np.maximum(a, b)

    if prefix == "pred" and frame_col in df.columns:
        frame = pd.to_numeric(df[frame_col], errors="raise")
        if frame.isna().any() or not frame.isin((0, 1, 2)).all():
            raise ValueError("Prediction frame must be 0, 1 or 2")
        result["pred_frame"] = frame.astype(int)
    return result


def score_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ORF coverage and exact boundary metrics per candidate."""
    result = df.copy()
    p0, p1 = result["pred_start"], result["pred_end"]
    r0, r1 = result["ref_start"], result["ref_end"]
    overlap = np.maximum(0, np.minimum(p1, r1) - np.maximum(p0, r0) + 1)
    pred_len, ref_len = p1 - p0 + 1, r1 - r0 + 1
    result["overlap_len"] = overlap
    result["pred_len"] = pred_len
    result["ref_len"] = ref_len
    result["coverage_ref"] = overlap / ref_len
    result["coverage_pred"] = overlap / pred_len
    result["precision"] = result["coverage_pred"]
    result["recall"] = result["coverage_ref"]
    result["f1_score"] = 2 * overlap / (pred_len + ref_len)
    result["iou"] = overlap / (pred_len + ref_len - overlap)
    result["start_abs_error"] = (p0 - r0).abs()
    result["stop_abs_error"] = (p1 - r1).abs()
    result["boundary_distance"] = result["start_abs_error"] + result["stop_abs_error"]
    result["start_exact"] = p0.eq(r0)
    result["stop_exact"] = p1.eq(r1)
    result["exact_match"] = result["start_exact"] & result["stop_exact"]
    if {"pred_frame", "true_frame"}.issubset(result.columns):
        result["frame_correct"] = result["pred_frame"].eq(result["true_frame"])
        #result["strict_orf_correct"] = result["frame_correct"] & result["exact_match"]
    return result


def frame_classification_metrics(selected: pd.DataFrame):
    df = selected.dropna(
        subset=["true_frame", "pred_frame"]
    ).copy()

    truth = df["true_frame"]
    prediction = df["pred_frame"]

    records = []

    for frame in (0, 1, 2):
        tp = int(((truth == frame) & (prediction == frame)).sum())
        fp = int(((truth != frame) & (prediction == frame)).sum())
        fn = int(((truth == frame) & (prediction != frame)).sum())
        tn = int(((truth != frame) & (prediction != frame)).sum())

        records.append({
            "frame": frame,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "accuracy": (tp + tn) / len(df) if len(df) else np.nan,
        })

    table = pd.DataFrame(records)

    n_correct = int((truth == prediction).sum())

    metrics = {
        "n_frame_evaluated": len(df),
        "n_frame_correct": n_correct,
        "frame_accuracy": n_correct / len(df) if len(df) else np.nan,
    }

    return table, metrics


def evaluate_orfs(ref: pd.DataFrame,pred: pd.DataFrame,offset_nt: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Evaluate all candidate ORFs and select one closest
    ORF per gene/fragment.

    selected contains ONLY genes/fragments with predictions.
    """

    if ref["gene_name"].duplicated().any():raise ValueError("Multiple reference CDS per gene")
    unknown = sorted(set(pred["gene_name"]) - set(ref["gene_name"]))

    if unknown:
        raise ValueError(
            f"Predicted genes absent from reference: {unknown[:10]}"
        )

    # --------------------------------------------------
    # 1. Calculate true reading frame
    # --------------------------------------------------

    reference = ref.copy()
    reference["true_frame"] = ((reference["ref_start"] - 1 + offset_nt) % 3).astype(int)
    
    # --------------------------------------------------
    # 2. Score all candidate ORFs
    # --------------------------------------------------

    scored = score_intervals(
        pred.merge(
            reference,
            on="gene_name",
            how="left",
            validate="many_to_one"
        )
    )

    scored["_original_order"] = np.arange(len(scored))

    # --------------------------------------------------
    # 3. Select closest candidate for each fragment
    # --------------------------------------------------

    selected = (
        scored.sort_values(
            [
                "gene_name",
                "boundary_distance",
                "iou",
                "_original_order"
            ],
            ascending=[True, True, False, True],
            kind="stable"
        ).drop_duplicates("gene_name").drop(columns="_original_order").reset_index(drop=True))

    # selected ONLY contains predicted ORFs

    # --------------------------------------------------
    # 4. Detection statistics
    # --------------------------------------------------

    n_reference = len(reference)
    n_detected = len(selected)
    n_undetected = n_reference - n_detected

    detection_rate = (
        n_detected / n_reference
        if n_reference else np.nan
    )

    # --------------------------------------------------
    # 5. ORF evaluation metrics
    # --------------------------------------------------

    metrics = {
        "n_reference": n_reference,
        "n_detected": n_detected,
        "n_undetected": n_undetected,
        "detection_rate": detection_rate,

        #"n_exact_cds": int(selected["exact_match"].sum()),
        #"n_start_exact": int(selected["start_exact"].sum()),
        #"n_stop_exact": int(selected["stop_exact"].sum()),

        "mean_precision_detected": selected["precision"].mean(),
        "mean_recall_detected": selected["recall"].mean(),
        "mean_f1_detected": selected["f1_score"].mean(),
        "mean_iou_detected": selected["iou"].mean(),

        "start_mae_detected": selected["start_abs_error"].mean(),
        "stop_mae_detected": selected["stop_abs_error"].mean(),
    }

    # Exact-match rates among selected predictions
    metrics["start_exact_rate_detected"] = (
        selected["start_exact"].mean()
    )

    metrics["stop_exact_rate_detected"] = (
        selected["stop_exact"].mean()
    )

    metrics["exact_cds_rate_detected"] = (
        selected["exact_match"].mean()
    )

    # --------------------------------------------------
    # 6. Frame classification
    # --------------------------------------------------

    if "pred_frame" in selected.columns:

        frame_table, frame_metrics = (
            frame_classification_metrics(selected)
        )

        metrics.update(frame_metrics)

        #metrics["n_pred_frame_coordinate_mismatches"] = int(
        #    (
        #        (
        #            selected["pred_start"] - 1 + offset_nt
        #        ) % 3 != selected["pred_frame"]
        #    ).sum()
        #)

    # --------------------------------------------------
    # 7. Return results
    # --------------------------------------------------

    return (
        scored.drop(columns="_original_order"),
        selected,
        metrics
    )

def format_report(metrics: dict) -> str:
    lines = ["ORF evaluation (reference-guided closest candidate per gene)",
             "Selection: minimum start + stop absolute errors; tie-break: maximum IoU", ""]
    for k, v in metrics.items():
        lines.append(f"{k:38s} : {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k:38s} : {v}")
    lines += ["", "Notes:",
              "- detection_rate = references with >=1 ORF / all reference records.",
              "- mean_coverage_ref includes undetected references as zero.",
              "- frame_accuracy = correct frame / detected references",
              "- frame precision = correct frame / detected references.",
              "- frame recall = correct frame / all references.",
              "- frame per-class precision/recall/accuracy in frame_metrics.tsv.",
              "- strict_orf_correct requires frame, exact start and exact end.",
              "- Boundary MAE excludes undetected references; nt only at 1-nt stride.",
              "- Region overlap metrics are k-mer-position-level.",
              "- Closest candidate is reference-guided (oracle), not score-based Top-1."]
    return "\n".join(lines) + "\n"