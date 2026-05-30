"""Clinical report generator for AI chest X-ray analysis.

Produces a structured plain-text radiology report from model predictions
and Monte Carlo Dropout uncertainty estimates. Reports are saved as .txt
files and include a mandatory AI disclaimer.

NOT INTENDED FOR CLINICAL USE — requires radiologist validation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.uncertainty import MCDropoutPredictor, UncertaintyResult


# Probability threshold above which a label is listed as a finding
FINDING_THRESHOLD: float = 0.3


def generate_clinical_report(
    image_path: str,
    predictions: dict[str, float],
    uncertainty: UncertaintyResult,
    output_path: str,
) -> str:
    """Generate a structured AI clinical report and save it to disk.

    The report contains:
    - Header identifying it as an AI analysis
    - Date and time of generation
    - Input image path
    - Findings: all labels with predicted probability > 0.3, listed with
      confidence percentage and uncertainty tier (low / medium / high)
    - Impression: summary finding count and radiologist-review flag
    - Mandatory AI disclaimer

    Args:
        image_path: Path to the source image file (used for provenance).
        predictions: Mapping of label name → sigmoid probability.
        uncertainty: UncertaintyResult from MCDropoutPredictor.
        output_path: Destination path for the .txt report file.

    Returns:
        The full report as a string (also saved to output_path).
    """
    label_names = list(predictions.keys())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []

    # --- Header ---
    lines.append("=" * 60)
    lines.append("       CHEST X-RAY AI ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"Date/Time : {now}")
    lines.append(f"Image     : {image_path}")
    lines.append("")

    # --- Findings ---
    lines.append("FINDINGS")
    lines.append("-" * 40)

    findings: list[tuple[str, float, str]] = []
    for i, (label, prob) in enumerate(predictions.items()):
        if prob > FINDING_THRESHOLD:
            std_val = float(uncertainty.std[i]) if i < len(uncertainty.std) else 0.0
            tier = MCDropoutPredictor.classify_uncertainty(std_val)
            findings.append((label, prob, tier))

    if findings:
        for label, prob, tier in findings:
            lines.append(f"  - {label} - {prob * 100:.1f}% confidence ({tier} uncertainty)")
    else:
        lines.append("  No significant findings above threshold.")
    lines.append("")

    # --- Uncertainty summary ---
    lines.append("UNCERTAINTY SUMMARY")
    lines.append("-" * 40)
    for i, (label, prob) in enumerate(predictions.items()):
        std_val = float(uncertainty.std[i]) if i < len(uncertainty.std) else 0.0
        ci_lo = float(uncertainty.ci_lower[i]) if i < len(uncertainty.ci_lower) else 0.0
        ci_hi = float(uncertainty.ci_upper[i]) if i < len(uncertainty.ci_upper) else 0.0
        tier = MCDropoutPredictor.classify_uncertainty(std_val)
        lines.append(
            f"  {label}: std={std_val:.4f} "
            f"95% CI=[{ci_lo:.3f}, {ci_hi:.3f}] "
            f"[{tier}]"
        )
    lines.append("")

    # --- Impression ---
    high_uncertainty_count = sum(
        1 for i in range(len(label_names))
        if i < len(uncertainty.std)
        and MCDropoutPredictor.classify_uncertainty(float(uncertainty.std[i])) == "high"
    )

    lines.append("IMPRESSION")
    lines.append("-" * 40)
    lines.append(
        f"  {len(findings)} finding(s) identified with probability > "
        f"{int(FINDING_THRESHOLD * 100)}%."
    )
    if high_uncertainty_count > 0:
        lines.append(
            f"  {high_uncertainty_count} finding(s) with high uncertainty "
            "— radiologist review recommended."
        )
    else:
        lines.append("  All findings have low-to-medium uncertainty.")
    lines.append("")

    # --- Disclaimer ---
    lines.append("DISCLAIMER")
    lines.append("-" * 40)
    lines.append(
        "  AI-generated report. Not for clinical use. "
        "Requires radiologist validation."
    )
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    # Save to disk
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report_text, encoding="utf-8")

    return report_text
