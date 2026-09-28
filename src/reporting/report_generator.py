"""HTML and PDF experiment reports.

The narrative is filled from measured results. It does not claim that synthetic
data is anonymous, perfect, or differentially private.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.evaluation.privacy import DISCLAIMER
from src.utils.metrics import to_jsonable

SECTION_LIMITATION = (
    "Synthetic records can still leak information through memorization, rare combinations, "
    "or attribute inference. The diagnostics in this report are finite-sample comparisons. "
    "They are not a formal privacy guarantee, and dimensionality-reduction plots are qualitative only. "
    "Downstream models are lightly configured probes, not tuned production models."
)


def build_report(
    experiment: dict[str, Any],
    path_html: Path,
    path_pdf: Path,
    *,
    real: Any = None,
    synthetic: Any = None,
) -> dict[str, str]:
    """Write an HTML report and a PDF report for one experiment."""
    images = _figures(path_html.parent, experiment, real, synthetic)
    document = _html(experiment, images)
    path_html.write_text(document, encoding="utf-8")
    _pdf(path_pdf, experiment, images)
    return {"html": str(path_html), "pdf": str(path_pdf)}


def _html(experiment: dict[str, Any], images: dict[str, str]) -> str:
    exp_id = html.escape(str(experiment.get("experiment_id", "unassigned")))
    sections = [
        _section("1. Executive summary", _summary_text(experiment)),
        _section("2. Dataset description", _dataset_text(experiment)),
        _section("3. Data quality", _quality_source(experiment)),
        _section("4. Preprocessing", _pre_text(experiment)),
        _section("5. Model architecture", _model_text(experiment)),
        _section("6. Training configuration", _train_text(experiment)),
        _section("7. Synthetic data generation", _gen_text(experiment)),
        _section("8. Statistical evaluation", _stat_text(experiment)),
        _section("9. Privacy diagnostics", _privacy_text(experiment)),
        _section("10. ML utility", _utility_text(experiment)),
        _section("11. Baseline comparison", _baseline_text(experiment)),
        _section("12. Limitations", f"<p>{html.escape(SECTION_LIMITATION)}</p><p>{html.escape(DISCLAIMER)}</p>"),
        _section("13. Conclusion", _conclusion_text(experiment)),
    ]
    gallery = "".join(
        f"<figure><img src='data:image/png;base64,{data}' alt='{html.escape(name)}'/><figcaption>{html.escape(name)}</figcaption></figure>"
        for name, data in images.items()
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>TabDiffuse report {exp_id}</title>
<style>
body {{ font-family: "Segoe UI", Helvetica, Arial, sans-serif; margin: 0; background: #f3f5f7; color: #1b2430; }}
main {{ max-width: 920px; margin: 32px auto; background: white; padding: 40px 48px; }}
h1 {{ margin-bottom: 0; font-weight: 650; }}
h2 {{ font-size: 1.15rem; margin-top: 2rem; border-bottom: 1px solid #e2e8ee; padding-bottom: 0.3rem; }}
.meta {{ color: #5c6b7a; }}
.disclaimer {{ background: #f7f1e6; border-left: 4px solid #c9842a; padding: 12px 14px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
th, td {{ border-bottom: 1px solid #e2e8ee; text-align: left; padding: 6px 8px; vertical-align: top; }}
figure {{ margin: 18px 0; }}
img {{ max-width: 100%; }}
</style></head><body><main>
<p class="meta">TabDiffuse · Score-based diffusion for privacy-aware synthetic tabular data</p>
<h1>Experiment report {exp_id}</h1>
<p class="meta">Generated {html.escape(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))}</p>
<div class="disclaimer">{html.escape(DISCLAIMER)}</div>
{''.join(sections)}
<h2>Figures</h2>
{gallery or '<p>No figures were available for this run.</p>'}
</main></body></html>"""


def _section(title: str, body: str) -> str:
    return f"<h2>{html.escape(title)}</h2>{body}"


def _p(text: str) -> str:
    return f"<p>{html.escape(text)}</p>"


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return _p("No table is available for this section.")
    headers = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(_fmt(row.get(header)))}</td>" for header in headers)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _summary_text(experiment: dict[str, Any]) -> str:
    quality = (experiment.get("evaluation") or {}).get("similarity") or {}
    privacy = experiment.get("privacy") or {}
    return _p(
        f"Experiment {experiment.get('experiment_id')} trained a variance-preserving score-based diffusion model "
        f"on {experiment.get('dataset_name', 'an uploaded table')} "
        f"({experiment.get('rows', '—')} rows, {experiment.get('columns', '—')} columns). "
        f"Distribution similarity was {_fmt(quality.get('distribution_similarity'))}, "
        f"correlation similarity was {_fmt(quality.get('correlation_similarity'))}, and the privacy diagnostic label was "
        f"{privacy.get('overall_label') or 'not computed'}."
    )


def _dataset_text(experiment: dict[str, Any]) -> str:
    return _p(
        f"Dataset name: {experiment.get('dataset_name', 'uploaded table')}. "
        f"Rows: {experiment.get('rows')}. Columns: {experiment.get('columns')}. "
        f"Random seed: {experiment.get('seed')}. Timestamp: {experiment.get('created_at')}."
    ) + _p("Python and package versions are stored with the experiment so the run can be repeated.") + _table(
        [{"package": name, "version": version} for name, version in (experiment.get("packages") or {}).items()]
    )


def _quality_source(experiment: dict[str, Any]) -> str:
    warnings = (experiment.get("profile_warnings") or [])[:12]
    if not warnings:
        return _p("No data-quality warnings were stored for this experiment.")
    return _table([{"warning": warning} for warning in warnings])


def _pre_text(experiment: dict[str, Any]) -> str:
    actions = ((experiment.get("preprocessing") or {}).get("actions") or [])[:40]
    return _p("The preprocessor was fit on the training split only. Exclusions and transforms are listed below.") + _table(actions)


def _model_text(experiment: dict[str, Any]) -> str:
    config = experiment.get("model_config") or {}
    return _p(
        "The model estimates the score of a variance-preserving SDE. Numerical columns are scaled. "
        "Categorical columns are one-hot encoded, or embedded and decoded by nearest prototype when cardinality is high. "
        "Category codes are not treated as numeric magnitudes."
    ) + _table([{"parameter": key, "value": _fmt(config.get(key))} for key in (
        "sde", "schedule", "steps", "hidden_dim", "layers", "dropout", "sampler", "time_sampling", "condition_column"
    )])


def _train_text(experiment: dict[str, Any]) -> str:
    training = experiment.get("training") or {}
    return _p(
        f"Best validation loss: {_fmt(training.get('best_val_loss'))}. "
        f"Duration: {_fmt(training.get('duration_seconds'))} seconds. "
        f"Epochs completed: {training.get('epoch')}."
    )


def _gen_text(experiment: dict[str, Any]) -> str:
    generation = experiment.get("generation") or {}
    if not generation:
        return _p("Synthetic data had not been generated when this report was written.")
    return _p(
        f"Requested rows: {generation.get('requested')}. Restored rows: {generation.get('rows')}. "
        f"Sampler: {generation.get('sampler')}. Steps: {generation.get('steps')}. "
        f"Temperature: {generation.get('temperature')}. Seed: {generation.get('seed')}. "
        f"Duration: {_fmt(generation.get('duration_seconds'))} seconds."
    ) + _p(str(generation.get("conditioning_method") or "Unconditional sampling."))


def _stat_text(experiment: dict[str, Any]) -> str:
    evaluation = experiment.get("evaluation") or {}
    numerical = ((evaluation.get("numerical") or {}).get("summary")) or {}
    categorical = ((evaluation.get("categorical") or {}).get("summary")) or {}
    correlation = evaluation.get("correlation") or {}
    rows = [
        {"metric": "Mean KS statistic", "value": _fmt(numerical.get("mean_ks"))},
        {"metric": "Mean Wasserstein distance", "value": _fmt(numerical.get("mean_wasserstein"))},
        {"metric": "Mean PSI", "value": _fmt(numerical.get("mean_psi"))},
        {"metric": "Pearson mean absolute difference", "value": _fmt(correlation.get("pearson_mean_abs_difference"))},
        {"metric": "Spearman mean absolute difference", "value": _fmt(correlation.get("spearman_mean_abs_difference"))},
        {"metric": "Mean categorical total variation", "value": _fmt(categorical.get("mean_total_variation"))},
    ]
    return _table(rows)


def _privacy_text(experiment: dict[str, Any]) -> str:
    privacy = experiment.get("privacy") or {}
    if not privacy:
        return _p("Privacy diagnostics were not run.")
    exact = privacy.get("exact_match") or {}
    neighbor = privacy.get("nearest_neighbor") or {}
    membership = privacy.get("membership") or {}
    return _p(DISCLAIMER) + _table(
        [
            {"diagnostic": "Exact match rate", "value": _fmt(exact.get("exact_match_rate")), "label": exact.get("label")},
            {"diagnostic": "Distance ratio", "value": _fmt(neighbor.get("distance_ratio")), "label": neighbor.get("label")},
            {
                "diagnostic": "Membership advantage",
                "value": _fmt(membership.get("advantage_over_half")),
                "label": membership.get("label"),
            },
            {"diagnostic": "Overall observed label", "value": privacy.get("overall_label"), "label": ""},
        ]
    )


def _utility_text(experiment: dict[str, Any]) -> str:
    utility = experiment.get("utility") or {}
    rows = utility.get("rows") or []
    if not rows:
        return _p("ML utility was not evaluated.")
    compact = []
    for row in rows:
        compact.append(
            {
                "model": row.get("model"),
                "training_data": row.get("training_data"),
                "accuracy_or_r2": row.get("accuracy") if row.get("accuracy") is not None else row.get("r2"),
                "f1_or_rmse": row.get("f1") if row.get("f1") is not None else row.get("rmse"),
                "roc_auc_or_mae": row.get("roc_auc") if row.get("roc_auc") is not None else row.get("mae"),
            }
        )
    return _p("Scores below are measured on held-out real rows. They are not an automatic ranking.") + _table(compact)


def _baseline_text(experiment: dict[str, Any]) -> str:
    rows = experiment.get("baselines") or []
    if not rows:
        return _p("Baselines were not compared for this experiment.")
    return _p("Each baseline was measured with the same quality and privacy summaries where available.") + _table(rows)


def _conclusion_text(experiment: dict[str, Any]) -> str:
    notes = experiment.get("notes") or {}
    user = notes.get("conclusion") or "No written conclusion was saved with this experiment."
    return _p(user) + _p(
        "Interpret the tables above as empirical evidence about this dataset, seed, and configuration. "
        "A different split or a longer training run can change the measurements."
    )


def _figures(folder: Path, experiment: dict[str, Any], real, synthetic) -> dict[str, str]:
    import base64

    folder.mkdir(parents=True, exist_ok=True)
    images: dict[str, str] = {}
    history = ((experiment.get("training") or {}).get("history")) or {}
    if history.get("train_loss"):
        images["Training loss"] = _encode_plot(folder / "loss.png", history["train_loss"], history.get("val_loss") or [])
    correlation = (experiment.get("evaluation") or {}).get("correlation") or {}
    if correlation.get("available") and correlation.get("pearson_difference"):
        path = folder / "corr_diff.png"
        matrix = correlation["pearson_difference"]
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("Pearson correlation difference")
        fig.colorbar(image, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        images["Correlation difference"] = base64.b64encode(path.read_bytes()).decode("ascii")
    if real is not None and synthetic is not None:
        numeric = [column for column in real.columns if column in synthetic.columns and _is_numeric(real[column])]
        for column in numeric[:2]:
            path = folder / f"dist_{column}.png"
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(real[column].dropna(), bins=25, alpha=0.65, label="Real", density=True)
            ax.hist(synthetic[column].dropna(), bins=25, alpha=0.65, label="Synthetic", density=True)
            ax.legend()
            ax.set_title(str(column))
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
            import base64 as b64
            images[f"Distribution: {column}"] = b64.b64encode(path.read_bytes()).decode("ascii")
    return images


def _encode_plot(path: Path, train: list[float], val: list[float]) -> str:
    import base64

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(train, label="Training")
    if val:
        ax.plot(val, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _pdf(path: Path, experiment: dict[str, Any], images: dict[str, str]) -> None:
    import base64

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12))
    story = [
        Paragraph(f"TabDiffuse report {experiment.get('experiment_id', '')}", styles["Title"]),
        Paragraph(DISCLAIMER, styles["Small"]),
        Spacer(1, 0.4 * cm),
        Paragraph(_summary_text(experiment).replace("<p>", "").replace("</p>", ""), styles["BodyText"]),
        Spacer(1, 0.3 * cm),
        Paragraph("Limitations", styles["Heading2"]),
        Paragraph(SECTION_LIMITATION, styles["Small"]),
    ]
    for name, payload in list(images.items())[:4]:
        buffer = BytesIO(base64.b64decode(payload))
        story.extend([Spacer(1, 0.3 * cm), Paragraph(name, styles["Heading3"]), Image(buffer, width=14 * cm, height=8 * cm)])
    privacy = experiment.get("privacy") or {}
    exact = (privacy.get("exact_match") or {}).get("exact_match_rate")
    data = [
        ["Item", "Value"],
        ["Experiment", str(experiment.get("experiment_id"))],
        ["Best validation loss", _fmt((experiment.get("training") or {}).get("best_val_loss"))],
        ["Exact match rate", _fmt(exact)],
        ["Privacy label", str(privacy.get("overall_label") or "not computed")],
    ]
    table = Table(data, colWidths=[7 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e7c74")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d5dde5")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([Spacer(1, 0.4 * cm), table])
    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=A4, title=f"TabDiffuse {experiment.get('experiment_id')}").build(story)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _is_numeric(series) -> bool:
    try:
        import pandas as pd
        return pd.api.types.is_numeric_dtype(series)
    except Exception:
        return False


def serializable_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    return to_jsonable(experiment)
