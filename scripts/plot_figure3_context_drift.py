"""
Create Figure 3 for the IBERAMIA paper planning:
class-specific context drift score across epochs.

Inputs are the neighbor_analysis.json files generated for the corrected
multi-seed runs. The script writes:
  - outputs/figures/figure3_context_drift.html  (self-contained Plotly HTML)
  - outputs/figures/figure3_context_drift.svg   (optional, with --write-kaleido)
  - outputs/figures/figure3_context_drift.png   (optional, with --write-kaleido)
  - outputs/figures/figure3_context_drift_data.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px

from src.timeformer.nomenclature import model_label


DEFAULT_MODELS = ("Static", "Additive", "Joint", "Timeformer")
CLASS_LABELS = {
    "stable": "Stable",
    "drift": "Drift",
    "bifurc": "Bifurcation",
}
# Short labels for the plot: drop the word "Transformer"/"Timeformer"
PLOT_LABELS = {
    "Static":     "Standard",
    "Additive":   "Additive",
    "Joint":      "Token-Time",
    "Timeformer": "Memory-Augmented",
}
MODEL_ORDER = [PLOT_LABELS[m] for m in DEFAULT_MODELS]
CLASS_ORDER = ["Stable", "Drift", "Bifurcation"]


def default_runs() -> list[str]:
    """Return run IDs from multiseed_raw.json, falling back to the old 3-seed default."""
    raw_path = Path("outputs/multiseed/multiseed_raw.json")
    if raw_path.exists():
        data = json.loads(raw_path.read_text())
        return [r["run_id"] for r in data]
    return ["20260523_006", "20260523_007", "20260523_008"]


def load_rows(run_ids: list[str], models: list[str]) -> list[dict]:
    rows: list[dict] = []
    for run_id in run_ids:
        path = Path(f"outputs/runs/{run_id}/results/neighbor_analysis.json")
        data = json.loads(path.read_text())
        by_class = data["drift_score_by_class"]
        for model in models:
            for class_id, epoch_scores in by_class[model].items():
                for epoch_str, score in epoch_scores.items():
                    rows.append({
                        "run_id": run_id,
                        "model": model,
                        "model_label": PLOT_LABELS.get(model, model),
                        "class": class_id,
                        "class_label": CLASS_LABELS[class_id],
                        "epoch": int(epoch_str),
                        "context_a_neighbor_share": float(score),
                    })
    return rows


def aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["model", "model_label", "class", "class_label", "epoch"], as_index=False)
        .agg(
            mean=("context_a_neighbor_share", "mean"),
            std=("context_a_neighbor_share", "std"),
        )
    )
    agg["lower"] = (agg["mean"] - agg["std"].fillna(0.0)).clip(lower=0.0)
    agg["upper"] = (agg["mean"] + agg["std"].fillna(0.0)).clip(upper=1.0)
    return agg


def make_figure(agg: pd.DataFrame):
    fig = px.line(
        agg,
        x="epoch",
        y="mean",
        color="model_label",
        facet_col="class_label",
        facet_col_wrap=3,
        category_orders={
            "model_label": MODEL_ORDER,
            "class_label": CLASS_ORDER,
        },
        markers=True,
        labels={
            "epoch": "Epoch",
            "mean": "Proportion of N₁ neighbors",
            "model_label": "Model",
            "class_label": "Subject class",
        },
    )

    fig.update_traces(line={"width": 2.6}, marker={"size": 8})
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Subject class=", "")))
    fig.update_layout(
        template="plotly_white",
        width=1200,
        height=500,
        font={"family": "Arial", "size": 16},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.28,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 15},
        },
        margin={"l": 65, "r": 25, "t": 30, "b": 100},
    )
    fig.update_yaxes(range=[0, 1], dtick=0.2)
    fig.update_xaxes(dtick=1)

    # Add a light variability band (mean +/- 1 std) behind each line.
    colors = {
        trace.name: trace.line.color
        for trace in fig.data
        if getattr(trace, "mode", "") and "lines" in trace.mode
    }
    facet_axis_by_class = {
        CLASS_ORDER[0]: ("x", "y"),
        CLASS_ORDER[1]: ("x2", "y2"),
        CLASS_ORDER[2]: ("x3", "y3"),
    }
    for class_label in CLASS_ORDER:
        xaxis, yaxis = facet_axis_by_class[class_label]
        class_df = agg[agg["class_label"] == class_label]
        for model in MODEL_ORDER:
            model_df = class_df[class_df["model_label"] == model].sort_values("epoch")
            if model_df.empty:
                continue
            rgba = colors.get(model, "#999999")
            band_color = rgba.replace("rgb(", "rgba(").replace(")", ",0.12)")
            fig.add_scatter(
                x=list(model_df["epoch"]) + list(model_df["epoch"])[::-1],
                y=list(model_df["upper"]) + list(model_df["lower"])[::-1],
                fill="toself",
                fillcolor=band_color,
                line={"color": "rgba(255,255,255,0)"},
                hoverinfo="skip",
                showlegend=False,
                xaxis=xaxis,
                yaxis=yaxis,
            )

    # Keep bands visually behind the line traces.
    line_traces = [t for t in fig.data if getattr(t, "mode", "") and "lines" in t.mode]
    band_traces = [t for t in fig.data if not (getattr(t, "mode", "") and "lines" in t.mode)]
    fig.data = tuple(band_traces + line_traces)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure 3 context drift score")
    parser.add_argument("--runs", nargs="+", default=default_runs())
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument(
        "--write-kaleido",
        action="store_true",
        help="Also try Plotly/Kaleido static export. Disabled by default because "
             "Kaleido is unstable on some macOS setups.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.runs, args.models)
    agg = aggregate(rows)
    agg.to_csv(out_dir / "figure3_context_drift_data.csv", index=False)

    fig = make_figure(agg)
    html_path = out_dir / "figure3_context_drift.html"
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    print(f"Wrote {html_path}")

    if args.write_kaleido:
        for ext in ("svg", "png"):
            path = out_dir / f"figure3_context_drift.{ext}"
            try:
                fig.write_image(path)
                print(f"Wrote {path}")
            except Exception as exc:
                print(f"Skipped {path}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
