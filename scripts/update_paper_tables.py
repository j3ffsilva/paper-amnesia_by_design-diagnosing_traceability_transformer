"""
Lê outputs/multiseed/multiseed_stats.json e imprime os valores atualizados
para as três tabelas do paper, prontos para copiar/colar no LaTeX.

Uso:
  python scripts/update_paper_tables.py
  python scripts/update_paper_tables.py --format sd   # mean ± sd em vez de CI
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STATS_PATH = Path("outputs/multiseed/multiseed_stats.json")
RAW_PATH   = Path("outputs/multiseed/multiseed_raw.json")

MODELS = ["Static", "Additive", "Joint", "Timeformer"]
MODEL_LABELS = {
    "Static":     "Standard",
    "Additive":   "Additive",
    "Joint":      "Joint",
    "Timeformer": "Mem-Aug.",
}
CLASSES = [("stable", "Stable"), ("drift", "Drift"), ("bifurc", "Bifurcation")]


def fmt_ci(mean: float, lo: float, hi: float, fmt: str = "ci") -> str:
    if np.isnan(mean):
        return "---"
    if fmt == "ci":
        hw = (hi - lo) / 2
        return f"{mean:.3f} $\\pm$ {hw:.3f}"
    else:  # sd from raw
        return f"{mean:.3f}"


def load_sd(raw: list[dict], key_fn) -> dict[str, float]:
    """Computes std dev for a key function applied to each seed's data."""
    result = {}
    for model in MODELS:
        vals = [key_fn(r, model) for r in raw if key_fn(r, model) is not None]
        arr = np.array([v for v in vals if not np.isnan(v)])
        result[model] = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    return result


def print_table1(stats: dict, raw: list[dict], fmt: str) -> None:
    """Table 1: context drift score t0/t9/Δ by (class, model)."""
    print("\n% ─── TABLE 1: Context drift score ───────────────────────────────")
    print("% Substitua os valores em \\begin{tabular}{llccc}")
    print(f"% n_seeds = {stats['n_seeds']}")
    print()

    for cls_key, cls_label in CLASSES:
        for model in MODELS:
            label = MODEL_LABELS[model]
            d = stats["drift"].get(model, {})

            t0_d  = d.get(f"{cls_key}_t0", {})
            t9_d  = d.get(f"{cls_key}_t9", {})
            dlt_d = d.get(f"{cls_key}_delta", {})

            t0_mean  = t0_d.get("mean", float("nan"))
            t0_lo    = t0_d.get("ci95_lo", float("nan"))
            t0_hi    = t0_d.get("ci95_hi", float("nan"))

            t9_mean  = t9_d.get("mean", float("nan"))
            t9_lo    = t9_d.get("ci95_lo", float("nan"))
            t9_hi    = t9_d.get("ci95_hi", float("nan"))

            dl_mean  = dlt_d.get("mean", float("nan"))
            dl_lo    = dlt_d.get("ci95_lo", float("nan"))
            dl_hi    = dlt_d.get("ci95_hi", float("nan"))

            if fmt == "point":
                row = (f"       & {label:<9} & {t0_mean:.3f} & {t9_mean:.3f} "
                       f"& {dl_mean:+.3f} \\\\")
            else:
                hw_t0 = (t0_hi - t0_lo) / 2
                hw_t9 = (t9_hi - t9_lo) / 2
                hw_dl = (dl_hi - dl_lo) / 2
                row = (f"       & {label:<9} "
                       f"& {t0_mean:.3f}{{\\tiny$\\pm${hw_t0:.3f}}} "
                       f"& {t9_mean:.3f}{{\\tiny$\\pm${hw_t9:.3f}}} "
                       f"& {dl_mean:+.3f}{{\\tiny$\\pm${hw_dl:.3f}}} \\\\")
            print(f"{cls_label:<12}{row}")
        print("\\midrule")


def print_table2(stats: dict, fmt: str) -> None:
    """Table 2: main traceability diagnostics."""
    print("\n% ─── TABLE 2: Main traceability diagnostics ─────────────────────")
    print("% Colunas: Drift Δ | Flip | Ambig. probe | Cont. probe")
    print(f"% n_seeds = {stats['n_seeds']}")
    print()

    for model in MODELS:
        label = MODEL_LABELS[model]
        drift = stats["drift"].get(model, {}).get("drift_delta", {})
        flip  = stats["sign_flip"].get(model, {})
        ambig = stats["probe"].get(model, {}).get("ambiguous", {})
        cont  = stats["probe"].get(model, {}).get("continuation", {})

        def f(d: dict) -> str:
            m, lo, hi = d.get("mean", float("nan")), d.get("ci95_lo", float("nan")), d.get("ci95_hi", float("nan"))
            if np.isnan(m):
                return "---"
            if fmt == "point":
                return f"{m:.3f}"
            hw = (hi - lo) / 2
            return f"{m:.3f}{{\\tiny$\\pm${hw:.3f}}}"

        sign = "+" if (drift.get("mean", 0) or 0) >= 0 else ""
        drift_str = f"{sign}{drift.get('mean', float('nan')):.3f}" if fmt == "point" else f(drift)

        print(f"{label:<12} & {drift_str} & {f(flip)} & {f(ambig)} & {f(cont)} \\\\")


def print_table3(stats: dict, fmt: str) -> None:
    """Table 3: memory diagnostics (Joint vs Mem-Aug learned vs oracle)."""
    print("\n% ─── TABLE 3: Memory diagnostics ────────────────────────────────")
    print("% Colunas: Cont. (D4) | Flip (D3) | Δ vs Joint")
    print(f"% n_seeds = {stats['n_seeds']}")
    print()

    joint_cont = stats["probe"].get("Joint", {}).get("continuation", {}).get("mean", float("nan"))
    joint_flip = stats["sign_flip"].get("Joint", {}).get("mean", float("nan"))
    mem_cont   = stats["probe"].get("Timeformer", {}).get("continuation", {}).get("mean", float("nan"))
    mem_flip   = stats["sign_flip"].get("Timeformer", {}).get("mean", float("nan"))

    def f(d: dict) -> str:
        m, lo, hi = d.get("mean", float("nan")), d.get("ci95_lo", float("nan")), d.get("ci95_hi", float("nan"))
        if np.isnan(m):
            return "---"
        if fmt == "point":
            return f"{m:.3f}"
        hw = (hi - lo) / 2
        return f"{m:.3f}{{\\tiny$\\pm${hw:.3f}}}"

    j_cont_d = stats["probe"].get("Joint", {}).get("continuation", {})
    j_flip_d = stats["sign_flip"].get("Joint", {})
    m_cont_d = stats["probe"].get("Timeformer", {}).get("continuation", {})
    m_flip_d = stats["sign_flip"].get("Timeformer", {})

    delta_cont = mem_cont - joint_cont if not (np.isnan(mem_cont) or np.isnan(joint_cont)) else float("nan")
    delta_flip = mem_flip - joint_flip if not (np.isnan(mem_flip) or np.isnan(joint_flip)) else float("nan")

    print(f"Joint (reference) & {f(j_cont_d)} & {f(j_flip_d)} & -- \\\\")
    dc = f"{delta_cont:+.3f}" if not np.isnan(delta_cont) else "---"
    df = f"{delta_flip:+.3f}" if not np.isnan(delta_flip) else "---"
    print(f"Mem-Aug. learned  & {f(m_cont_d)} & {f(m_flip_d)} & {dc} cont.; {df} flip \\\\")
    print("Mem-Aug. oracle   & (re-run needed) & (re-run needed) & -- \\\\")


def print_convergence(raw: list[dict]) -> None:
    """Mostra como a média e o IC 95% do drift Δ convergem com mais seeds."""
    vals = []
    for r in raw:
        v = r.get("drift", {}).get("Joint", {}).get("drift", {}).get("delta")
        if v is not None and not np.isnan(v):
            vals.append(v)

    print("\n% ─── Convergência do IC — drift Δ Joint (drift class) ───────────")
    checkpoints = [5, 10, 15, 20, 25, 30, 50, len(vals)]
    for n in checkpoints:
        if n > len(vals):
            break
        arr = np.array(vals[:n])
        mean = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(n)
        hw = 1.96 * se
        print(f"  n={n:<4}  mean={mean:+.3f}  CI_hw={hw:.4f}  [{mean-hw:+.3f}, {mean+hw:+.3f}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["ci", "sd", "point"], default="ci",
                        help="ci: mean±hw(IC95%), sd: mean±sd, point: só mean")
    args = parser.parse_args()

    if not STATS_PATH.exists():
        print(f"Arquivo não encontrado: {STATS_PATH}")
        print("Rode scripts/run_multiseed.py primeiro.")
        return

    with open(STATS_PATH) as f:
        stats = json.load(f)

    raw = []
    if RAW_PATH.exists():
        with open(RAW_PATH) as f:
            raw = json.load(f)

    print(f"\nN seeds = {stats['n_seeds']}  |  formato = {args.format}")
    print_table1(stats, raw, args.format)
    print_table2(stats, args.format)
    print_table3(stats, args.format)
    if raw:
        print_convergence(raw)


if __name__ == "__main__":
    main()
