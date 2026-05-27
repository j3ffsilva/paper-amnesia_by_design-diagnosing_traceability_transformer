"""
Plot Figure 1: Benchmark Design Schematic (3-panel, matplotlib).

Panels:
  Left   — SVO example sentences for a drifting subject at t0 / t7 / t9,
            showing how planted context shifts and how local markers can be noisy
  Center — Planted P(A|s,t) trajectories for Stable, Drift, and Bifurcating
  Right  — Expected representation-state matrix: subjects × {t0, t4, t9}

Outputs:
  outputs/figures/figure1_benchmark_design.pdf
  outputs/figures/figure1_benchmark_design.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── colour palette ─────────────────────────────────────────────────────────────
C_A      = "#4878CF"   # context A (blue)
C_B      = "#E87E10"   # context B (orange)
C_MIX    = "#9467BD"   # A/B mixed (purple)
C_TRANS  = "#7AB0D4"   # transition ~A→B (light blue)
C_STABLE = "#888888"   # stable class line (gray)

OUT_DIR = Path("outputs/figures")


# ── Panel 1: SVO sentences ─────────────────────────────────────────────────────

def draw_svo_panel(ax: plt.Axes) -> None:
    """Three example SVO rows for a drifting subject, coloured by context."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Corpus sentences", fontsize=9, fontweight="bold", pad=6)

    # (epoch, subj, verb, obj, true_ctx, verb_ctx, obj_ctx)
    rows = [
        ("t0", "S5", "V2", "O3", "N1", "N1", "N1"),
        ("t7", "S5", "V2", "O7", "N2", "N1", "N2"),   # V2 is noisy: N1-verb in N2 sentence
        ("t9", "S5", "V6", "O8", "N2", "N2", "N2"),
    ]

    col_x = [0.05, 0.23, 0.41, 0.59, 0.79]   # epoch / subj / verb / obj / label
    col_w = [0.16, 0.16, 0.16, 0.16, 0.17]
    row_y = [0.72, 0.44, 0.16]
    rh    = 0.21

    # column headers
    for ci, hdr in enumerate(["Epoch", "Subj", "Verb", "Obj", "Context"]):
        ax.text(col_x[ci] + col_w[ci] / 2, 0.93, hdr,
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="#444444")

    ctx_color = {"N1": C_A, "N2": C_B}

    for ri, (ep, subj, verb, obj, true_ctx, verb_ctx, obj_ctx) in enumerate(rows):
        y = row_y[ri]
        noisy_verb = verb_ctx != true_ctx
        noisy_obj  = obj_ctx  != true_ctx

        cell_bg = [
            "#f2f2f2",                         # epoch: neutral
            "#f2f2f2",                         # subj:  neutral
            ctx_color[verb_ctx],               # verb: colour by its actual context
            ctx_color[obj_ctx],                # obj:  colour by its actual context
            ctx_color[true_ctx],               # label: planted true context
        ]
        tokens = [ep, subj, verb, obj, true_ctx]

        for ci, (tok, bg) in enumerate(zip(tokens, cell_bg)):
            rect = mpatches.FancyBboxPatch(
                (col_x[ci] + 0.005, y - rh / 2 + 0.01),
                col_w[ci] - 0.01, rh - 0.02,
                boxstyle="round,pad=0.008",
                facecolor=bg,
                edgecolor="#cccccc" if ci < 4 else "none",
                linewidth=0.5,
                alpha=0.65 if ci < 2 else (0.72 if ci < 4 else 0.82),
                zorder=2,
            )
            ax.add_patch(rect)

            text_color = "white" if ci == 4 else "#1a1a1a"
            ax.text(col_x[ci] + col_w[ci] / 2, y, tok,
                    ha="center", va="center",
                    fontsize=8, color=text_color,
                    fontfamily="monospace", zorder=3)

            # asterisk on noisy markers
            if (ci == 2 and noisy_verb) or (ci == 3 and noisy_obj):
                ax.text(col_x[ci] + col_w[ci] - 0.005, y + rh / 2 - 0.02,
                        "∗", ha="right", va="top",
                        fontsize=7.5, color="#cc3300", zorder=4)

    ax.text(0.02, 0.01, "∗ noisy marker (opposite context)",
            fontsize=6.5, color="#cc3300", va="bottom")


# ── Panel 2: planted P(A|s,t) trajectories ────────────────────────────────────

def _stable(t):    return np.full_like(t, 0.80, dtype=float)
def _drift(t):     return np.clip(1.0 - t / 9.0, 0.0, 1.0)
def _bifurc(t):    return 0.50 + 0.50 * np.exp(-0.55 * t)


def draw_trajectory_panel(ax: plt.Axes) -> None:
    t = np.arange(10)

    ax.plot(t, _stable(t),  color=C_STABLE, linewidth=1.8, linestyle="--", label="Stable",       zorder=3)
    ax.plot(t, _drift(t),   color=C_A,      linewidth=1.8, linestyle="-",  label="Drift",        zorder=3)
    ax.plot(t, _bifurc(t),  color=C_B,      linewidth=1.8, linestyle="-",  label="Bifurcating",  zorder=3)

    ax.axhline(0.5, color="#bbbbbb", linewidth=0.8, linestyle=":", zorder=1)

    ax.set_xlim(-0.3, 9.3)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(range(0, 10, 2))
    ax.set_xticklabels([f"t{i}" for i in range(0, 10, 2)], fontsize=8)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"], fontsize=7.5)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel(r"$P(N_1 \mid s,\,t)$", fontsize=8)
    ax.set_title("Planted trajectories", fontsize=9, fontweight="bold", pad=6)
    ax.legend(fontsize=7.5, loc="center right",
              framealpha=0.88, edgecolor="#cccccc", handlelength=2.0)
    ax.tick_params(axis="both", length=3)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


# ── Panel 3: expected representation-state matrix ─────────────────────────────

def draw_matrix_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Expected neighborhood", fontsize=9, fontweight="bold", pad=6)

    col_labels = ["t0", "t4", "t9"]
    row_labels  = ["Stable", "Drift", "Bifurc."]

    # (display text, background colour)
    cells: list[list[tuple[str, str]]] = [
        [("N1",      C_A),   ("N1",      C_A),    ("N1",    C_A)],
        [("N1",      C_A),   ("~N1→N2",  C_TRANS), ("N2",   C_B)],
        [("N1",      C_A),   ("~N1",     C_TRANS), ("N1/N2", C_MIX)],
    ]

    col_x = [0.38, 0.60, 0.82]
    row_y = [0.72, 0.47, 0.22]
    cw, rh = 0.19, 0.19

    # column headers
    for ci, cl in enumerate(col_labels):
        ax.text(col_x[ci], 0.90, cl,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#444444")

    # row labels
    for ri, rl in enumerate(row_labels):
        ax.text(0.16, row_y[ri], rl,
                ha="center", va="center", fontsize=8, color="#222222")

    for ri, row in enumerate(cells):
        for ci, (label, bg) in enumerate(row):
            rect = mpatches.FancyBboxPatch(
                (col_x[ci] - cw / 2, row_y[ri] - rh / 2),
                cw, rh,
                boxstyle="round,pad=0.008",
                facecolor=bg, edgecolor="white",
                linewidth=1.2, alpha=0.82, zorder=2,
            )
            ax.add_patch(rect)
            ax.text(col_x[ci], row_y[ri], label,
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", zorder=3)

    legend_handles = [
        mpatches.Patch(facecolor=C_A,    alpha=0.82, label="N1"),
        mpatches.Patch(facecolor=C_B,    alpha=0.82, label="N2"),
        mpatches.Patch(facecolor=C_MIX,  alpha=0.82, label="N1/N2 mixed"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower center",
              ncol=3, framealpha=0.75, edgecolor="#cccccc",
              bbox_to_anchor=(0.58, -0.01))


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.size":        9,
        "axes.linewidth":   0.7,
        "figure.facecolor": "white",
        "savefig.dpi":      300,
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1, 3, figsize=(12, 3.4),
        gridspec_kw={"width_ratios": [1.05, 1.0, 0.85]},
    )

    draw_svo_panel(axes[0])
    draw_trajectory_panel(axes[1])
    draw_matrix_panel(axes[2])

    fig.tight_layout(pad=1.4)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"figure1_benchmark_design.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
