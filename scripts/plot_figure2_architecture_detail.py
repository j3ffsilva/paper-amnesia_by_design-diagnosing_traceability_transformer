"""
Create Figure 2 for the IBERAMIA paper:
Timeformer ablation chain — 4-column layout showing Static → Additive → Joint → Timeformer.

Each column is one variant; new components added at each step are highlighted.
Columns share the same vertical alignment so differences are immediately visible.

The script writes:
  - outputs/figures/figure2_architecture_detail.html
  - outputs/figures/figure2_architecture_detail.json

The PNG is rendered from the self-contained HTML with Chrome headless.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

OUT_DIR = Path("outputs/figures")
HTML_PATH = OUT_DIR / "figure2_architecture_detail.html"
JSON_PATH = OUT_DIR / "figure2_architecture_detail.json"

# ── Palette ──────────────────────────────────────────────────────────────────
C = {
    "ink":         "#1c3557",
    "muted":       "#8899aa",
    "grid":        "#dce6f2",
    "panel":       "#f7f9fc",
    "token":       "#636EFA",
    "time":        "#EF553B",
    "interaction": "#00CC96",
    "encoder":     "#2f80ed",
    "memory":      "#AB63FA",
    "output":      "#e05252",
    "new_fill":    "#fffbe6",
    "base_fill":   "#ffffff",
    "absent_line": "#cccccc",
    "absent_text": "#cccccc",
}

# ── Layout constants ──────────────────────────────────────────────────────────
# 4 columns fit symmetrically in [0.01, 0.99].
# COL_W is the half-width of each box (box spans cx ± COL_W).
# Spacing between column centers is 0.25; inter-column gap = 0.25 - 2*COL_W = 0.03.

COL_CENTERS = [0.125, 0.375, 0.625, 0.875]
COL_W  = 0.11    # half-width of each box
BOX_H  = 0.055   # half-height of standard boxes

# Row centers (y) — shared by all four columns.
# Static-Joint: token → [time] → [interact] → encoder → h_s   (h_s at ROWS["memory"])
# Timeformer:   token → time → interact → encoder              (memory+attn at ROWS["memory"])
#                                                               (h_s at HS_Y_TF, below)
ROWS = {
    "token":    0.87,
    "time":     0.73,
    "interact": 0.59,
    "encoder":  0.45,
    "memory":   0.27,   # Static/Additive/Joint output (h_s); Timeformer memory+attn level
}

# Timeformer-specific vertical positions (memory module + h_s below it)
MEM_BOX_Y0  = 0.215   # bottom of Prototype Memory / Temporal Attention boxes
MEM_BOX_Y1  = 0.325   # top    of Prototype Memory / Temporal Attention boxes
HALF_W_TF   = 0.042   # half-width of each Timeformer sub-box
MEM_OFFSET  = COL_W * 0.45   # horizontal offset of sub-boxes from column center
HS_Y_TF     = 0.11    # center of the h_s output box for Timeformer

VARIANT_LABELS = [
    "Standard<br>Transformer",
    "Additive<br>Time-Conditioned",
    "Joint<br>Token-Time",
    "Timeformer<br>Memory-Augmented",
]


def box(fig: go.Figure,
        cx: float, cy: float,
        title: str, body: str,
        line_color: str,
        fill: str = C["base_fill"],
        absent: bool = False) -> None:
    lc = C["absent_line"] if absent else line_color
    fc = "#f8f8f8"     if absent else fill
    fig.add_shape(type="rect",
                  x0=cx - COL_W, y0=cy - BOX_H,
                  x1=cx + COL_W, y1=cy + BOX_H,
                  line={"color": lc, "width": 1.5 if absent else 2,
                        "dash": "dot" if absent else "solid"},
                  fillcolor=fc, layer="below")
    tc = C["absent_text"] if absent else line_color
    fig.add_annotation(x=cx, y=cy + BOX_H - 0.020,
                       text=f"<b>{title}</b>",
                       showarrow=False,
                       font={"size": 10, "color": tc}, align="center")
    bc = C["absent_text"] if absent else C["ink"]
    fig.add_annotation(x=cx, y=cy - 0.015,
                       text=body,
                       showarrow=False,
                       font={"size": 9, "color": bc}, align="center")


def arrow(fig: go.Figure,
          x0: float, y0: float,
          x1: float, y1: float,
          color: str = C["muted"]) -> None:
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
                       xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3,
                       arrowsize=1.0, arrowwidth=1.8,
                       arrowcolor=color)


def build_figure() -> go.Figure:
    fig = go.Figure()

    # Background panel
    fig.add_shape(type="rect", x0=0.01, y0=0.03, x1=0.99, y1=0.97,
                  line={"color": C["grid"], "width": 1},
                  fillcolor=C["panel"], layer="below")

    # Column dividers
    for i in range(1, 4):
        xd = (COL_CENTERS[i - 1] + COL_CENTERS[i]) / 2
        fig.add_shape(type="line",
                      x0=xd, y0=0.06, x1=xd, y1=0.94,
                      line={"color": C["grid"], "width": 1, "dash": "dot"})

    # ── Column headers ────────────────────────────────────────────────────────
    header_colors = [C["muted"], C["time"], C["interaction"], C["memory"]]
    for cx, label, hc in zip(COL_CENTERS, VARIANT_LABELS, header_colors):
        fig.add_annotation(x=cx, y=0.965,
                           text=f"<b>{label}</b>",
                           showarrow=False,
                           font={"size": 11, "color": hc}, align="center")

    # ── Row: Token input (all variants) ──────────────────────────────────────
    for cx in COL_CENTERS:
        box(fig, cx, ROWS["token"],
            "Token input", "S V O + position", C["token"])

    # ── Row: Time input (Additive, Joint, Timeformer); absent in Static ─────────
    box(fig, COL_CENTERS[0], ROWS["time"],
        "Time input", "— absent —", C["time"], absent=True)
    for cx in COL_CENTERS[1:]:
        fill = C["new_fill"] if cx == COL_CENTERS[1] else C["base_fill"]
        box(fig, cx, ROWS["time"],
            "Time input", "τ(t)  continuous", C["time"], fill=fill)

    # ── Row: Token-Time Interaction (Joint, Timeformer); absent in Static and Additive ──
    for cx in COL_CENTERS[:2]:
        box(fig, cx, ROWS["interact"],
            "Token-Time<br>Interaction", "— absent —", C["interaction"],
            absent=True)
    for cx in COL_CENTERS[2:]:
        fill = C["new_fill"] if cx == COL_CENTERS[2] else C["base_fill"]
        box(fig, cx, ROWS["interact"],
            "Token-Time<br>Interaction", "W [e(w); τ(t)]", C["interaction"],
            fill=fill)

    # ── Row: Transformer Encoder (all variants) ───────────────────────────────
    for cx in COL_CENTERS:
        box(fig, cx, ROWS["encoder"],
            "Transformer<br>Encoder", "contextualizes sentence", C["encoder"])

    # ── Row: output / memory ──────────────────────────────────────────────────
    # Static, Additive, Joint: h_s sits at ROWS["memory"]
    for cx in COL_CENTERS[:3]:
        box(fig, cx, ROWS["memory"],
            "h_s", "token@time state", C["output"])

    # Timeformer: Prototype Memory + Temporal Attention side by side at this level
    mem_cx  = COL_CENTERS[3] - MEM_OFFSET
    attn_cx = COL_CENTERS[3] + MEM_OFFSET

    # Prototype Memory box
    fig.add_shape(type="rect",
                  x0=mem_cx  - HALF_W_TF, y0=MEM_BOX_Y0,
                  x1=mem_cx  + HALF_W_TF, y1=MEM_BOX_Y1,
                  line={"color": C["memory"], "width": 2},
                  fillcolor=C["new_fill"], layer="below")
    fig.add_annotation(x=mem_cx, y=MEM_BOX_Y1 - 0.022,
                       text="<b>Prototype<br>Memory</b>",
                       showarrow=False,
                       font={"size": 9, "color": C["memory"]}, align="center")
    fig.add_annotation(x=mem_cx, y=MEM_BOX_Y0 + 0.022,
                       text="m(s,t),  t &lt; t_k",
                       showarrow=False,
                       font={"size": 8, "color": C["ink"]}, align="center")

    # Temporal Attention box
    fig.add_shape(type="rect",
                  x0=attn_cx - HALF_W_TF, y0=MEM_BOX_Y0,
                  x1=attn_cx + HALF_W_TF, y1=MEM_BOX_Y1,
                  line={"color": C["memory"], "width": 2},
                  fillcolor=C["new_fill"], layer="below")
    fig.add_annotation(x=attn_cx, y=MEM_BOX_Y1 - 0.022,
                       text="<b>Temporal<br>Attention</b>",
                       showarrow=False,
                       font={"size": 9, "color": C["memory"]}, align="center")
    fig.add_annotation(x=attn_cx, y=MEM_BOX_Y0 + 0.022,
                       text="gate g ∈ [0,1]",
                       showarrow=False,
                       font={"size": 8, "color": C["ink"]}, align="center")

    # h_s for Timeformer sits below the memory boxes
    fig.add_shape(type="rect",
                  x0=COL_CENTERS[3] - COL_W, y0=HS_Y_TF - BOX_H * 0.8,
                  x1=COL_CENTERS[3] + COL_W, y1=HS_Y_TF + BOX_H * 0.8,
                  line={"color": C["output"], "width": 2},
                  fillcolor=C["base_fill"], layer="below")
    fig.add_annotation(x=COL_CENTERS[3], y=HS_Y_TF + BOX_H * 0.45,
                       text="<b>h_s</b>",
                       showarrow=False,
                       font={"size": 10, "color": C["output"]}, align="center")
    fig.add_annotation(x=COL_CENTERS[3], y=HS_Y_TF - BOX_H * 0.45,
                       text="token@time state",
                       showarrow=False,
                       font={"size": 9, "color": C["ink"]}, align="center")

    # ── Arrows ────────────────────────────────────────────────────────────────
    tok_bot = ROWS["token"]   - BOX_H
    tim_top = ROWS["time"]    + BOX_H
    tim_bot = ROWS["time"]    - BOX_H
    int_top = ROWS["interact"] + BOX_H
    int_bot = ROWS["interact"] - BOX_H
    enc_top = ROWS["encoder"] + BOX_H
    enc_bot = ROWS["encoder"] - BOX_H
    hs_top  = ROWS["memory"]  + BOX_H   # top of h_s for Static/Additive/Joint

    # Static: Token → Encoder → h_s  (skips time and interact rows)
    arrow(fig, COL_CENTERS[0], tok_bot, COL_CENTERS[0], enc_top, C["token"])
    arrow(fig, COL_CENTERS[0], enc_bot, COL_CENTERS[0], hs_top,  C["encoder"])

    # Additive: Token → Encoder; Time → Encoder (by addition); Encoder → h_s
    arrow(fig, COL_CENTERS[1], tok_bot, COL_CENTERS[1], enc_top, C["token"])
    arrow(fig, COL_CENTERS[1], tim_bot, COL_CENTERS[1], enc_top, C["time"])
    arrow(fig, COL_CENTERS[1], enc_bot, COL_CENTERS[1], hs_top,  C["encoder"])
    fig.add_annotation(x=COL_CENTERS[1] + 0.022,
                       y=(tim_bot + enc_top) / 2,
                       text="<i>add</i>",
                       showarrow=False,
                       font={"size": 9, "color": C["time"]})

    # Joint: Token → Interact; Time → Interact; Interact → Encoder → h_s
    arrow(fig, COL_CENTERS[2], tok_bot, COL_CENTERS[2], int_top, C["token"])
    arrow(fig, COL_CENTERS[2], tim_bot, COL_CENTERS[2], int_top, C["time"])
    arrow(fig, COL_CENTERS[2], int_bot, COL_CENTERS[2], enc_top, C["interaction"])
    arrow(fig, COL_CENTERS[2], enc_bot, COL_CENTERS[2], hs_top,  C["encoder"])

    # Timeformer: same as Joint up to encoder, then memory gates into encoder, then h_s
    arrow(fig, COL_CENTERS[3], tok_bot, COL_CENTERS[3], int_top, C["token"])
    arrow(fig, COL_CENTERS[3], tim_bot, COL_CENTERS[3], int_top, C["time"])
    arrow(fig, COL_CENTERS[3], int_bot, COL_CENTERS[3], enc_top, C["interaction"])
    # Memory → Temporal Attention (horizontal, through the gap between boxes)
    arrow(fig, mem_cx  + HALF_W_TF, (MEM_BOX_Y0 + MEM_BOX_Y1) / 2,
               attn_cx - HALF_W_TF, (MEM_BOX_Y0 + MEM_BOX_Y1) / 2,
               C["memory"])
    # Temporal Attention → Encoder (diagonal upward)
    arrow(fig, attn_cx, MEM_BOX_Y1, COL_CENTERS[3], enc_bot, C["memory"])
    # Encoder → h_s (vertical; passes through the gap between the two sub-boxes)
    arrow(fig, COL_CENTERS[3], enc_bot,
               COL_CENTERS[3], HS_Y_TF + BOX_H * 0.8, C["encoder"])

    # ── Legend ────────────────────────────────────────────────────────────────
    fig.add_shape(type="rect", x0=0.03, y0=0.033, x1=0.085, y1=0.062,
                  line={"color": C["interaction"], "width": 1.5},
                  fillcolor=C["new_fill"], layer="below")
    fig.add_annotation(x=0.092, y=0.048,
                       text="new component at this step",
                       showarrow=False,
                       font={"size": 9, "color": C["muted"]},
                       xanchor="left")
    fig.add_shape(type="rect", x0=0.40, y0=0.033, x1=0.455, y1=0.062,
                  line={"color": C["absent_line"], "width": 1.5, "dash": "dot"},
                  fillcolor="#f8f8f8", layer="below")
    fig.add_annotation(x=0.462, y=0.048,
                       text="component absent in this variant",
                       showarrow=False,
                       font={"size": 9, "color": C["muted"]},
                       xanchor="left")

    fig.update_layout(
        template="plotly_white",
        width=1400,
        height=700,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        title={"text": "Figure 2. Timeformer ablation chain",
               "x": 0.02, "xanchor": "left"},
        font={"family": "Arial", "size": 12, "color": C["ink"]},
        xaxis={"range": [0, 1], "visible": False, "fixedrange": True},
        yaxis={"range": [0, 1], "visible": False, "fixedrange": True},
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.write_html(HTML_PATH, include_plotlyjs=True, full_html=True)
    fig.write_json(JSON_PATH)
    print(f"Wrote {HTML_PATH}")
    print(f"Wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
