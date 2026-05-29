"""
Executa N seeds sequencialmente e agrega resultados em médias + ICs 95%.

Para cada seed:
  1. run_phase_b.py (treina Static/Additive/Joint/Timeformer e avalia)
  2. neighbor_analysis.py (computa drift scores por classe)

Ao final salva outputs/multiseed/multiseed_raw.json e multiseed_stats.json.

Uso:
  python scripts/run_multiseed.py --n-seeds 30
  python scripts/run_multiseed.py --n-seeds 30 --start-seed 100   # continua
  python scripts/run_multiseed.py --stats-only                     # só agrega resultados existentes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("outputs/multiseed")
RAW_PATH = OUT_DIR / "multiseed_raw.json"
STATS_PATH = OUT_DIR / "multiseed_stats.json"

MODELS = ["Static", "Additive", "Joint", "Timeformer"]
CLASSES = ["stable", "drift", "bifurc"]


# ── Execução de um seed ───────────────────────────────────────────────────────

def run_seed(seed: int, epochs: int, device: str) -> str | None:
    """Treina e avalia um seed. Retorna run_id ou None se falhou."""
    cmd_train = [
        sys.executable, "run_phase_b.py",
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--device", device,
        "--skip-contrastive",
        "--notes", f"multiseed seed={seed}",
    ]
    result = subprocess.run(cmd_train, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [seed={seed}] ERRO no treino:\n{result.stderr[-500:]}")
        return None

    # Extrai run_id da última linha que menciona "Run:"
    run_id = None
    for line in result.stdout.splitlines():
        if line.strip().startswith("Run:"):
            run_id = line.split("Run:")[-1].strip()

    if not run_id:
        print(f"  [seed={seed}] Não encontrou run_id no stdout")
        return None

    cmd_neighbor = [
        sys.executable, "scripts/neighbor_analysis.py",
        "--run-id", run_id,
        "--device", device,
        "--models", "Static", "Additive", "Joint", "Timeformer",
    ]
    result2 = subprocess.run(cmd_neighbor, capture_output=True, text=True)
    if result2.returncode != 0:
        print(f"  [seed={seed}] ERRO na análise de vizinhança:\n{result2.stderr[-500:]}")
        return run_id  # run_id válido, mas sem drift scores

    return run_id


# ── Coleta de métricas de um run ──────────────────────────────────────────────

def collect_run_metrics(run_id: str) -> dict | None:
    """
    Coleta as métricas necessárias de um run já completo.
    Retorna dict com probe accs, sign-flip, e drift scores por classe.
    """
    base = Path(f"outputs/runs/{run_id}/results")

    results_path = base / "results_full.json"
    neighbor_path = base / "neighbor_analysis.json"

    if not results_path.exists():
        return None

    with open(results_path) as f:
        full = json.load(f)

    metrics: dict = {"run_id": run_id, "seed": None, "probe": {}, "sign_flip": {}, "drift": {}}

    for model in MODELS:
        if model not in full:
            continue
        r = full[model]

        metrics["probe"][model] = {
            "test":         r.get("test", {}).get("probe_subj", {}).get("accuracy"),
            "ambiguous":    r.get("ambiguous_test", {}).get("probe_subj", {}).get("accuracy"),
            "continuation": r.get("continuation", {}).get("probe_subj", {}).get("accuracy"),
        }
        metrics["sign_flip"][model] = r.get("contrastive", {}).get("sign_flip_rate")

    if neighbor_path.exists():
        with open(neighbor_path) as f:
            nb = json.load(f)

        drift_by_class = nb.get("drift_score_by_class", {})
        for model in MODELS:
            if model not in drift_by_class:
                continue
            metrics["drift"][model] = {}
            for cls in CLASSES:
                scores = drift_by_class[model].get(cls, {})
                t0 = scores.get("0") or scores.get(0)
                t9 = scores.get("9") or scores.get(9)
                if t0 is not None and t9 is not None:
                    metrics["drift"][model][cls] = {
                        "t0": t0, "t9": t9, "delta": t9 - t0,
                    }

    return metrics


# ── Agregação estatística ─────────────────────────────────────────────────────

def ci95(values: list[float]) -> tuple[float, float, float]:
    """Retorna (mean, lower_ci, upper_ci) com IC 95% bootstrap-free (normal approx)."""
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, mean - 1.96 * se, mean + 1.96 * se


def aggregate(raw: list[dict]) -> dict:
    """Agrega lista de métricas por seed em médias e ICs."""
    stats: dict = {"n_seeds": len(raw), "probe": {}, "sign_flip": {}, "drift": {}}

    for model in MODELS:
        stats["probe"][model] = {}
        for split in ("test", "ambiguous", "continuation"):
            vals = [r["probe"].get(model, {}).get(split) for r in raw if r.get("probe", {}).get(model)]
            mean, lo, hi = ci95(vals)
            stats["probe"][model][split] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi}

        vals_sf = [r["sign_flip"].get(model) for r in raw if r.get("sign_flip", {}).get(model) is not None]
        mean, lo, hi = ci95(vals_sf)
        stats["sign_flip"][model] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi}

        stats["drift"][model] = {}
        for cls in CLASSES:
            for stat in ("t0", "t9", "delta"):
                vals_d = [
                    r.get("drift", {}).get(model, {}).get(cls, {}).get(stat)
                    for r in raw
                ]
                mean, lo, hi = ci95(vals_d)
                key = f"{cls}_{stat}"
                stats["drift"][model][key] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi}

    return stats


def print_summary(stats: dict) -> None:
    n = stats["n_seeds"]
    print(f"\n{'='*60}")
    print(f"RESUMO — {n} seeds")
    print(f"{'='*60}")

    print("\nDrift score Δ(t9−t0) — classe drift:")
    for model in MODELS:
        d = stats["drift"].get(model, {}).get("drift_delta", {})
        mean, lo, hi = d.get("mean"), d.get("ci95_lo"), d.get("ci95_hi")
        if mean is not None and not np.isnan(mean):
            print(f"  {model:<12}  {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    print("\nProbe accuracy — ambiguous split:")
    for model in MODELS:
        d = stats["probe"].get(model, {}).get("ambiguous", {})
        mean, lo, hi = d.get("mean"), d.get("ci95_lo"), d.get("ci95_hi")
        if mean is not None and not np.isnan(mean):
            print(f"  {model:<12}  {mean:.3f}  [{lo:.3f}, {hi:.3f}]")

    print("\nSign-flip rate (D3):")
    for model in MODELS:
        d = stats["sign_flip"].get(model, {})
        mean, lo, hi = d.get("mean"), d.get("ci95_lo"), d.get("ci95_hi")
        if mean is not None and not np.isnan(mean):
            print(f"  {model:<12}  {mean:.3f}  [{lo:.3f}, {hi:.3f}]")

    print("\nProbe accuracy — continuation split:")
    for model in MODELS:
        d = stats["probe"].get(model, {}).get("continuation", {})
        mean, lo, hi = d.get("mean"), d.get("ci95_lo"), d.get("ci95_hi")
        if mean is not None and not np.isnan(mean):
            print(f"  {model:<12}  {mean:.3f}  [{lo:.3f}, {hi:.3f}]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds",    type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=0,
                        help="Primeiro seed (útil para continuar uma rodada)")
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--stats-only", action="store_true",
                        help="Só agrega resultados de runs já existentes no raw.json")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Carrega dados já coletados se existirem
    raw: list[dict] = []
    if RAW_PATH.exists():
        with open(RAW_PATH) as f:
            raw = json.load(f)
        print(f"Carregados {len(raw)} seeds de {RAW_PATH}")

    if not args.stats_only:
        seeds_done = {r["run_id"] for r in raw}  # não serve diretamente, mas evita re-coletar

        for i in range(args.n_seeds):
            seed = args.start_seed + i
            print(f"\n[{i+1}/{args.n_seeds}] seed={seed}")

            run_id = run_seed(seed, args.epochs, args.device)
            if run_id is None:
                print(f"  Pulando seed={seed}")
                continue

            metrics = collect_run_metrics(run_id)
            if metrics is None:
                print(f"  Sem métricas para {run_id}")
                continue

            metrics["seed"] = seed
            raw.append(metrics)
            # Salva incrementalmente
            RAW_PATH.write_text(json.dumps(raw, indent=2))
            print(f"  run_id={run_id}  seeds coletados={len(raw)}")

    if not raw:
        print("Nenhum dado para agregar.")
        return

    stats = aggregate(raw)
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"\nEstatísticas salvas em {STATS_PATH}")
    print_summary(stats)


if __name__ == "__main__":
    main()
