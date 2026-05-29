"""
Roda o oracle diagnostic para cada seed já coletado em multiseed_raw.json.

Para cada run_id:
  1. Constrói OracleMemory (determinística — mesmo resultado em todo seed)
  2. Treina Timeformer-oracle com o mesmo seed usado no run original
  3. Avalia: continuation (D4) e sign-flip (D3)
  4. Adiciona chave "oracle" ao registro do seed em multiseed_raw.json

Ao final salva outputs/multiseed/multiseed_oracle_stats.json com médias e ICs.

Uso:
  python scripts/run_multiseed_oracle.py
  python scripts/run_multiseed_oracle.py --epochs 30  # padrão igual ao run_multiseed
  python scripts/run_multiseed_oracle.py --stats-only  # só agrega sem treinar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.timeformer.dataset import (
    load_corpus, MLMDataset, TimeformerDataset, make_continuation_split,
    context_to_id, SUBJECTS, N_EPOCHS,
)
from src.timeformer.models import build_model, DEFAULT_HPARAMS
from src.timeformer.memory import PrototypeMemory
from src.timeformer.train import MLMTrainer, load_checkpoint
from src.timeformer.eval import Evaluator
from src.timeformer.run import RunManager

CORPUS_PATH      = Path("data/corpus.tsv")
AMBIGUOUS_PATH   = Path("data/corpus_ambiguous.tsv")
CONTRASTIVE_PATH = Path("data/contrastive_set.tsv")
RAW_PATH         = Path("outputs/multiseed/multiseed_raw.json")
ORACLE_STATS     = Path("outputs/multiseed/multiseed_oracle_stats.json")


# ── Oracle Memory (determinística) ────────────────────────────────────────────

_oracle_mem_cache: PrototypeMemory | None = None

def get_oracle_memory(d_model: int, device: str) -> PrototypeMemory:
    global _oracle_mem_cache
    if _oracle_mem_cache is not None:
        return _oracle_mem_cache.to(device)

    rows = load_corpus(CORPUS_PATH)
    train_rows, _ = make_continuation_split(rows)
    n_subjects = len(SUBJECTS)
    subj2idx = {s: i for i, s in enumerate(SUBJECTS)}

    counts_a     = np.zeros((n_subjects, N_EPOCHS))
    counts_total = np.zeros((n_subjects, N_EPOCHS))
    for r in train_rows:
        s = subj2idx[r["subject"]]
        t = int(r["epoch"][1:])
        counts_total[s, t] += 1
        if context_to_id(r["true_context"]) == 0:
            counts_a[s, t] += 1

    p_a = np.where(counts_total > 0, counts_a / counts_total, 0.5)

    proto_norm = 11.0
    rng = np.random.default_rng(0)
    raw_A = rng.standard_normal(d_model).astype(np.float32)
    raw_B = rng.standard_normal(d_model).astype(np.float32)
    raw_B -= raw_B.dot(raw_A) / (raw_A.dot(raw_A)) * raw_A
    v_A = torch.tensor(raw_A / np.linalg.norm(raw_A) * proto_norm)
    v_B = torch.tensor(raw_B / np.linalg.norm(raw_B) * proto_norm)

    mem = PrototypeMemory(n_subjects, N_EPOCHS, d_model, device)
    for s in range(n_subjects):
        for t in range(N_EPOCHS):
            if counts_total[s, t] > 0:
                pa = float(p_a[s, t])
                mem._protos[s, t, :] = pa * v_A + (1 - pa) * v_B
                mem._valid[s, t] = True

    _oracle_mem_cache = mem
    return mem


# ── Treino com oracle memory fixada ──────────────────────────────────────────

def train_oracle(seed: int, epochs: int, device: str) -> tuple[torch.nn.Module, PrototypeMemory]:
    d_model = DEFAULT_HPARAMS["d_model"]
    oracle_mem = get_oracle_memory(d_model, device)

    rows = load_corpus(CORPUS_PATH)
    train_rows, _ = make_continuation_split(rows)
    val_rows = [r for r in rows if r["split"] == "test"]

    train_ds = TimeformerDataset(train_rows, seed=seed)
    val_ds   = MLMDataset(val_rows, seed=seed)

    model = build_model("Timeformer")
    out_dir = Path(f"outputs/multiseed/oracle_runs/seed_{seed:04d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = MLMTrainer(model, output_dir=out_dir, device=device)

    # Monkey-patch: não atualiza a oracle memory durante treino
    import types

    def _train_no_mem_update(self, train_dataset, val_dataset=None, memory=None,
                             n_epochs=30, batch_size=64, lr=1e-3, seed=42):
        import time
        from torch.utils.data import DataLoader
        from src.timeformer.dataset import timeformer_collate_fn

        torch.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=timeformer_collate_fn,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) \
            if val_dataset is not None else None

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        history = []
        best_val = float("inf")
        for epoch in range(n_epochs):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader, optimizer, memory)
            val_loss   = self._eval_epoch(val_loader, memory) if val_loader else None
            scheduler.step()
            record = {"epoch": epoch, "train_loss": train_loss,
                      "val_loss": val_loss, "elapsed_s": round(time.time() - t0, 2)}
            history.append(record)
            monitor = val_loss if val_loss is not None else train_loss
            if monitor < best_val:
                best_val = monitor
                self._save_checkpoint("best.pt")
        self._save_checkpoint("final.pt")
        self._save_history(history)
        return history

    trainer.train = types.MethodType(_train_no_mem_update, trainer)

    trainer.train(
        train_ds, val_ds,
        memory=oracle_mem,
        n_epochs=epochs,
        batch_size=64,
        lr=1e-3,
        seed=seed,
    )

    load_checkpoint(model, out_dir / "best.pt")
    return model, oracle_mem


# ── Avaliação ─────────────────────────────────────────────────────────────────

def evaluate_oracle(model, memory, device: str) -> dict:
    evaluator = Evaluator(
        corpus_path=CORPUS_PATH,
        ambiguous_path=AMBIGUOUS_PATH,
        contrastive_path=CONTRASTIVE_PATH,
        device=device,
    )
    res = evaluator.evaluate(model, memory=memory)
    return {
        "continuation": res.get("continuation", {}).get("probe_subj", {}).get("accuracy"),
        "sign_flip":    res.get("contrastive", {}).get("sign_flip_rate"),
        "ambiguous":    res.get("ambiguous_test", {}).get("probe_subj", {}).get("accuracy"),
    }


# ── Agregação ─────────────────────────────────────────────────────────────────

def ci95(vals: list) -> dict:
    arr = np.array([v for v in vals if v is not None and not np.isnan(float(v))])
    if len(arr) == 0:
        return {"mean": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    mean = float(arr.mean())
    se   = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"mean": mean, "ci95_lo": mean - 1.96 * se, "ci95_hi": mean + 1.96 * se}


def aggregate_oracle(raw: list[dict]) -> dict:
    oracle_rows = [r["oracle"] for r in raw if "oracle" in r]
    n = len(oracle_rows)

    cont_vals  = [r.get("continuation") for r in oracle_rows]
    flip_vals  = [r.get("sign_flip")    for r in oracle_rows]
    ambig_vals = [r.get("ambiguous")    for r in oracle_rows]

    joint_cont  = [r.get("probe", {}).get("Joint", {}).get("continuation") for r in raw if "oracle" in r]
    joint_flip  = [r.get("sign_flip", {}).get("Joint") for r in raw if "oracle" in r]

    delta_cont = [
        (c - j) if (c is not None and j is not None) else None
        for c, j in zip(cont_vals, joint_cont)
    ]
    delta_flip = [
        (f - j) if (f is not None and j is not None) else None
        for f, j in zip(flip_vals, joint_flip)
    ]

    return {
        "n_seeds":        n,
        "continuation":   ci95(cont_vals),
        "sign_flip":      ci95(flip_vals),
        "ambiguous":      ci95(ambig_vals),
        "delta_cont_vs_joint": ci95(delta_cont),
        "delta_flip_vs_joint": ci95(delta_flip),
    }


def print_oracle_summary(stats: dict, raw_stats_path: Path) -> None:
    n = stats["n_seeds"]

    def fmt(d: dict) -> str:
        m, lo, hi = d["mean"], d["ci95_lo"], d["ci95_hi"]
        if np.isnan(m):
            return "---"
        hw = (hi - lo) / 2
        return f"{m:.3f} ± {hw:.3f}"

    print(f"\n{'='*55}")
    print(f"Oracle diagnostic — {n} seeds")
    print(f"{'='*55}")
    print(f"  Continuation (D4):          {fmt(stats['continuation'])}")
    print(f"  Sign-flip    (D3):          {fmt(stats['sign_flip'])}")
    print(f"  Ambiguous probe:            {fmt(stats['ambiguous'])}")
    print(f"  Δ cont  vs. Joint:          {fmt(stats['delta_cont_vs_joint'])}")
    print(f"  Δ flip  vs. Joint:          {fmt(stats['delta_flip_vs_joint'])}")

    # Compara com Timeformer-learned (se disponível em multiseed_stats.json)
    ms_path = Path("outputs/multiseed/multiseed_stats.json")
    if ms_path.exists():
        with open(ms_path) as f:
            ms = json.load(f)
        learned_cont = ms.get("probe", {}).get("Timeformer", {}).get("continuation", {})
        learned_flip = ms.get("sign_flip", {}).get("Timeformer", {})
        print(f"\n  Mem-Aug. learned cont.:     {fmt(learned_cont)}")
        print(f"  Mem-Aug. oracle  cont.:     {fmt(stats['continuation'])}")
        print(f"  Mem-Aug. learned flip:      {fmt(learned_flip)}")
        print(f"  Mem-Aug. oracle  flip:      {fmt(stats['sign_flip'])}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    if not RAW_PATH.exists():
        print(f"Não encontrou {RAW_PATH}. Rode run_multiseed.py primeiro.")
        return

    with open(RAW_PATH) as f:
        raw = json.load(f)

    if not args.stats_only:
        for i, record in enumerate(raw):
            if "oracle" in record:
                print(f"[{i+1}/{len(raw)}] seed em {record['run_id']} — oracle já feito, pulando")
                continue

            seed = record.get("seed", i)  # usa seed gravado; fallback: posição na lista
            print(f"\n[{i+1}/{len(raw)}] Treinando oracle seed={seed} (run={record['run_id']})")

            model, oracle_mem = train_oracle(seed, args.epochs, args.device)
            oracle_metrics = evaluate_oracle(model, oracle_mem, args.device)
            record["oracle"] = oracle_metrics

            RAW_PATH.write_text(json.dumps(raw, indent=2))
            print(f"  cont={oracle_metrics.get('continuation', 'nan'):.3f}  "
                  f"flip={oracle_metrics.get('sign_flip', 'nan'):.3f}")

    oracle_records = [r for r in raw if "oracle" in r]
    if not oracle_records:
        print("Nenhum resultado oracle disponível para agregar.")
        return

    stats = aggregate_oracle(raw)
    ORACLE_STATS.write_text(json.dumps(stats, indent=2))
    print(f"\nEstatísticas oracle salvas em {ORACLE_STATS}")
    print_oracle_summary(stats, ORACLE_STATS)


if __name__ == "__main__":
    main()
