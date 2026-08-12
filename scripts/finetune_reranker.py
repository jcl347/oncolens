#!/usr/bin/env python
"""Fine-tune a cross-encoder on this corpus's own citation pairs.

**The measurement that motivates it (§4.18).** The correct paper is in the retrieved pool
**96.2%** of the time and ranked first **39.0%** of the time. That 0.57 gap is ranking, not
recall. `rerank_medcpt_cross` — a *general* biomedical reranker — closed part of it for
+0.0441 mrr. Nothing else in the system has that much headroom left.

**Starting from MedCPT-Cross rather than a bare encoder.** It already scores +0.0441 here,
so it is a strictly better initialisation than PubMedBERT, and the risk it carries —
catastrophic forgetting — is bounded by a low learning rate and a single epoch. Training
from scratch on 3.5k queries would almost certainly be worse.

⚠️ **Leakage.** Training pairs come from `ft_train`; the resulting model is scored on
`ft_holdout`. Both partition DEV. The locked `test` split is touched by neither, and
`load_qrels` verifies the partition is exact with zero overlap.

⚠️ **What a win here would and would not mean.** Beating the base model on `ft_holdout`
shows the task-specific signal is learnable. It does NOT license shipping: `ft_holdout` is
a slice of dev, so it is still the data this loop has been iterating against all along.
Shipping still requires the Pareto rule across strata and one spend of the test split.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.retrieval.cross_encoder import MEDCPT_CROSS, finetuned_path  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--base", default=MEDCPT_CROSS)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="low on purpose: this is adaptation, not training")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    load_env(REPO)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pairs_path = Path(args.pairs) if args.pairs else local_data_dir() / "finetune_pairs.jsonl"
    if not pairs_path.exists():
        raise SystemExit(f"no pairs at {pairs_path}; run scripts/build_finetune_data.py")
    rows = [json.loads(l) for l in pairs_path.read_text(encoding="utf-8").splitlines() if l]
    print(f"pairs: {len(rows):,}  "
          f"({sum(r['label'] for r in rows):,} positive)")

    # Split by QUERY, never by row: the positive and its four hard negatives share a query,
    # so a row-level split would put the same query on both sides and the validation loss
    # would measure memorisation.
    qids = sorted({r["qid"] for r in rows})
    random.Random(23).shuffle(qids)
    n_val = max(1, int(len(qids) * args.val_frac))
    val_q = set(qids[:n_val])
    train = [r for r in rows if r["qid"] not in val_q]
    val = [r for r in rows if r["qid"] in val_q]
    print(f"  train {len(train):,} rows / {len(qids)-n_val:,} queries")
    print(f"  val   {len(val):,} rows / {n_val:,} queries  (split by query, not by row)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\ndevice: {dev}  base: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1)
    model.to(dev)

    class Pairs(Dataset):
        def __init__(self, rs): self.rs = rs
        def __len__(self): return len(self.rs)
        def __getitem__(self, i): return self.rs[i]

    def collate(batch):
        enc = tok([b["query"] for b in batch], [b["passage"] for b in batch],
                  truncation="only_second", padding=True,
                  max_length=args.max_length, return_tensors="pt")
        enc["labels"] = torch.tensor([[float(b["label"])] for b in batch])
        return enc

    dl = DataLoader(Pairs(train), batch_size=args.batch, shuffle=True, collate_fn=collate)
    vdl = DataLoader(Pairs(val), batch_size=args.batch * 2, collate_fn=collate)

    steps = int(len(dl) * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1)
    lossf = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda") if dev == "cuda" else None

    def validate() -> tuple[float, float]:
        model.eval()
        losses, correct, n = [], 0, 0
        with torch.no_grad():
            for b in vdl:
                lab = b.pop("labels").to(dev)
                b = {k: v.to(dev) for k, v in b.items()}
                lg = model(**b).logits
                losses.append(lossf(lg, lab).item())
                correct += ((lg > 0).float() == lab).sum().item()
                n += lab.numel()
        model.train()
        return float(np.mean(losses)), correct / max(n, 1)

    vl, va = validate()
    print(f"\nbefore training: val loss {vl:.4f}  acc {va:.3f}")

    model.train()
    step = 0
    for epoch in range(int(args.epochs) + 1):
        for b in dl:
            if step >= steps:
                break
            lab = b.pop("labels").to(dev)
            b = {k: v.to(dev) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = lossf(model(**b).logits, lab)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss = lossf(model(**b).logits, lab)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"  step {step:>5}/{steps}  loss {loss.item():.4f}")
        if step >= steps:
            break

    vl2, va2 = validate()
    print(f"\nafter training : val loss {vl2:.4f}  acc {va2:.3f}   "
          f"(Δ loss {vl2-vl:+.4f}, Δ acc {va2-va:+.3f})")
    if vl2 >= vl:
        print("  ⚠️  validation loss did NOT improve. Recorded, not hidden: this is the "
              "signal that adaptation failed, and the retrieval eval will likely agree.")

    out = finetuned_path()
    out.mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(out)   # fp16 to match how the reranker serves
    tok.save_pretrained(out)
    (out / "training_meta.json").write_text(json.dumps({
        "base": args.base, "pairs": len(rows), "train_rows": len(train),
        "val_rows": len(val), "steps": steps, "lr": args.lr,
        "val_loss_before": vl, "val_loss_after": vl2,
        "val_acc_before": va, "val_acc_after": va2,
        "split": "ft_train (partitions DEV; test never touched)",
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    print("Next: improve_loop.py --run rerank_finetuned --stratum claim --split ft_holdout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
