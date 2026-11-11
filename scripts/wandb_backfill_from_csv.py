"""Backfill per-epoch metrics missing from a W&B run, using the training run's metrics_epoch.csv.

Training writes every epoch to ``metrics_epoch.csv`` (next to the checkpoint) independently of
W&B, so epochs trained after uploads stopped can be re-logged from it. Rows are logged exactly
as ``WandbMetricsLogger`` does: each column ``c`` becomes ``epoch/c`` and ``epoch/epoch`` carries
the epoch number, which the run already defines as the x-axis for every ``epoch/*`` metric.

Only epoch-level scalars are recoverable this way. Per-batch metrics (``batch/*``) and the
periodic ``test/*`` images for the missing epochs are not in the CSV and are not backfilled.

Dry run by default; pass --execute to write. Usage:
    python scripts/wandb_backfill_from_csv.py --entity extreme-earth --project fronts \
        --run_id 1u8316rt --csv /ourdisk/hpc/ai2es/tman/models/<run>/metrics_epoch.csv
"""

import argparse
import csv
import math

import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--csv", required=True, help="Path to the run's metrics_epoch.csv")
    parser.add_argument("--execute", action="store_true", help="Actually log (default: dry run)")
    args = parser.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "epoch" not in rows[0]:
        raise SystemExit(f"{args.csv}: no rows or no 'epoch' column")
    csv_last = max(int(r["epoch"]) for r in rows)

    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
    logged = [r["epoch/epoch"] for r in run.scan_history(keys=["epoch/epoch"]) if r.get("epoch/epoch") is not None]
    wandb_last = int(max(logged)) if logged else -1
    todo = [r for r in rows if int(r["epoch"]) > wandb_last]

    print(f"W&B has epochs up to {wandb_last} ({len(logged)} rows); CSV has epochs up to {csv_last} ({len(rows)} rows)")
    print(f"To backfill: {len(todo)} epochs" + (f" ({todo[0]['epoch']}..{todo[-1]['epoch']})" if todo else ""))
    if todo:
        print("Columns:", ", ".join(c for c in rows[0] if c != "epoch"))
    if not args.execute or not todo:
        print("Dry run only." if not args.execute else "Nothing to do.")
        return

    resumed = wandb.init(entity=args.entity, project=args.project, id=args.run_id, resume="allow")
    for r in todo:
        payload = {"epoch/epoch": int(r["epoch"])}
        for key, value in r.items():
            if key == "epoch" or value in ("", None):
                continue
            number = float(value)
            if not math.isnan(number):
                payload[f"epoch/{key}"] = number
        wandb.log(payload)
    resumed.finish(exit_code=0)
    print(f"Backfilled {len(todo)} epochs.")


if __name__ == "__main__":
    main()
