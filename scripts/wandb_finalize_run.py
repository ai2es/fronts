"""Mark a W&B run as finished after it was wrongly flagged "crashed".

Moving a run between projects mid-flight breaks the client's heartbeat, so the server flags
it crashed even though training completed. Resuming the run by id and calling ``finish()``
sends a clean exit, which flips the state to "finished". Run state is not editable through
``Api().run(...).update()``, so this is the supported route.

If training metrics stopped arriving after the move, sync the local offline data first
(from the directory training ran in):
    wandb sync --id <run_id> -p <new_project> wandb/run-*<run_id>/

Usage:
    python scripts/wandb_finalize_run.py --entity E --project fronts \
        --name nbs_square_dilation_trof_tt_no_inst_no_dewpoint
"""

import argparse

import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True, help="The project the run was moved INTO")
    parser.add_argument("--name", help="Run display name (required unless --run_id is given)")
    parser.add_argument("--run_id", help="Skip the name lookup and use this run id")
    args = parser.parse_args()
    if not (args.run_id or args.name):
        parser.error("one of --run_id or --name is required")

    api = wandb.Api()
    if args.run_id:
        run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
    else:
        matches = list(api.runs(f"{args.entity}/{args.project}", filters={"display_name": args.name}))
        if len(matches) != 1:
            raise SystemExit(f"Expected 1 run named {args.name!r}, found {len(matches)}: {[r.id for r in matches]}")
        run = matches[0]
    print(f"Found {run.path} state={run.state}")

    resumed = wandb.init(entity=args.entity, project=args.project, id=run.id, resume="allow")
    resumed.finish(exit_code=0)
    print(f"State after finish: {api.run('/'.join(run.path)).state}")


if __name__ == "__main__":
    main()
