"""
tools/sweeper.py

Provides utilities to run systematic experiments:
 - Local grid/random search runner that executes run_experiment() programmatically
   and logs results to CSV (for reproducible offline sweeps).
 - Optional wandb sweep launcher wrapper (delegates to wandb when available).
Usage (local grid):
  python -m tools.sweeper --mode grid --grid examples/grid_params.json --out runs/grid_results.csv

Usage (wandb sweep):
  python -m tools.sweeper --mode wandb --sweep examples/wandb_sweep.yaml --config examples/speeddating_config.json

Notes:
 - This module imports run_experiment from tools.experiment and expects that function
   returns (trained_model, cfg, history).
 - Local grid mode runs experiments sequentially in the current process; for parallel runs
   consider launching multiple processes or using a cluster/CI runner.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import time
from itertools import product
from typing import Any, Dict, Iterable, List, Optional
import logging
import traceback
import sys

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    wandb = None
    _HAS_WANDB = False

from tools.experiment import run_experiment  # expects run_experiment(cfg) -> (model, cfg, history)


def _expand_grid(grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """
    Expand a dict of lists into a list of config overrides.
    Example: {"dropout":[0.1,0.3], "use_skip":[true,false]} -> 4 combinations.
    """
    keys = sorted(grid.keys())
    values = [list(grid[k]) for k in keys]
    combos = []
    for prod in product(*values):
        combo = dict(zip(keys, prod))
        combos.append(combo)
    return combos


def _merge_cfg(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(base)
    for k, v in overrides.items():
        # handle nested override for optimizer.lr e.g. "optimizer.lr": 1e-4
        if "." in k:
            parts = k.split(".")
            sub = cfg
            for p in parts[:-1]:
                if p not in sub or not isinstance(sub[p], dict):
                    sub[p] = {}
                sub = sub[p]
            sub[parts[-1]] = v
        else:
            cfg[k] = v
    return cfg


def run_local_grid_search(base_cfg: Dict[str, Any], grid: Dict[str, Iterable[Any]], out_csv: str, run_name_prefix: str = "grid"):
    combos = _expand_grid(grid)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = ["run_id", "experiment_name", "duration_s", "avg_train_loss", "avg_val_loss", "final_val_acc", "n_epochs", "config"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, combo in enumerate(combos):
            run_id = f"{run_name_prefix}_{i+1:03d}"
            cfg = _merge_cfg(base_cfg, combo)
            cfg["experiment_name"] = cfg.get("experiment_name", run_id)
            start = time.time()
            print(f"\n=== Running experiment {i+1}/{len(combos)} -> {cfg['experiment_name']} ===")
            try:
                _, used_cfg, history = run_experiment(cfg)
                duration = time.time() - start
                metrics = {
                    "run_id": run_id,
                    "experiment_name": used_cfg.get("experiment_name"),
                    "duration_s": round(duration, 2),
                    "avg_train_loss": history.get("avg_train_loss"),
                    "avg_val_loss": history.get("avg_val_loss"),
                    "final_val_acc": history.get("val_acc")[-1] if history.get("val_acc") else None,
                    "n_epochs": len(history.get("train_loss", [])),
                    "config": json.dumps(combo),
                }
                writer.writerow(metrics)
                f.flush()
                print(f"Saved results for {run_id}")
            except Exception as e:
                print(f"Experiment {run_id} failed: {e}")
                writer.writerow({
                    "run_id": run_id,
                    "experiment_name": cfg.get("experiment_name"),
                    "duration_s": None,
                    "avg_train_loss": None,
                    "avg_val_loss": None,
                    "final_val_acc": None,
                    "n_epochs": 0,
                    "config": json.dumps(combo),
                })
                f.flush()


def run_wandb_sweep(sweep_yaml: str, base_cfg: Dict[str, Any], project: Optional[str] = None, entity: Optional[str] = None):
    if not _HAS_WANDB:
        raise RuntimeError("wandb is not installed in this environment. pip install wandb to use wandb sweeps.")
    with open(sweep_yaml, "r") as f:
        sweep_spec = f.read()
    sweep_id = wandb.sweep(sweep_spec, project=project, entity=entity)
    print(f"Created wandb sweep: {sweep_id}")

    # define the function to be called by wandb agent
    def _wandb_run():
        run = wandb.init()
        cfg = dict(base_cfg)
        # merge wandb.config into cfg
        for k, v in dict(wandb.config).items():
            # allow nested keys with dot notation
            if "." in k:
                parts = k.split(".")
                sub = cfg
                for p in parts[:-1]:
                    if p not in sub or not isinstance(sub[p], dict):
                        sub[p] = {}
                    sub = sub[p]
                sub[parts[-1]] = v
            else:
                cfg[k] = v
        cfg["experiment_name"] = cfg.get("experiment_name", run.name)
        try:
            model, used_cfg, history = run_experiment(cfg)
            # log final metrics to wandb
            wandb.log({"avg_train_loss": history.get("avg_train_loss"), "avg_val_loss": history.get("avg_val_loss"), "final_val_acc": history.get("val_acc")[-1] if history.get("val_acc") else None})
        finally:
            wandb.finish()

    print("Launching wandb agent (this will block and run until sweep complete or interrupted)...")
    wandb.agent(sweep_id, function=_wandb_run)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweeper for grid/random/wandb sweeps")
    parser.add_argument("--mode", choices=["grid", "wandb"], default="grid", help="Mode to run")
    parser.add_argument("--base-cfg", type=str, default="examples/speeddating_config.json", help="Path to base JSON config")
    parser.add_argument("--grid", type=str, default="examples/grid_params.json", help="Path to JSON grid params (for grid mode)")
    parser.add_argument("--out", type=str, default="runs/grid_results.csv", help="CSV to store grid results")
    parser.add_argument("--sweep", type=str, default="examples/wandb_sweep.yaml", help="WandB sweep YAML (for wandb mode)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main():
    args = parse_args()

    # configure logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("sweeper")
    logger.info("Starting sweeper mode=%s base_cfg=%s", args.mode, args.base_cfg)

    try:
        if not os.path.exists(args.base_cfg):
            logger.error("Base config not found: %s", args.base_cfg)
            sys.exit(2)

        with open(args.base_cfg, "r") as f:
            try:
                base_cfg = json.load(f)
            except Exception as e:
                logger.error("Failed to parse base config %s: %s", args.base_cfg, e)
                traceback.print_exc()
                sys.exit(2)

        logger.debug("Loaded base config keys: %s", list(base_cfg.keys()))

        if args.mode == "grid":
            if not os.path.exists(args.grid):
                logger.error("Grid file not found: %s", args.grid)
                sys.exit(2)
            with open(args.grid, "r") as f:
                try:
                    grid = json.load(f)
                except Exception as e:
                    logger.error("Failed to parse grid file %s: %s", args.grid, e)
                    traceback.print_exc()
                    sys.exit(2)

            if not grid:
                logger.warning("Grid is empty: %s", args.grid)
                return

            logger.info("Running local grid search: %d combinations (approx)", sum(len(v) for v in grid.values()))
            try:
                run_local_grid_search(base_cfg, grid, args.out)
            except Exception as e:
                logger.error("run_local_grid_search failed: %s", e)
                traceback.print_exc()
                sys.exit(1)

        elif args.mode == "wandb":
            project = base_cfg.get("wandb_project") or None
            entity = base_cfg.get("wandb_entity") or None
            try:
                run_wandb_sweep(args.sweep, base_cfg, project=project, entity=entity)
            except Exception as e:
                logger.error("run_wandb_sweep failed: %s", e)
                traceback.print_exc()
                sys.exit(1)

    except Exception as exc:
        logger.critical("Unhandled exception in sweeper main: %s", exc)
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    main()