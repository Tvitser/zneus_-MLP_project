"""
Experiment runner with optional Weights & Biases (wandb) integration.

- Logs config, per-epoch train/val loss & val_acc
- Logs avg_train_loss and avg_val_loss at the end
- Uploads artifacts (model.pt, loss/acc plots, history.json) to wandb if enabled
- Keeps all previous functionality: CSV loading, EDA, feature building, plotting, saving artifacts locally

Usage:
  # run without wandb
  python -m tools.experiment --config examples/speeddating_config.json

  # run with wandb (ensure wandb installed and logged in)
  python -m tools.experiment --config examples/speeddating_config.json

Config additions:
  "use_wandb": true,
  "wandb_project": "your-project",
  "wandb_entity": "your-entity-or-team",
  "wandb_run_name": "optional-run-name"
"""
import argparse
import json
import os
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from models.configurable_mlp import ConfigurableMLP
from tools.main import SpeedDatingEDA

# Try to import wandb; gracefully degrade if not available
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    wandb = None
    _HAS_WANDB = False


def default_config() -> Dict[str, Any]:
    return {
        "experiment_name": f"mlp_exp_{int(time.time())}",
        "input_dim": 20,
        "output_dim": 2,
        "hidden_dims": [256, 128, 64],
        "activation": "relu",
        "dropout": 0.2,
        "norm": "batch",
        "use_skip": False,
        "bottleneck": {"position": "middle", "factor": 4},
        "optimizer": {"type": "adam", "lr": 1e-3},
        "epochs": 10,
        "batch_size": 64,
        "seed": 42,
        "output_dir": "runs",
        "csv_path": None,
        "target_col": "match",
        "categorical_max_unique": 30,
        "show_plots": True,
        # wandb settings
        "use_wandb": False,
        "wandb_project": "mlp-experiments",
        "wandb_entity": None,
        "wandb_run_name": None,
        # optional three-way split sizes
        # "test_size": 0.1,
        # "val_size": 0.1,
    }


def build_model_from_cfg(cfg: Dict[str, Any]) -> torch.nn.Module:
    act_map = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "leakyrelu": nn.LeakyReLU,
    }
    activation = act_map.get(cfg.get("activation", "relu"), nn.ReLU)
    model = ConfigurableMLP(
        input_dim=int(cfg["input_dim"]),
        output_dim=int(cfg["output_dim"]),
        hidden_dims=cfg.get("hidden_dims"),
        activation=activation,
        dropout=cfg.get("dropout", 0.0),
        norm=cfg.get("norm", None),
        use_skip=cfg.get("use_skip", False),
        bottleneck=cfg.get("bottleneck", None),
        final_activation=None,
    )
    return model


# ---------------------------
# CSV dataset preparation
# ---------------------------
# python
def _prepare_X_y_and_split(df, cfg):
    import numpy as np
    from sklearn.model_selection import train_test_split

    # determine target column
    cfg_target = cfg.get("target_col", None)
    if cfg_target and cfg_target in df.columns:
        target_col = cfg_target
    elif "target" in df.columns:
        target_col = "target"
    elif cfg_target and cfg_target not in df.columns:
        print(f"Warning: configured target_col '{cfg_target}' not found in dataframe; falling back to 'target' if present.")
        target_col = "target" if "target" in df.columns else None
    else:
        target_col = None

    if target_col is None or target_col not in df.columns:
        raise ValueError(f"No target column found. looked for cfg['target_col']={cfg_target} or 'target'.")

    # build features and remove leakage columns
    X = df.drop(columns=[target_col], errors='ignore').copy()
    leakage_defaults = ["match", "decision", "decision_o", "target"]
    leakage_blocklist = cfg.get("leakage_blocklist", leakage_defaults)
    # ensure defaults present in blocklist
    for c in leakage_defaults:
        if c not in leakage_blocklist:
            leakage_blocklist.append(c)

    removed = []
    for leak in list(dict.fromkeys(leakage_blocklist)):  # preserve order, remove duplicates
        if leak in X.columns:
            X.drop(columns=[leak], inplace=True, errors='ignore')
            removed.append(leak)

    # also remove any column identical to the target (exact equality)
    try:
        for col in list(X.columns):
            try:
                if df[col].equals(df[target_col]):
                    X.drop(columns=[col], inplace=True)
                    removed.append(col)
            except Exception:
                continue
    except Exception:
        pass

    y = df[target_col].copy()
    # coerce boolean-like / numeric
    if not np.issubdtype(y.dtype, np.number):
        try:
            y = y.astype(int)
        except Exception:
            y = pd.to_numeric(y.astype(str).str.lower().replace({'yes':'1','no':'0','true':'1','false':'0'}), errors='coerce').fillna(0).astype(int)

    print(f"Target column used: {target_col}")
    print(f"Removed leakage columns from features: {removed}")
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

    # final safety
    if target_col in X.columns:
        X = X.drop(columns=[target_col], errors='ignore')

    test_size = float(cfg.get("test_size", 0.2))
    val_size = cfg.get("val_size", None)
    seed = int(cfg.get("seed", 42))

    stratify_param = y if (y.nunique() > 1 and len(y) >= 10) else None

    if val_size is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify_param
        )
        print(f"Train/Val shapes: {X_train.shape}, {X_val.shape}")
        return X_train, X_val, y_train, y_val
    else:
        # 3-way split: first split off test, then split train->val
        X_rest, X_test, y_rest, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify_param
        )
        stratify_rest = y_rest if (y_rest.nunique() > 1 and len(y_rest) >= 10) else None
        val_fraction_of_rest = float(val_size) / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest, test_size=val_fraction_of_rest, random_state=seed, shuffle=True, stratify=stratify_rest
        )
        print(f"Train/Val/Test shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")
        return X_train, X_val, y_train, y_val, X_test, y_test



def prepare_speeddating_data(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    csv_path = cfg.get("csv_path")
    if csv_path is None:
        raise ValueError("csv_path must be provided in config to load CSV data")

    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Raw shape: {df.shape}")

    eda = SpeedDatingEDA(df, auto_clean=True, verbose=False, show_plots=cfg.get("show_plots", False))

    if cfg.get("show_plots", False):
        try:
            eda.basic_info()
            eda.target_variable_analysis(cfg.get("target_col", "match"))
            eda.demographic_analysis()
            eda.correlation_analysis(cfg.get("target_col", "match"))
        except Exception as e:
            print("Error while plotting EDA graphs:", e)

    data = eda.data

    target_col = cfg.get("target_col", "match")
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in cleaned data")

    # Robustly coerce target to binary/integer if possible
    def _to_binary_series(s: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(s):
            return s.astype(int)
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors='coerce').fillna(0).astype(int)
        ss = s.astype(str).str.lower().str.strip()
        mapping = {'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1,
                   'no': 0, 'n': 0, 'false': 0, 'f': 0, '0': 0}
        ss_mapped = ss.replace(mapping)
        coerced = pd.to_numeric(ss_mapped, errors='coerce')
        # if many non-coercible values remain, fallback to boolean-like detection
        if coerced.notna().sum() / len(coerced) < 0.5:
            coerced = (ss.isin(['yes', 'y', 'true', 't', '1'])).astype(int)
        else:
            coerced = coerced.fillna(0).astype(int)
        return coerced

    try:
        data[target_col] = _to_binary_series(data[target_col])
    except Exception:
        # best-effort fallback: use pd.to_numeric then fillna with 0
        data[target_col] = pd.to_numeric(data[target_col], errors='coerce').fillna(0).astype(int)

    before = data.shape[0]
    data = data.dropna(subset=[target_col]).reset_index(drop=True)
    after = data.shape[0]
    if after < before:
        print(f"Dropped {before - after} rows with missing target '{target_col}'")

    y = data[target_col].astype(int).to_numpy()

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_to_use = []
    for c in cat_cols:
        nunique = data[c].nunique(dropna=True)
        if nunique <= cfg.get("categorical_max_unique", 30):
            cat_to_use.append(c)
        else:
            print(f"Skipping high-cardinality column '{c}' (unique={nunique})")

    X_df_parts = []
    if numeric_cols:
        X_df_parts.append(data[numeric_cols].copy())

    if cat_to_use:
        # convert to str then get_dummies (keeps numeric result)
        dummies = pd.get_dummies(data[cat_to_use].astype(str), dummy_na=True, drop_first=False)
        X_df_parts.append(dummies)

    if not X_df_parts:
        raise ValueError("No usable features found in dataset")

    X_df = pd.concat(X_df_parts, axis=1)

    # remove constant columns
    nunique = X_df.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"Removing constant feature columns: {const_cols}")
        X_df.drop(columns=const_cols, inplace=True)

    # Ensure all feature columns are numeric. Try pd.to_numeric, else factorize.
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            coerced = pd.to_numeric(X_df[col], errors='coerce')
            non_na_frac = coerced.notna().sum() / len(coerced) if len(coerced) > 0 else 0.0
            if non_na_frac >= 0.6:
                X_df[col] = coerced
            else:
                # fallback: factorize strings to integers (stable)
                vals, _ = pd.factorize(X_df[col].astype(str), sort=True)
                X_df[col] = vals.astype(float)

    # Fill remaining NaNs: numeric -> median, else 0
    for col in X_df.columns:
        if X_df[col].isna().any():
            try:
                X_df[col] = X_df[col].astype(float)
                median = np.nanmedian(X_df[col])
                if np.isnan(median):
                    median = 0.0
                X_df[col] = X_df[col].fillna(median)
            except Exception:
                X_df[col] = X_df[col].fillna(0.0)

    # final numeric array
    try:
        X = X_df.values.astype(float)
    except Exception as e:
        raise RuntimeError(f"Failed to convert feature dataframe to numeric array: {e}")

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    cfg["input_dim"] = X.shape[1]
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    cfg["output_dim"] = n_classes if n_classes > 2 else 2

    seed = int(cfg.get("seed", 42))
    test_size = cfg.get("test_size", None)
    val_size = cfg.get("val_size", None)

    if test_size is not None and val_size is not None:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=float(test_size), random_state=seed, stratify=y if n_classes > 1 else None
        )
        val_fraction = float(val_size) / (1.0 - float(test_size))
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_fraction, random_state=seed, stratify=y_trainval if n_classes > 1 else None
        )
        extras = {"scaler": scaler, "eda": eda, "report": {
            "n_samples": len(y),
            "n_features": X.shape[1],
            "n_classes": n_classes,
            "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        }, "test_set": (X_test, y_test)}
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y if n_classes > 1 else None
        )
        extras = {"scaler": scaler, "eda": eda, "report": {
            "n_samples": len(y),
            "n_features": X.shape[1],
            "n_classes": n_classes,
            "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        }}

    report = extras.get("report")
    print("Dataset prepared:", report)
    return X_train, y_train, X_val, y_val, extras


# ---------------------------
# Training loop with wandb logging
# ---------------------------
def run_training_loop(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Dict[str, Any],
    device: Optional[torch.device] = None,
):
    """
    Improved training loop with:
      - explicit index-based DataLoader to inspect indices per-epoch
      - deterministic per-epoch shuffling using torch.Generator.manual_seed(seed + epoch)
      - proper batch counting (uses all samples)
      - diagnostics to detect repeated identical index sequences across epochs
      - checks for degenerate features/labels or target leakage
    """
    import math
    from torch.utils.data import DataLoader, TensorDataset

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    epochs = int(cfg.get("epochs", 10))
    batch_size = int(cfg.get("batch_size", 64))
    opt_cfg = cfg.get("optimizer", {"type": "adam", "lr": 1e-3})
    lr = float(opt_cfg.get("lr", 1e-3))
    optim_type = opt_cfg.get("type", "adam").lower()
    if optim_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    criterion = nn.CrossEntropyLoss() if cfg.get("output_dim", 2) > 1 else nn.BCEWithLogitsLoss()

    # Basic dataset checks to detect degenerate data / leakage
    if X_train.size == 0 or X_val.size == 0:
        raise ValueError("Empty train or validation set provided to training loop")

    # Check feature variance (if all features are constant, training is meaningless)
    feat_std = np.nanstd(X_train, axis=0)
    if np.allclose(feat_std, 0.0):
        print("WARNING: All training features have zero variance. Check preprocessing / feature selection.")
    else:
        # warn about any completely-constant columns
        const_cols = np.where(np.isclose(feat_std, 0.0))[0].tolist()
        if const_cols:
            print(f"WARNING: Some feature columns are constant (indices): {const_cols}")

    # Check target diversity
    unique_labels, counts = np.unique(y_train, return_counts=True)
    if len(unique_labels) == 1:
        print(f"WARNING: Training labels are degenerate (single class = {unique_labels[0]}).")
    else:
        print(f"Training label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")

    # Convert numpy arrays to tensors (on CPU first)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long if cfg.get("output_dim", 2) > 1 else torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long if cfg.get("output_dim", 2) > 1 else torch.float32).to(device)

    num_train = X_train_t.shape[0]

    # We'll create an index DataLoader so we can inspect the order of indices used each epoch.
    index_dataset = torch.arange(num_train)
    # We'll create DataLoader inside the epoch loop with a per-epoch generator to avoid Tkinter/backend issues
    train_losses = []
    val_losses = []
    val_accs = []

    prev_perm = None
    perms_equal_count = 0

    use_wandb = bool(cfg.get("use_wandb", False)) and ('wandb' in globals() or 'wandb' in locals())
    if cfg.get("use_wandb", False) and not use_wandb:
        print("Warning: use_wandb=True but wandb is not available in this environment.")

    # training
    for epoch in range(1, epochs + 1):
        model.train()

        # deterministic but different per epoch: seed + epoch
        base_seed = int(cfg.get("seed", 42))
        gen = torch.Generator()
        gen.manual_seed(base_seed + epoch)

        idx_loader = DataLoader(index_dataset, batch_size=batch_size, shuffle=True, generator=gen, drop_last=False)

        epoch_indices_order = []  # collect indices order used in this epoch
        epoch_train_loss = 0.0
        n_batches = 0

        for batch_indices in idx_loader:
            # batch_indices is a 1D tensor of dataset indices
            idx = batch_indices.long()
            epoch_indices_order.extend(idx.tolist())

            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)

            logits = model(xb)
            if cfg.get("output_dim", 2) == 1:
                loss = criterion(logits.squeeze(-1), yb.float())
            else:
                loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += float(loss.item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No batches were produced (batch_size too large or dataset empty)")

        avg_epoch_train_loss = epoch_train_loss / n_batches
        train_losses.append(avg_epoch_train_loss)

        # detect permutation issues
        perm_array = np.array(epoch_indices_order, dtype=int)
        if prev_perm is not None:
            if perm_array.shape == prev_perm.shape and np.array_equal(perm_array, prev_perm):
                perms_equal_count += 1
                print(f"WARNING: epoch {epoch} permutation identical to previous epoch")
            else:
                # compute fraction of indices that are identical in same order
                same_in_order = np.sum(perm_array == prev_perm) / perm_array.size if perm_array.size == prev_perm.size else 0.0
                if same_in_order > 0.95:
                    print(f"WARNING: epoch {epoch} and previous epoch have very similar index order (same_in_order={same_in_order:.3f})")
        prev_perm = perm_array.copy()

        # sanity: ensure epoch used all (or most) training indices at least once
        unique_used = np.unique(perm_array)
        if unique_used.size < max(1, int(0.9 * num_train)):
            # if less than 90% of samples used in an epoch, warn (likely batch_size/errors)
            print(f"WARNING: epoch {epoch} used only {unique_used.size}/{num_train} unique training indices")

        # Validation (full val eval)
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            if cfg.get("output_dim", 2) == 1:
                val_loss = float(criterion(logits_val.squeeze(-1), y_val_t.float()).item())
                preds = (torch.sigmoid(logits_val.squeeze(-1)) > 0.5).long().cpu().numpy()
            else:
                val_loss = float(criterion(logits_val, y_val_t).item())
                preds = torch.argmax(logits_val, dim=-1).cpu().numpy()
            acc = (preds == y_val).mean()
        val_losses.append(val_loss)
        val_accs.append(float(acc))

        print(f"Epoch {epoch:03d} | train_loss={avg_epoch_train_loss:.6f} val_loss={val_loss:.6f} val_acc={acc:.4f}")

        # optionally log to wandb if enabled
        if use_wandb:
            try:
                wandb.log({"epoch": epoch, "train_loss": avg_epoch_train_loss, "val_loss": val_loss, "val_acc": acc})
            except Exception:
                pass

    # final diagnostics
    if perms_equal_count == epochs - 1:
        # identical perms across all epochs
        raise RuntimeError("Detected identical training index order across all epochs. This likely indicates that the RNG was reseeded each epoch with the same seed. Investigate seed usage.")

    avg_train_loss = float(np.mean(train_losses)) if train_losses else None
    avg_val_loss = float(np.mean(val_losses)) if val_losses else None

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_acc": val_accs,
        "avg_train_loss": avg_train_loss,
        "avg_val_loss": avg_val_loss,
    }

    return model, history


# ---------------------------
# Main runner
# ---------------------------
def run_experiment(cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or default_config()
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cfg.get("csv_path"):
        X_train, y_train, X_val, y_val, extras = prepare_speeddating_data(cfg)
    else:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=2000, n_features=int(cfg["input_dim"]), n_informative=max(2, int(cfg["input_dim"])//2), n_classes=int(cfg.get("output_dim",2)), random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y if len(np.unique(y))>1 else None)
        extras = {}

    cfg["input_dim"] = int(cfg.get("input_dim", X_train.shape[1]))
    cfg["output_dim"] = int(cfg.get("output_dim", cfg.get("output_dim", 2)))

    # If using wandb and the package is available, prefer to use wandb.config copy
    use_wandb = bool(cfg.get("use_wandb", False)) and _HAS_WANDB
    if cfg.get("use_wandb", False) and not _HAS_WANDB:
        print("Warning: use_wandb=True but wandb is not installed. Install it to enable W&B logging.")

    model = build_model_from_cfg(cfg)
    print(model)

    # If wandb is used, initialize a run at top-level too (so other modules can access)
    if use_wandb:
        run_name = cfg.get("wandb_run_name") or cfg.get("experiment_name")
        wandb.init(project=cfg.get("wandb_project"), entity=cfg.get("wandb_entity"), config=cfg, name=run_name, reinit=True)

    trained, history = run_training_loop(model, X_train, y_train, X_val, y_val, cfg)

    out_dir = os.path.join(cfg.get("output_dir", "runs"), cfg.get("experiment_name", "exp"))
    os.makedirs(out_dir, exist_ok=True)

    # Save artifacts locally
    cfg_path = os.path.join(out_dir, "config.json")
    model_path = os.path.join(out_dir, "model.pt")
    history_path = os.path.join(out_dir, "history.json")
    metrics_path = os.path.join(out_dir, "metrics.json")

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    torch.save(trained.state_dict(), model_path)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    metrics = {
        "avg_train_loss": history.get("avg_train_loss"),
        "avg_val_loss": history.get("avg_val_loss"),
        "final_val_acc": history.get("val_acc")[-1] if history.get("val_acc") else None,
        "n_epochs": len(history.get("train_loss", [])),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved config -> {cfg_path}")
    print(f"Saved model weights -> {model_path}")
    print(f"Saved history -> {history_path}")
    print(f"Saved metrics -> {metrics_path}")

    # If wandb enabled, log final artifacts and finish the run
    if use_wandb:
        try:
            artifact = wandb.Artifact(name=f"{cfg.get('experiment_name')}_run_artifacts", type="run")
            artifact.add_file(cfg_path)
            artifact.add_file(history_path)
            artifact.add_file(metrics_path)
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            wandb.finish()
            print("Logged artifacts to wandb and finished run.")
        except Exception as e:
            print("Failed to upload artifacts to wandb:", e)

    # Save additional heatmap if available
    if "eda" in extras and cfg.get("show_plots", True):
        try:
            eda = extras["eda"]
            numeric_cols = eda.data.select_dtypes(include=[np.number]).columns.tolist()
            if cfg.get("target_col") in numeric_cols:
                numeric_cols.remove(cfg.get("target_col"))
            if numeric_cols:
                corr = eda.data[numeric_cols].corr()
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr, cmap="coolwarm", center=0)
                heat_path = os.path.join(out_dir, "feature_correlation.png")
                plt.title("Feature Correlation")
                plt.tight_layout()
                plt.savefig(heat_path)
                print(f"Saved feature correlation heatmap -> {heat_path}")
                plt.show()
        except Exception as e:
            print("Failed to save additional heatmap:", e)

    return trained, cfg, history


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run configurable MLP experiments (wandb-enabled)")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--csv", type=str, help="Path to CSV file (overrides config)")
    parser.add_argument("--out", type=str, default=None, help="Override output_dir in config")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show-plots", action="store_true", help="Show and save plots during EDA and training")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)
    else:
        cfg = default_config()

    if args.csv:
        cfg["csv_path"] = args.csv
    if args.out:
        cfg["output_dir"] = args.out
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.show_plots:
        cfg["show_plots"] = True

    run_experiment(cfg)


if __name__ == "__main__":
    main()