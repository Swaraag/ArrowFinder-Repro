#!/usr/bin/env python3
"""
Train ONE Siamese ranker model with hyperparameters from a JSON config.

Required CLI args:
  --train_h5         Path to TRAIN HDF5 (expects pos_features, neg_features, targets)
  --val_h5           Path to VAL HDF5 (same keys as train)
  --out_dir          Output directory for model, logs, and plots
  --ranker_config    JSON file containing hyperparameters (see example below)

Example ranker_config.json:
{
  "num_layers": 3,
  "hidden_dim": 140,
  "dropout": 0.468,
  "activation": "relu",
  "lr": 0.01,
  "beta1": 0.895124,
  "beta2": 0.933391,
  "epsilon": 1e-8,
  "amsgrad": false,
  "epochs": 100,
  "batch_size": 2048,
  "patience": 50,
  "restore_best": true
}
"""

import os
import json
import time
import gc
import h5py
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import NonNeg

# --- plotting (headless) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------
# Repro & logging
# --------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass
np.random.seed(0)
tf.random.set_seed(0)

# ==========================
# Defaults (used if missing from JSON)
# ==========================
DEFAULT_HPARAMS = {
    # Architecture
    "num_layers": 3,
    "hidden_dim": 140,
    "dropout": 0.468,
    "activation": "relu",

    # Optimization
    "lr": 0.01,
    "beta1": 0.895124,
    "beta2": 0.933391,
    "epsilon": 1e-8,
    "amsgrad": False,

    # Training
    "epochs": 100,
    "batch_size": 2048,
    "patience": 50,
    "restore_best": True
}

# ==========================
# Data loading (no split)
# ==========================
def load_ranker_data(h5_path):
    with h5py.File(h5_path, "r", libver="latest") as f:
        X_pos = f["pos_features"][:].astype(np.float32)
        X_neg = f["neg_features"][:].astype(np.float32)
        y_raw = np.array(list(f["targets"][:]), dtype=np.float32)
    y = y_raw.reshape(-1)
    assert len(X_pos) == len(X_neg) == len(y), "Mismatch in dataset lengths"
    return X_pos, X_neg, y

# ==========================
# Plotting helper + callback
# ==========================
def _save_curve(y_tr, y_val, ylabel, title, out_png):
    plt.figure()
    plt.plot(y_tr, label="train")
    if any([not np.isnan(v) for v in y_val]):
        plt.plot(y_val, label="val")
        plt.legend()
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

class LiveHistoryPlotter(Callback):
    """
    Collects train/val loss & accuracy, updates history.json and
    saves loss.png / accuracy.png at the end of each epoch.
    """
    def __init__(self, hist_json_path, loss_png_path, acc_png_path):
        super().__init__()
        self.hist_json_path = hist_json_path
        self.loss_png_path = loss_png_path
        self.acc_png_path = acc_png_path
        self.history = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": []
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["loss"].append(float(logs.get("loss", np.nan)))
        self.history["val_loss"].append(float(logs.get("val_loss", np.nan)))
        self.history["accuracy"].append(float(logs.get("accuracy", np.nan)))
        self.history["val_accuracy"].append(float(logs.get("val_accuracy", np.nan)))

        try:
            with open(self.hist_json_path, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"[warn] Could not write history.json: {e}", flush=True)

        try:
            _save_curve(self.history["loss"], self.history["val_loss"], "loss", "Loss", self.loss_png_path)
            _save_curve(self.history["accuracy"], self.history["val_accuracy"], "accuracy", "Accuracy", self.acc_png_path)
        except Exception as e:
            print(f"[warn] Could not save plots: {e}", flush=True)

# ==========================
# Model builder (Siamese)
# ==========================
def build_siamese(input_dim, num_layers, hidden_dim, dropout_rate, activation):
    shared = Sequential(name="shared_network")
    for i in range(num_layers):
        if i == 0:
            shared.add(Dense(hidden_dim, activation=activation, input_shape=(input_dim,)))
        else:
            shared.add(Dense(hidden_dim, activation=activation))
        if dropout_rate and dropout_rate > 0:
            shared.add(Dropout(dropout_rate))
    # Linear head → score
    shared.add(Dense(1, activation="linear"))

    inp_l = Input(shape=(input_dim,), name="input_l")
    inp_r = Input(shape=(input_dim,), name="input_r")
    s_l = shared(inp_l)
    s_r = shared(inp_r)

    diff = tf.keras.layers.Subtract(name="diff")([s_l, s_r])  # (left - right)
    out = Dense(1, activation="sigmoid", name="pred")(diff)

    return Model(inputs=[inp_l, inp_r], outputs=out, name="siamese_ranker")

# ==========================
# HPARAMS loader
# ==========================
def load_hparams(cfg_path):
    with open(cfg_path, "r") as f:
        user = json.load(f)
    hp = {**DEFAULT_HPARAMS, **user}  # user overrides defaults
    # minimal validation
    assert hp["num_layers"] >= 1, "num_layers must be >= 1"
    assert hp["hidden_dim"] >= 1, "hidden_dim must be >= 1"
    assert 0.0 <= float(hp["dropout"]) < 1.0, "dropout must be in [0,1)"
    assert hp["activation"] in ("relu", "tanh"), "activation must be 'relu' or 'tanh'"
    assert hp["epochs"] >= 1, "epochs must be >= 1"
    assert hp["batch_size"] >= 1, "batch_size must be >= 1"
    return hp

# ==========================
# Train one model
# ==========================
def train_one(train_h5, val_h5, out_dir, hparams):
    # Resolve output paths
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, "polar_ranker_single_best.h5")
    run_log_json    = os.path.join(out_dir, "single_run_log.json")
    hist_json_path  = os.path.join(out_dir, "history.json")
    loss_png_path   = os.path.join(out_dir, "loss.png")
    acc_png_path    = os.path.join(out_dir, "accuracy.png")

    # Load data
    X_pos_tr, X_neg_tr, y_tr = load_ranker_data(train_h5)
    X_pos_v,  X_neg_v,  y_v  = load_ranker_data(val_h5)

    # Sanity checks
    assert X_pos_tr.shape[1] == X_neg_tr.shape[1], "Train: pos/neg feature dims differ"
    assert X_pos_v.shape[1]  == X_neg_v.shape[1],  "Val: pos/neg feature dims differ"
    assert X_pos_tr.shape[1] == X_pos_v.shape[1],  "Train/Val feature dims differ"

    N_tr, D = X_pos_tr.shape
    N_v  = X_pos_v.shape[0]
    print(f"Loaded TRAIN: N={N_tr}, D={D}")
    print(f"Loaded VAL:   N={N_v},  D={D}")

    # Build model
    K.clear_session()
    gc.collect()

    model = build_siamese(
        input_dim=D,
        num_layers=hparams["num_layers"],
        hidden_dim=hparams["hidden_dim"],
        dropout_rate=hparams["dropout"],
        activation=hparams["activation"],
    )

    opt = Adam(
        learning_rate=hparams["lr"],
        beta_1=hparams["beta1"],
        beta_2=hparams["beta2"],
        epsilon=hparams["epsilon"],
        amsgrad=bool(hparams["amsgrad"]),
    )
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    # Callbacks
    es = EarlyStopping(
        monitor="val_loss",
        patience=hparams["patience"],
        mode="min",
        restore_best_weights=bool(hparams["restore_best"]),
        verbose=1,
    )
    ckpt = ModelCheckpoint(
        best_model_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    live = LiveHistoryPlotter(hist_json_path, loss_png_path, acc_png_path)

    # Train
    t0 = time.time()
    hist = model.fit(
        [X_pos_tr, X_neg_tr], y_tr,
        validation_data=([X_pos_v, X_neg_v], y_v),
        epochs=hparams["epochs"],
        batch_size=hparams["batch_size"],
        shuffle=True,
        callbacks=[live, es, ckpt],
        verbose=1,
    )
    seconds = round(time.time() - t0, 2)

    # Metrics from history
    val_loss_hist = hist.history.get("val_loss", [])
    best_epoch = int(np.argmin(val_loss_hist)) if len(val_loss_hist) else -1
    best_val_loss = float(np.min(val_loss_hist)) if len(val_loss_hist) else float("nan")

    val_acc_hist = hist.history.get("val_accuracy", [])
    best_val_acc = (float(val_acc_hist[best_epoch])
                    if len(val_acc_hist) and best_epoch >= 0 else float("nan"))

    train_loss_hist = hist.history.get("loss", [])
    last_train_loss = float(train_loss_hist[-1]) if len(train_loss_hist) else float("nan")

    # Save run log
    run_log = {
        "paths": {
            "train_h5": train_h5,
            "val_h5": val_h5,
            "best_model_path": best_model_path,
            "history_json": hist_json_path,
            "loss_png": loss_png_path,
            "accuracy_png": acc_png_path,
        },
        "hparams": {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in hparams.items()},
        "metrics": {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_accuracy_at_best_epoch": best_val_acc,
            "last_train_loss": last_train_loss,
            "seconds": seconds
        }
    }
    try:
        with open(run_log_json, "w") as f:
            json.dump(run_log, f, indent=2)
    except Exception as e:
        print("Warning: could not write run log JSON:", e, flush=True)

    # Final summary
    print(
        f"\nDone. best_val_loss={best_val_loss:.6f} "
        f"(val_acc@best={best_val_acc:.4f})  [best_epoch={best_epoch}]  [{seconds} s]"
    )
    print("\nSaved best model to:", best_model_path)
    print("Run log JSON:", run_log_json)
    print("Live plots:", loss_png_path, acc_png_path)
    print("History JSON:", hist_json_path)

    # Cleanup
    del model
    K.clear_session()
    gc.collect()

# ==========================
# CLI
# ==========================
def parse_args():
    p = argparse.ArgumentParser(description="Train a Siamese ranker with JSON hyperparameters.")
    p.add_argument("--train_h5", required=True, help="Path to TRAIN HDF5 file")
    p.add_argument("--val_h5", required=True, help="Path to VAL HDF5 file")
    p.add_argument("--out_dir", required=True, help="Output directory for artifacts")
    p.add_argument("--ranker_config", required=True, help="Path to JSON file of hyperparameters")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    hparams = load_hparams(args.ranker_config)
    train_one(args.train_h5, args.val_h5, args.out_dir, hparams)
