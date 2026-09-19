from __future__ import annotations
import os, sys, time, random, argparse, logging, shutil
import numpy as np
from datetime import datetime
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("train")

layers = keras.layers

_orig_input = layers.InputLayer.__init__
def _p_input(self, *a, **kw):
    kw.pop("optional", None)
    if "batch_shape" in kw: kw["batch_input_shape"] = kw.pop("batch_shape")
    _orig_input(self, *a, **kw)
layers.InputLayer.__init__ = _p_input

_orig_dense = layers.Dense.__init__
def _p_dense(self, *a, **kw):
    kw.pop("quantization_config", None)
    _orig_dense(self, *a, **kw)
layers.Dense.__init__ = _p_dense


class TrainingProgressCallback(keras.callbacks.Callback):
    BAR_LEN = 25

    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self._ep_start    = 0.0
        self._train_start = 0.0

    def on_train_begin(self, logs=None):
        self._train_start = time.time()
        print(f"\n{'='*65}")
        print(f"  Training bat dau | {self.total_epochs} epochs")
        print(f"{'='*65}")

    def on_epoch_begin(self, epoch, logs=None):
        self._ep_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        ep_time = time.time() - self._ep_start
        elapsed = time.time() - self._train_start
        done    = epoch + 1
        pct     = done / self.total_epochs

        eta_s   = (elapsed / done) * (self.total_epochs - done) if done > 0 else 0
        eta_str = f"{int(eta_s//60):02d}:{int(eta_s%60):02d}"

        filled = int(self.BAR_LEN * pct)
        bar    = "#" * filled + "-" * (self.BAR_LEN - filled)

        loss    = logs.get("loss", 0)
        acc     = logs.get("accuracy", 0) * 100
        v_loss  = logs.get("val_loss", 0)
        v_acc   = logs.get("val_accuracy", 0) * 100
        lr      = float(keras.backend.get_value(self.model.optimizer.learning_rate))

        star = "***" if v_acc >= 90 else ("**" if v_acc >= 80 else ("*" if v_acc >= 70 else ""))

        print(
            f"\n  Epoch {done:>3}/{self.total_epochs} [{bar}] {pct*100:5.1f}% {star}"
            f"\n  Loss={loss:.4f}  Acc={acc:5.1f}%  |"
            f"  Val_Loss={v_loss:.4f}  Val_Acc={v_acc:5.1f}%"
            f"\n  LR={lr:.2e}  Epoch={ep_time:.1f}s  ETA={eta_str}"
        )

        if done == self.total_epochs:
            total_m = (time.time() - self._train_start) / 60
            print(f"\n{'='*65}")
            print(f"  Training xong! Tong thoi gian: {total_m:.1f} phut")
            print(f"{'='*65}\n")

    def on_train_batch_end(self, batch, logs=None):
        logs    = logs or {}
        total_b = self.params.get("steps", 1)
        pct     = (batch + 1) / total_b
        filled  = int(20 * pct)
        bar     = "#" * filled + "-" * (20 - filled)
        acc     = logs.get("accuracy", 0) * 100
        loss    = logs.get("loss", 0)
        print(f"\r  [{bar}] {pct*100:5.1f}%  loss={loss:.4f}  acc={acc:.1f}%",
              end="", flush=True)


def build_feature_extractor(input_size: int = 96) -> keras.Model:
    model = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(input_size, input_size, 3), pooling="avg",
    )
    model.trainable = False
    log.info(f"Feature extractor: MobileNetV2 ({input_size}x{input_size}) | frozen")
    return model


def build_sequence_model(feature_dim: int = 1280,
                          seq_len: int = 10) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(seq_len, feature_dim)),
        layers.LSTM(64, return_sequences=False,
                    kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(2, activation="softmax"),
    ], name="violence_seq_model")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    log.info(f"Sequence model: LSTM(64) | seq_len={seq_len} | params={model.count_params():,}")
    return model


def make_callbacks(model_path: str, patience: int = 8,
                   total_epochs: int = 40) -> list:
    return [
        TrainingProgressCallback(total_epochs),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy", save_best_only=True, mode="max", verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.CSVLogger("training_log.csv"),
    ]


def compute_class_weights(y: np.ndarray) -> dict:
    unique, counts = np.unique(y, return_counts=True)
    total = len(y); n_cls = len(unique)
    weights = {}
    for cls, cnt in zip(unique, counts):
        weights[int(cls)] = total / (n_cls * cnt)
    log.info(f"Class weights: {weights}")
    return weights


def plot_history(history, out: str = "training_history.png"):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, key, title in [
            (axes[0], "loss",     "Loss"),
            (axes[1], "accuracy", "Accuracy"),
        ]:
            ax.plot(history.history[key],       label=f"Train {title}", lw=2)
            ax.plot(history.history[f"val_{key}"], label=f"Val {title}", lw=2, ls="--")
            if key == "accuracy":
                ax.axhline(0.9, color="red", ls=":", label="90% target")
                ax.set_ylim([0, 1])
            ax.set_title(title, fontsize=13); ax.legend(); ax.grid(True, alpha=0.3)
        plt.suptitle("Violence Detection — Training History", fontsize=14, fontweight="bold")
        plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"Chart: {out}")
    except ImportError:
        log.warning("matplotlib không có, bỏ qua vẽ biểu đồ")


def plot_confusion_matrix(y_true, y_pred, out: str = "confusion_matrix.png"):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = ["NonViolence", "Violence"]
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, y_pred): cm[t][p] += 1
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(labels, fontsize=11); ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix\nAcc={np.diag(cm).sum()/cm.sum()*100:.1f}%", fontsize=12)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if cm[i][j] > cm.max()/2 else "black")
        plt.colorbar(im, ax=ax); plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"Confusion matrix: {out}")
    except ImportError:
        pass


def compute_metrics(y_true, y_pred) -> dict:
    tp = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
    tn = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==0)
    fp = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
    fn = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
    total = len(y_true)
    acc  = (tp+tn)/total if total else 0
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec  = tp/(tp+fn) if (tp+fn) else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)


def save_report(metrics, history, args, out: str = "training_report.txt"):
    best_acc = max(history.history.get("val_accuracy", [0]))
    best_ep  = history.history.get("val_accuracy", [0]).index(best_acc) + 1
    total_ep = len(history.history.get("val_accuracy", []))
    lines = [
        "=" * 60,
        " VIOLENCE DETECTION — TRAINING REPORT",
        "=" * 60,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "[ Config ]",
        f"  Data folder   : {args.data}",
        f"  Epochs        : {total_ep} / {args.epochs}",
        f"  Batch size    : {args.batch}",
        f"  Seq length    : {args.seq_len}",
        f"  Stride        : {args.stride}",
        f"  Input size    : {args.input_size}px",
        "",
        "[ Best Model ]",
        f"  Val Accuracy  : {best_acc*100:.2f}%  (epoch {best_ep})",
        f"  Saved to      : {args.output}",
        "",
        "[ Validation Metrics ]",
        f"  Accuracy  : {metrics['accuracy']*100:.2f}%",
        f"  Precision : {metrics['precision']*100:.2f}%",
        f"  Recall    : {metrics['recall']*100:.2f}%",
        f"  F1 Score  : {metrics['f1']*100:.2f}%",
        "",
        "[ Confusion Matrix ]",
        f"  TP (Violence đúng)  : {metrics['tp']}",
        f"  TN (Normal đúng)    : {metrics['tn']}",
        f"  FP (Nhầm Violence)  : {metrics['fp']}",
        f"  FN (Bỏ sót Violence): {metrics['fn']}",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(out, "w", encoding="utf-8") as f: f.write(report)


def main():
    p = argparse.ArgumentParser(description="Violence Detection Trainer")
    p.add_argument("--data",        default=".",    help="Thư mục chứa violence/ và nonviolence/")
    p.add_argument("--output",      default="violence_detection_model.h5")
    p.add_argument("--epochs",      type=int,   default=40)
    p.add_argument("--batch",       type=int,   default=16)
    p.add_argument("--seq-len",     type=int,   default=10,  dest="seq_len")
    p.add_argument("--stride",      type=int,   default=5,   help="Sliding window stride")
    p.add_argument("--input-size",  type=int,   default=96,  dest="input_size")
    p.add_argument("--val-ratio",   type=float, default=0.2, dest="val_ratio")
    p.add_argument("--patience",    type=int,   default=8)
    p.add_argument("--resume",      action="store_true", help="Tiếp tục từ checkpoint")
    p.add_argument("--no-cache",    action="store_true", dest="no_cache",
                   help="Xóa cache cũ, build lại từ đầu")
    p.add_argument("--seed",        type=int,   default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)

    log.info("=" * 60)
    log.info(" VIOLENCE DETECTION TRAINING ★ VIP PRO ★")
    log.info("=" * 60)
    log.info(f"TF {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU')}")
    log.info(f"Config: epochs={args.epochs} batch={args.batch} "
             f"seq={args.seq_len} stride={args.stride} input={args.input_size}px")

    if args.no_cache and Path("feature_cache").exists():
        shutil.rmtree("feature_cache")
        log.info("Đã xóa feature cache")

    log.info("\n[1/6] Quét dataset...")
    from data_loader import scan_dataset, train_val_split, build_dataset

    samples, _ = scan_dataset(args.data)
    train_samples, val_samples = train_val_split(samples, args.val_ratio)

    log.info("\n[2/6] Load MobileNetV2...")
    fe = build_feature_extractor(args.input_size)
    feat_dim = fe.output_shape[-1]

    cache_note = "(cache)" if not args.no_cache else "(no cache)"
    log.info(f"\n[3/6] Extract + cache features {cache_note}...")
    log.info("  Lần đầu: ~5-15 phút | Lần sau: < 1 phút (từ cache)")

    t0 = time.time()
    X_train, y_train = build_dataset(
        train_samples, fe,
        seq_len=args.seq_len, stride=args.stride,
        input_size=args.input_size, training=True,
        use_cache=not args.no_cache, desc="Train",
    )
    X_val, y_val = build_dataset(
        val_samples, fe,
        seq_len=args.seq_len, stride=args.stride,
        input_size=args.input_size, training=False,
        use_cache=not args.no_cache, desc="Val",
    )
    log.info(f"  Xong! Thời gian extract: {(time.time()-t0)/60:.1f} phút")
    log.info(f"  X_train: {X_train.shape} | X_val: {X_val.shape}")

    log.info("\n[4/6] Class weights...")
    cw = compute_class_weights(y_train)

    log.info(f"\n[5/6] Training LSTM model...")

    if args.resume and Path(args.output).exists():
        log.info(f"  Resume từ: {args.output}")
        seq_model = keras.models.load_model(args.output)
    else:
        seq_model = build_sequence_model(feat_dim, args.seq_len)

    history = seq_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=cw,
        callbacks=make_callbacks(args.output, args.patience, args.epochs),
        shuffle=True,
        verbose=0,
    )

    log.info("\n[6/6] Đánh giá...")
    y_pred = np.argmax(seq_model.predict(X_val, batch_size=args.batch, verbose=0), axis=1)
    metrics = compute_metrics(y_val, y_pred)

    plot_history(history)
    plot_confusion_matrix(y_val, y_pred)
    save_report(metrics, history, args)

    log.info(f"\n✅ Model: {args.output}")
    log.info(f"✅ Accuracy: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1']*100:.2f}%")

    if metrics["accuracy"] < 0.75:
        log.warning(
            "\n⚠️  Accuracy thấp — Thử:\n"
            "  1. Giảm stride để có nhiều sequences hơn: --stride 2\n"
            "  2. Thêm video vào dataset\n"
            "  3. Tăng epochs: --epochs 60\n"
            "  4. Kiểm tra video bị nhầm folder"
        )


if __name__ == "__main__":
    main()
