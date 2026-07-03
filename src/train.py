from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import datetime
import json
import logging
from typing import Optional, Tuple, Dict, List, Any

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm

from src.config import Config
from src.model import IMRModel, IMRLoss  # IMR architecture.
from src.dataset import load_all_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation helpers.
# ---------------------------------------------------------------------------


def _evaluate_iece_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    threshold: Optional[float] = None,
    tune_threshold: bool = Config.DL_TUNE_DECISION_THRESHOLD,
) -> Dict[str, float]:
    """Collect IECE cause extraction metrics in one pass, optionally with val_loss."""
    model.eval()

    cause_probs: List[float] = []
    cause_labels_all: List[int] = []
    total_val_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            text_emb = batch["text_emb"].to(device)
            concat_emb = batch["concat_emb"].to(device)
            event_emb = batch["event_emb"].to(device)
            emotion_labels = batch["emotion_label"].to(device)
            cause_labels = batch["cause_label"].to(device)
            center_labels = batch["center_label"].to(device)
            event_features = batch["event_features"].to(device)

            text_padding_mask = batch.get("key_padding_mask", None)
            if text_padding_mask is not None:
                text_padding_mask = text_padding_mask.to(device)

            event_padding_mask = batch.get("event_padding_mask", None)
            if event_padding_mask is not None:
                event_padding_mask = event_padding_mask.to(device)

            concat_padding_mask = batch.get("concat_padding_mask", None)
            if concat_padding_mask is not None:
                concat_padding_mask = concat_padding_mask.to(device)

            cause_input_emb = (
                concat_emb if Config.DL_CAUSE_INPUT_MODE == "concat" else event_emb
            )
            cause_padding_mask = (
                concat_padding_mask
                if Config.DL_CAUSE_INPUT_MODE == "concat"
                else event_padding_mask
            )

            emotion_logits, cause_logits, center_logits = model(
                text_emb=text_emb,
                cause_input_emb=cause_input_emb,
                text_padding_mask=text_padding_mask,
                cause_padding_mask=cause_padding_mask,
                concat_input_emb=concat_emb,
                concat_padding_mask=concat_padding_mask,
                event_features=event_features,
                return_auxiliary=True,
            )

            if loss_fn is not None:
                loss = loss_fn(
                    emotion_logits=emotion_logits,
                    cause_logits=cause_logits,
                    emotion_labels=emotion_labels,
                    cause_labels=cause_labels,
                    center_logits=center_logits,
                    center_labels=center_labels,
                )
                total_val_loss += float(loss.item())
                n_batches += 1

            cause_prob = torch.softmax(cause_logits, dim=-1)[:, 1].cpu()
            cause_labels_cpu = cause_labels.cpu()

            cause_probs.extend(cause_prob.tolist())
            cause_labels_all.extend(cause_labels_cpu.tolist())

    selected_threshold = 0.5 if threshold is None else float(threshold)
    if threshold is None and tune_threshold:
        selected_threshold = _find_best_threshold(cause_probs, cause_labels_all)
    cause_preds = [int(prob >= selected_threshold) for prob in cause_probs]

    precision = float(
        precision_score(cause_labels_all, cause_preds, pos_label=1, zero_division=0)
    )
    recall = float(
        recall_score(cause_labels_all, cause_preds, pos_label=1, zero_division=0)
    )
    f1 = float(f1_score(cause_labels_all, cause_preds, pos_label=1, zero_division=0))

    return {
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "val_loss": total_val_loss / max(n_batches, 1) if n_batches > 0 else 0.0,
        "val_threshold": selected_threshold,
    }


def _find_best_threshold(probs: List[float], labels: List[int]) -> float:
    """Pick the threshold with the highest positive-class F1 on validation data."""
    if not probs:
        return 0.5

    best_threshold = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(
        Config.DL_THRESHOLD_MIN,
        Config.DL_THRESHOLD_MAX,
        Config.DL_THRESHOLD_STEPS,
    )
    for threshold in thresholds:
        preds = [int(prob >= threshold) for prob in probs]
        f1 = float(f1_score(labels, preds, pos_label=1, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def _document_ranking_loss(
    cause_logits: torch.Tensor,
    cause_labels: torch.Tensor,
    doc_ids: List[str],
) -> torch.Tensor:
    """Rank cause events above non-cause events from the same document."""
    if float(Config.LOSS_WEIGHT_RANKING) <= 0:
        return cause_logits.new_zeros(())

    scores = cause_logits[:, 1] - cause_logits[:, 0]
    losses: List[torch.Tensor] = []
    for doc_id in sorted(set(doc_ids)):
        indices = [
            idx
            for idx, item_doc_id in enumerate(doc_ids)
            if item_doc_id == doc_id
        ]
        if len(indices) < 2:
            continue
        doc_indices = torch.as_tensor(indices, device=cause_logits.device)
        doc_labels = cause_labels.index_select(0, doc_indices)
        doc_scores = scores.index_select(0, doc_indices)
        pos_scores = doc_scores[doc_labels == 1]
        neg_scores = doc_scores[doc_labels == 0]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue
        pairwise_loss = torch.relu(
            float(Config.CAUSE_RANKING_MARGIN)
            - (pos_scores[:, None] - neg_scores[None, :])
        )
        losses.append(pairwise_loss.mean())

    if not losses:
        return cause_logits.new_zeros(())
    return torch.stack(losses).mean()


def _get_cause_class_weights(
    train_loader: DataLoader,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Compute inverse-frequency cause class weights from the current training fold."""
    mode = Config.CAUSE_CLASS_WEIGHT_MODE.lower().strip()
    if mode in {"", "none", "off", "false"}:
        return None
    if mode != "balanced":
        raise ValueError(
            f"Invalid CAUSE_CLASS_WEIGHT_MODE='{Config.CAUSE_CLASS_WEIGHT_MODE}'. "
            "Expected 'none' or 'balanced'."
        )

    dataset = train_loader.dataset
    pairs = getattr(dataset, "pairs", None)
    if pairs is None:
        logger.warning("Could not inspect training labels; cause class weights disabled.")
        return None

    counts = [0, 0]
    for pair in pairs:
        counts[int(pair.cause_label)] += 1
    if min(counts) == 0:
        logger.warning(f"Degenerate cause-label distribution {counts}; class weights disabled.")
        return None

    total = float(sum(counts))
    weights = torch.tensor(
        [total / (2.0 * counts[0]), total / (2.0 * counts[1])],
        dtype=torch.float32,
        device=device,
    )
    logger.info(
        "Cause class weights enabled: "
        f"non_cause={weights[0].item():.4f}, cause={weights[1].item():.4f} "
        f"from counts={counts}"
    )
    return weights


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluate Precision, Recall, and F1 on the validation set for the positive
    Y=1 class.
    Returns:
        (precision, recall, f1)
    """
    metrics = _evaluate_iece_metrics(model, dataloader, device, loss_fn=None)
    return metrics["val_precision"], metrics["val_recall"], metrics["val_f1"]


def configure_optimizer(
    model: nn.Module,
    loss_fn: nn.Module,
    learning_rate: float,
) -> AdamW:
    """Build an optimizer with branch-specific parameter groups."""
    lr_dict = Config.get_lr_dict(learning_rate)
    emotion_params: List[nn.Parameter] = []
    cause_params: List[nn.Parameter] = []
    shared_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(
            ("emotion_proj", "emotion_encoder", "emotion_attn_pool", "emotion_head")
        ):
            emotion_params.append(param)
        elif name.startswith("cause_"):
            cause_params.append(param)
        else:
            shared_params.append(param)

    shared_params.extend(list(loss_fn.parameters()))

    param_groups = []
    if shared_params:
        param_groups.append(
            {
                "params": shared_params,
                "lr": lr_dict["shared"],
                "initial_lr": lr_dict["shared"],
                "name": "shared",
            }
        )
    if emotion_params:
        param_groups.append(
            {
                "params": emotion_params,
                "lr": lr_dict["emotion"],
                "initial_lr": lr_dict["emotion"],
                "name": "emotion",
            }
        )
    if cause_params:
        param_groups.append(
            {
                "params": cause_params,
                "lr": lr_dict["cause"],
                "initial_lr": lr_dict["cause"],
                "name": "cause",
            }
        )

    logger.info(
        "Optimizer param groups: "
        f"shared={len(shared_params)}, emotion={len(emotion_params)}, cause={len(cause_params)} | "
        f"lr={lr_dict}"
    )

    return AdamW(param_groups, weight_decay=Config.DL_WEIGHT_DECAY)


# ---------------------------------------------------------------------------
# Training metrics logger.
# ---------------------------------------------------------------------------


class MetricsLogger:
    """
    Append per-epoch training metrics to a JSON Lines log file.

    Each line is an independent JSON object with timestamp, fold, phase, epoch,
    and metric fields, making it easy to load later with tools such as pandas.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Write an initial metadata row.
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "_comment": "IMR training metrics log",
                        "created": datetime.datetime.now().isoformat(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        logger.info(f"[MetricsLogger] Log file created: {log_path}")

    def log(self, record: dict) -> None:
        """Append one metric record and add a timestamp."""
        record = {"timestamp": datetime.datetime.now().isoformat(), **record}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Supervised training.
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    patience: int = Config.DL_EARLY_STOP_PATIENCE,
    metrics_logger: Optional["MetricsLogger"] = None,
    fold_idx: int = 0,
) -> Tuple[Dict, float, Dict[str, Any]]:
    """
    End-to-end supervised training.

    Jointly optimizes ER and IECE cause extraction losses and uses cause F1 for
    early stopping.

    Returns:
        (best_model_state, best_f1, best_metrics)
    """
    logger.info(
        f"[Supervised Training] Starting for up to {epochs} epochs "
        f"with early-stopping patience {patience}"
    )

    cause_class_weights = _get_cause_class_weights(train_loader, device)
    loss_fn = IMRLoss(
        label_smoothing=Config.DL_LABEL_SMOOTHING,
        cause_class_weights=cause_class_weights,
    )
    loss_fn = loss_fn.to(device)

    optimizer = configure_optimizer(
        model=model,
        loss_fn=loss_fn,
        learning_rate=learning_rate,
    )
    # LR schedule: manual linear warmup for the first DL_WARMUP_EPOCHS epochs,
    # then ReduceLROnPlateau adapts LR only when validation F1 stalls.
    warmup_epochs = Config.DL_WARMUP_EPOCHS
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=Config.DL_LR_PLATEAU_PATIENCE,
        min_lr=1e-6,
    )

    best_f1 = 0.0
    best_state: Dict = {k: v.cpu() for k, v in model.state_dict().items()}
    best_metrics: Dict[str, Any] = {
        "best_epoch": 0,
        "val_f1": 0.0,
        "val_precision": 0.0,
        "val_recall": 0.0,
        "val_loss": 0.0,
        "train_loss": 0.0,
        "val_threshold": 0.5,
    }
    epochs_no_improve = 0

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            scale = (epoch + 1) / max(warmup_epochs, 1)
            for group in optimizer.param_groups:
                group["lr"] = group.get("initial_lr", learning_rate) * scale

        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            text_emb = batch["text_emb"].to(device)
            concat_emb = batch["concat_emb"].to(device)
            event_emb = batch["event_emb"].to(device)
            emotion_labels = batch["emotion_label"].to(device)
            cause_labels = batch["cause_label"].to(device)
            center_labels = batch["center_label"].to(device)
            event_features = batch["event_features"].to(device)

            # Read padding masks.
            text_padding_mask = batch.get("key_padding_mask", None)
            if text_padding_mask is not None:
                text_padding_mask = text_padding_mask.to(device)

            event_padding_mask = batch.get("event_padding_mask", None)
            if event_padding_mask is not None:
                event_padding_mask = event_padding_mask.to(device)

            concat_padding_mask = batch.get("concat_padding_mask", None)
            if concat_padding_mask is not None:
                concat_padding_mask = concat_padding_mask.to(device)

            # Select the cause input according to configuration.
            cause_input_emb = (
                concat_emb if Config.DL_CAUSE_INPUT_MODE == "concat" else event_emb
            )
            cause_padding_mask = (
                concat_padding_mask
                if Config.DL_CAUSE_INPUT_MODE == "concat"
                else event_padding_mask
            )

            emotion_logits, cause_logits, center_logits = model(
                text_emb=text_emb,
                cause_input_emb=cause_input_emb,
                text_padding_mask=text_padding_mask,
                cause_padding_mask=cause_padding_mask,
                concat_input_emb=concat_emb,
                concat_padding_mask=concat_padding_mask,
                event_features=event_features,
                return_auxiliary=True,
            )

            loss = loss_fn(
                emotion_logits=emotion_logits,
                cause_logits=cause_logits,
                emotion_labels=emotion_labels,
                cause_labels=cause_labels,
                center_logits=center_logits,
                center_labels=center_labels,
            )
            ranking_loss = _document_ranking_loss(
                cause_logits=cause_logits,
                cause_labels=cause_labels,
                doc_ids=list(batch["doc_id"]),
            )
            loss = loss + float(Config.LOSS_WEIGHT_RANKING) * ranking_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        eval_metrics = _evaluate_iece_metrics(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=loss_fn,
        )
        precision = eval_metrics["val_precision"]
        recall = eval_metrics["val_recall"]
        f1 = eval_metrics["val_f1"]
        avg_val_loss = eval_metrics["val_loss"]
        threshold = eval_metrics["val_threshold"]

        if recall >= 0.99 and precision < 0.95:
            logger.warning(
                f"  [Degenerate solution] Detected all-positive behavior: "
                f"recall={recall:.4f}, precision={precision:.4f}"
            )

        logger.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"IECE P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f} | "
            f"Threshold: {threshold:.3f}"
        )
        if metrics_logger is not None:
            metrics_logger.log(
                {
                    "fold": fold_idx + 1,
                    "phase": "supervised",
                    "epoch": epoch + 1,
                    "loss": avg_train_loss,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_precision": precision,
                    "val_recall": recall,
                    "val_f1": f1,
                    "val_threshold": threshold,
                }
            )

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = {
                "best_epoch": epoch + 1,
                "val_f1": f1,
                "val_precision": precision,
                "val_recall": recall,
                "val_loss": avg_val_loss,
                "train_loss": avg_train_loss,
                "val_threshold": threshold,
            }
            epochs_no_improve = 0
            logger.info(f"  --> New Best IECE F1: {best_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"  --> Early stopping at epoch {epoch+1}")
                break

        # ReduceLROnPlateau starts only after warmup.
        if epoch >= warmup_epochs:
            scheduler.step(f1)

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    logger.info(f"[Supervised Training] Complete, Best IECE F1 = {best_f1:.4f}")
    return best_state, best_f1, best_metrics


# ---------------------------------------------------------------------------
# 10-fold cross-validation main loop.
# ---------------------------------------------------------------------------


def train_kfold(
    batch_size: int = Config.DL_BATCH_SIZE,
    learning_rate: float = Config.DL_LEARNING_RATE,
    epochs: int = Config.DL_EPOCHS,
    device: torch.device = Config.DEVICE,
    save_dir: Path = Path("checkpoints"),
    log_dir: Path = Path("logs"),
    smoke_test: bool = False,
    ablation_mode: str = Config.DL_ABLATION_MODE,
    dataset_type: str = Config.DATASET_TYPE,
) -> List[float]:
    """
    Main IECE 10-fold cross-validation flow for supervised training.

    Returns:
        List of best F1 values for the 10 folds.
    """
    if smoke_test:
        epochs = 2
        logger.info("[Smoke Test] Enabled: running only Fold 1 for up to 2 epochs")

    Config.apply_dataset_type(dataset_type)
    logger.info(
        f"Dataset mode: {Config.DATASET_TYPE} | data={Config.DATASET} | "
        f"emb={Config.DL_EMBEDDING_DIR}"
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    # Create the training metrics log in JSON Lines format.
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_log_path = log_dir / f"training_{run_ts}.jsonl"
    metrics_logger = MetricsLogger(metrics_log_path)

    # Load all data as 10-fold DataLoaders; no tokenizer is needed here.
    folds = load_all_data(batch_size=batch_size)
    logger.info(
        f"Data loading complete: 10-fold CV | dataset={Config.DATASET_TYPE} | "
        f"ablation_mode={ablation_mode}"
    )

    fold_f1_scores: List[float] = []

    for fold_idx, (train_loader, val_loader, labeled_loader) in enumerate(folds):
        if smoke_test and fold_idx >= 1:
            logger.info("[Smoke Test] Fold 1 complete, exiting")
            break

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Fold {fold_idx + 1}/10")
        logger.info(f"{'='*60}")

        # Create the IMR model.
        model = IMRModel(
            hidden_size=Config.DL_HIDDEN_SIZE,
            d_model=Config.DL_D_MODEL,
            nhead=Config.DL_NHEAD,
            dim_feedforward=Config.DL_DIM_FF,
            n_emotion_layers=Config.DL_N_EMOTION_LAYERS,
            n_cause_layers=Config.DL_N_CAUSE_LAYERS,
            num_emotions=Config.get_num_emotions(),
            n_iterations=Config.DL_IMR_ITERATIONS,
            dropout=Config.DL_DROPOUT,
            use_event_emb=Config.DL_USE_EVENT_EMB,
            cause_input_mode=Config.DL_CAUSE_INPUT_MODE,
            ablation_mode=ablation_mode,
        ).to(device)

        best_state, best_f1, best_metrics = train(
            model=model,
            train_loader=labeled_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            metrics_logger=metrics_logger,
            fold_idx=fold_idx,
        )

        # Save the best model.
        save_dir.mkdir(parents=True, exist_ok=True)
        fold_save_path = save_dir / f"fold_{fold_idx + 1}_best.pt"
        torch.save(best_state, fold_save_path)
        logger.info(f"Fold {fold_idx + 1} best model saved: {fold_save_path}")

        metrics_logger.log(
            {
                "fold": fold_idx + 1,
                "phase": "fold_best",
                "val_f1": best_f1,
                "val_precision": best_metrics["val_precision"],
                "val_recall": best_metrics["val_recall"],
                "val_loss": best_metrics["val_loss"],
                "train_loss": best_metrics["train_loss"],
                "best_epoch": best_metrics["best_epoch"],
                "val_threshold": best_metrics["val_threshold"],
            }
        )
        fold_f1_scores.append(best_f1)

        del model
        torch.cuda.empty_cache()

    # === Summarize results. ===
    logger.info(f"\n{'='*60}")
    logger.info("10-fold cross-validation complete")
    logger.info(f"{'='*60}")
    for i, f1 in enumerate(fold_f1_scores, 1):
        logger.info(f"Fold {i} Best F1: {f1:.4f}")
    avg_f1 = sum(fold_f1_scores) / max(len(fold_f1_scores), 1)

    logger.info(f"Average F1: {avg_f1:.4f}")

    metrics_logger.log(
        {
            "phase": "cv_summary",
            "fold_f1_scores": fold_f1_scores,
            "avg_f1": avg_f1,
        }
    )
    logger.info(f"[MetricsLogger] Training log saved to: {metrics_log_path}")

    return fold_f1_scores


def apply_hparams_from_file(hparams_path: Path) -> None:
    """Override Config from the JSON file produced by hparam_search."""
    if not hparams_path.exists():
        raise FileNotFoundError(f"Hyperparameter file does not exist: {hparams_path}")

    with open(hparams_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    params = payload.get("params", {})
    derived = payload.get("derived", {})

    mapping = {
        "learning_rate": "DL_LEARNING_RATE",
        "dropout": "DL_DROPOUT",
        "d_model": "DL_D_MODEL",
        "n_emotion_layers": "DL_N_EMOTION_LAYERS",
        "n_cause_layers": "DL_N_CAUSE_LAYERS",
        "imr_iterations": "DL_IMR_ITERATIONS",
        "batch_size": "DL_BATCH_SIZE",
        "weight_decay": "DL_WEIGHT_DECAY",
        "label_smoothing": "DL_LABEL_SMOOTHING",
        "warmup_epochs": "DL_WARMUP_EPOCHS",
        "lr_shared": "LR_SHARED",
        "lr_emo": "LR_EMO",
        "lr_cause": "LR_CAUSE",
    }

    for key, value in params.items():
        attr = mapping.get(key)
        if attr is None:
            continue
        setattr(Config, attr, value)

    if "nhead" in derived:
        Config.DL_NHEAD = int(derived["nhead"])
    if "dim_ff" in derived:
        Config.DL_DIM_FF = int(derived["dim_ff"])

    logger.info(f"Loaded best hyperparameters from: {hparams_path}")


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="IMR training script")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Smoke-test mode: run only fold 1 for up to 2 epochs",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=Config.DL_ABLATION_MODE,
        choices=["full", "wo_imr", "wo_backward"],
        help=(
            "Model variant: full=bidirectional IMR, "
            "wo_imr=no cross-task interaction, "
            "wo_backward=Emotion->Cause only"
        ),
    )
    parser.add_argument(
        "--hparams-file",
        type=str,
        default="logs/best_hparams.json",
        help=(
            "Path to best-hyperparameter JSON; loaded automatically if it exists, "
            "otherwise Config defaults are used"
        ),
    )
    parser.add_argument(
        "--no-hparams",
        action="store_true",
        help="Ignore an existing best-hyperparameter file and use Config defaults",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    Config.apply_dataset_type("IECE")
    hparams_path = Path(args.hparams_file)
    loaded_hparams = False
    if not args.no_hparams:
        if hparams_path.exists():
            apply_hparams_from_file(hparams_path)
            loaded_hparams = True
        else:
            logger.info(
                f"Best-hyperparameter file not found; using Config defaults: {hparams_path}"
            )

    emb_dir = Path(Config.DL_EMBEDDING_DIR)
    legacy_dir = Path(Config.DL_EMBEDDING_ROOT)
    has_embeddings = (emb_dir / "labeled_text.npy").exists() or (
        legacy_dir / "labeled_text.npy"
    ).exists()

    if not has_embeddings:
        from src.prepare_embeddings import pre_embed

        pre_embed()

    logger.info("IMR IECE training started")
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Dataset type: {Config.DATASET_TYPE}")
    logger.info(f"Data path: {Config.DATASET}")
    logger.info(f"Embedding directory: {Config.DL_EMBEDDING_DIR}")
    logger.info(f"Hyperparameter source: {hparams_path if loaded_hparams else 'Config defaults'}")
    logger.info(f"Ablation mode: {args.ablation}")

    fold_f1s = train_kfold(
        batch_size=Config.DL_BATCH_SIZE,
        learning_rate=Config.DL_LEARNING_RATE,
        epochs=Config.DL_EPOCHS,
        device=Config.DEVICE,
        log_dir=Path("logs"),
        smoke_test=args.smoke_test,
        ablation_mode=args.ablation,
        dataset_type=Config.DATASET_TYPE,
    )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
    # os.system(f"shutdown")
