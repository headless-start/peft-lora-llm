import json
import os
import time

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.data import build_loaders, sample_rows
from src.engine import run_epoch
from src.model import build_model, trainable_state_dict
from src.plot import save_curve, save_samples
from src.utils import count_parameters, set_seed


def build_scheduler(optimizer, cfg):
    """Linear warmup for the first epochs, then cosine annealing down to min_lr."""
    warmup = cfg.train.warmup_epochs
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs - warmup, eta_min=cfg.train.min_lr)
    if warmup == 0:
        return cosine
    linear = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1 / (warmup + 1), total_iters=warmup)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [linear, cosine], milestones=[warmup])


def run_training(cfg, ckpt_path="outputs/best.pt"):
    """Train the LoRA matrices and head with the given config; returns metrics and history."""
    set_seed(cfg.seed)
    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = cfg.train.amp and use_cuda

    train_loader, val_loader, num_classes = build_loaders(cfg)
    model, n_lora = build_model(cfg, num_classes)
    model.to(device)

    trainable, total = count_parameters(model)
    print(f"backbone={cfg.model.backbone}  lora_r={cfg.model.lora.r}  lora_layers={n_lora}  "
          f"trainable={trainable:,}/{total:,} ({100 * trainable / total:.2f}%)")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    wandb.init(project=cfg.wandb.project, mode=cfg.wandb.mode, name=cfg.wandb.name,
               config=OmegaConf.to_container(cfg, resolve=True))

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    history, best_acc, epoch_secs = [], 0.0, []
    for epoch in range(cfg.train.epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer, scaler,
                                    use_amp, cfg.train.limit_train_batches)
        epoch_secs.append(time.time() - t0)
        va_loss, va_acc = run_epoch(model, val_loader, device, use_amp=use_amp,
                                    limit_batches=cfg.train.limit_val_batches)
        scheduler.step()

        wandb.log({"epoch": epoch, "lr": scheduler.get_last_lr()[0],
                   "train/loss": tr_loss, "train/acc": tr_acc,
                   "val/loss": va_loss, "val/acc": va_acc})
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        print(f"epoch {epoch:3d}  train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")

        if va_acc > best_acc:
            best_acc = va_acc
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            # only the lora weights + head go in the checkpoint — a few MB instead of 1.4 GB
            torch.save({"model": trainable_state_dict(model), "epoch": epoch,
                        "val_acc": best_acc}, ckpt_path)

    wandb.finish()
    print(f"best val acc: {best_acc:.4f}")
    peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 2) if use_cuda else None
    return {"best_acc": best_acc, "trainable": trainable, "total": total,
            "history": history, "epoch_sec": round(sum(epoch_secs) / len(epoch_secs), 1),
            "peak_vram_gb": peak_vram_gb, "ckpt_path": ckpt_path}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Single training run: train, then write the curve figure and metrics.json."""
    res = run_training(cfg, ckpt_path=cfg.train.get("ckpt", "outputs/best.pt"))

    # smoke runs write their artifacts to outputs/ so they never clobber real results
    out_dir = "outputs" if cfg.data.fake else "results"
    os.makedirs(out_dir, exist_ok=True)
    save_curve(res["history"], f"{out_dir}/training_curve.png")
    if not cfg.data.fake:
        save_samples(sample_rows(cfg), f"{out_dir}/ag_news_samples.png")
    metrics = {
        "backbone": cfg.model.backbone,
        "dataset": cfg.data.name,
        "epochs": cfg.train.epochs,
        "top1_acc": round(res["best_acc"], 4),
        "trainable_params": res["trainable"],
        "total_params": res["total"],
        "trainable_pct": round(100 * res["trainable"] / res["total"], 3),
        "lora": OmegaConf.to_container(cfg.model.lora, resolve=True),
    }
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
