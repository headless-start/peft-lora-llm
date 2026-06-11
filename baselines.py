import argparse
import gc
import json
import os

import torch
from hydra import compose, initialize

from src.plot import save_tradeoff
from train import run_training

# linear probe (lower bound), lora (ours), full fine-tuning (upper bound);
# full ft gets the lower lr that pretrained weights want and a batch that fits 8 GB
METHODS = [
    ("head", ["model.mode=head", "wandb.name=head-only"]),
    ("lora", ["model.mode=lora", "wandb.name=lora-r8"]),
    ("full", ["model.mode=full", "train.lr=3e-5", "data.batch_size=8", "wandb.name=full-ft"]),
]


def main():
    parser = argparse.ArgumentParser(description="head-only vs lora vs full fine-tuning")
    parser.add_argument("--methods", default="head,lora,full",
                        help="comma-separated subset of head,lora,full")
    parser.add_argument("overrides", nargs="*",
                        help="extra hydra overrides, e.g. data.num_workers=0")
    args = parser.parse_args()
    wanted = args.methods.split(",")

    # keep rows from earlier invocations so rerunning one method doesn't lose the rest
    rows = []
    if os.path.exists("results/baselines.json"):
        with open("results/baselines.json") as f:
            rows = [r for r in json.load(f) if r["method"] not in wanted]

    order = [name for name, _ in METHODS]
    with initialize(version_base=None, config_path="configs"):
        for name, method_overrides in METHODS:
            if name not in wanted:
                continue
            cfg = compose(config_name="config", overrides=[*method_overrides, *args.overrides])
            print(f"\n=== {name} ===")
            res = run_training(cfg, ckpt_path=f"outputs/best_{name}.pt")
            rows.append({"method": name, "top1_acc": round(res["best_acc"], 4),
                         "trainable_params": res["trainable"],
                         "trainable_pct": round(100 * res["trainable"] / res["total"], 3),
                         "ckpt_mb": round(os.path.getsize(res["ckpt_path"]) / 1024**2, 1),
                         "epoch_sec": res["epoch_sec"],
                         "peak_vram_gb": res["peak_vram_gb"]})
            rows.sort(key=lambda r: order.index(r["method"]))
            # write after every run so a crash doesn't lose the finished ones
            os.makedirs("results", exist_ok=True)
            with open("results/baselines.json", "w") as f:
                json.dump(rows, f, indent=2)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_tradeoff(rows, "results/baselines.png")

    print("\n| method | top-1 acc | trainable params | ckpt size | s/epoch | peak vram |")
    print("|--------|-----------|------------------|-----------|---------|-----------|")
    for r in rows:
        vram = f"{r['peak_vram_gb']} GB" if r["peak_vram_gb"] else "—"
        print(f"| {r['method']} | {r['top1_acc']:.1%} | {r['trainable_params'] / 1e3:.0f}K "
              f"({r['trainable_pct']:.2f}%) | {r['ckpt_mb']} MB | {r['epoch_sec']} | {vram} |")


if __name__ == "__main__":
    main()
