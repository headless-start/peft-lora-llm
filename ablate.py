import argparse
import gc
import json
import os

import torch
from hydra import compose, initialize

from src.plot import save_ablation, save_placement
from train import run_training


def main():
    parser = argparse.ArgumentParser(description="LoRA rank / placement ablation")
    parser.add_argument("--ranks", default="4,8,16,32",
                        help="comma-separated lora ranks to sweep")
    parser.add_argument("--placements", default="qv",
                        help="comma-separated placements to sweep, e.g. q,v,qv,qkv")
    parser.add_argument("overrides", nargs="*",
                        help="extra hydra overrides, e.g. data.num_workers=0")
    args = parser.parse_args()
    ranks = [int(r) for r in args.ranks.split(",")]
    placements = args.placements.split(",")

    # rank sweeps write ablation.*, placement sweeps write placement.*
    stem = "placement" if len(placements) > 1 else "ablation"

    # keep rows from earlier invocations so rerunning part of a sweep doesn't lose the rest
    keys = {(r, p) for p in placements for r in ranks}
    rows = []
    if os.path.exists(f"results/{stem}.json"):
        with open(f"results/{stem}.json") as f:
            rows = [row for row in json.load(f) if (row["r"], row["placement"]) not in keys]

    with initialize(version_base=None, config_path="configs"):
        for p in placements:
            for r in ranks:
                overrides = [f"model.lora.r={r}",
                             f"model.lora.placement=[{','.join(p)}]",
                             f"wandb.name=lora-r{r}-{p}", *args.overrides]
                cfg = compose(config_name="config", overrides=overrides)
                print(f"\n=== lora r={r} placement={p} ===")
                res = run_training(cfg, ckpt_path=f"outputs/best_r{r}_{p}.pt")
                rows.append({"r": r, "placement": p,
                             "top1_acc": round(res["best_acc"], 4),
                             "trainable_params": res["trainable"],
                             "trainable_pct": round(100 * res["trainable"] / res["total"], 3)})
                rows.sort(key=lambda row: (len(row["placement"]),
                                           ["qkv".index(c) for c in row["placement"]],
                                           row["r"]))
                # write after every run so a crash doesn't lose the finished ones
                os.makedirs("results", exist_ok=True)
                with open(f"results/{stem}.json", "w") as f:
                    json.dump(rows, f, indent=2)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if len(placements) > 1:
        save_placement(rows, f"results/{stem}.png")
    else:
        save_ablation(rows, f"results/{stem}.png")

    print("\n| rank | placement | top-1 acc | trainable params | % of total |")
    print("|------|-----------|-----------|------------------|------------|")
    for row in rows:
        print(f"| {row['r']} | {row['placement']} | {row['top1_acc']:.1%} | "
              f"{row['trainable_params'] / 1e3:.0f}K | {row['trainable_pct']:.2f}% |")


if __name__ == "__main__":
    main()
