import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_curve(history, path):
    """Plot train/val loss and accuracy across epochs and save the figure to path."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="loss")
    ax_loss.legend()

    ax_acc.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax_acc.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_ablation(rows, path):
    """Plot accuracy and trainable-parameter count against LoRA rank."""
    ranks = [r["r"] for r in rows]
    fig, (ax_acc, ax_par) = plt.subplots(1, 2, figsize=(10, 4))

    ax_acc.plot(ranks, [r["top1_acc"] for r in rows], marker="o")
    ax_acc.set(xlabel="lora rank", ylabel="top-1 accuracy", title="accuracy vs rank",
               xscale="log", xticks=ranks, xticklabels=[str(r) for r in ranks])

    ax_par.plot(ranks, [r["trainable_params"] / 1e3 for r in rows], marker="o", color="tab:orange")
    ax_par.set(xlabel="lora rank", ylabel="trainable params (K)", title="cost vs rank",
               xscale="log", xticks=ranks, xticklabels=[str(r) for r in ranks])

    for ax in (ax_acc, ax_par):
        ax.minorticks_off()  # log-scale minor labels collide with the rank ticks

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_tradeoff(rows, path):
    """Scatter of accuracy against trainable parameters for the method comparison."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        ax.scatter(r["trainable_params"] / 1e6, r["top1_acc"], s=80, zorder=3)
        ax.annotate(r["method"], (r["trainable_params"] / 1e6, r["top1_acc"]),
                    textcoords="offset points", xytext=(8, -3))
    ax.set(xlabel="trainable params (M, log scale)", ylabel="top-1 accuracy",
           title="accuracy vs trainable parameters", xscale="log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_placement(rows, path):
    """Bar chart of accuracy by LoRA placement (q/k/v subsets) at fixed rank."""
    labels = [r["placement"] for r in rows]
    accs = [r["top1_acc"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, accs, color="tab:blue")
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    lo, hi = min(accs), max(accs)
    ax.set_ylim(lo - 3 * (hi - lo + 0.005), hi + 2 * (hi - lo + 0.005))
    ax.set(xlabel="lora placement", ylabel="top-1 accuracy", title="accuracy vs placement")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_samples(rows, path):
    """Render a few example texts with their topic labels as a small figure."""
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(rows)))
    ax.axis("off")
    for i, (text, label) in enumerate(rows):
        y = 1 - i / len(rows)
        snippet = textwrap.shorten(text, width=120, placeholder=" …")
        ax.text(0.0, y, label, fontsize=10, fontweight="bold", color="tab:blue",
                va="top", transform=ax.transAxes)
        ax.text(0.11, y, snippet, fontsize=9, va="top", wrap=True, transform=ax.transAxes)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
