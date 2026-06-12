import argparse

import torch
from omegaconf import OmegaConf

from src.data import CLASSES, NUM_CLASSES, build_tokenizer
from src.model import build_model


def load_model(ckpt_path, backbone, r, alpha_factor, device, placement="qv"):
    """Rebuild the model on pretrained weights, then load the trained LoRA weights and head."""
    cfg = OmegaConf.create({
        "model": {"backbone": backbone, "pretrained": True, "mode": "lora",
                  "lora": {"r": r, "alpha_factor": alpha_factor, "dropout": 0.0,
                           "placement": list(placement)}},
    })
    model, _ = build_model(cfg, NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert not unexpected, f"checkpoint keys not in model: {unexpected[:3]}"
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(model, tokenizer, text, device, topk=2, max_length=128):
    """Classify one string and return the top-k (topic, probability) pairs."""
    enc = tokenizer(text, truncation=True, max_length=max_length,
                    return_tensors="pt").to(device)
    probs = model(**enc).logits.softmax(-1).squeeze(0)
    top = probs.topk(topk)
    return [(CLASSES[i], p.item()) for p, i in zip(top.values, top.indices)]


def main():
    parser = argparse.ArgumentParser(description="Classify news snippets with the fine-tuned SmolLM2")
    parser.add_argument("texts", nargs="+", help="text snippet(s) to classify")
    parser.add_argument("--ckpt", default="outputs/best.pt")
    parser.add_argument("--backbone", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--alpha-factor", type=int, default=2)
    parser.add_argument("--placement", default="qv", help="which projections carry lora, e.g. qv")
    parser.add_argument("--topk", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, args.backbone, args.lora_r, args.alpha_factor, device,
                       args.placement)
    tokenizer = build_tokenizer(args.backbone)

    for text in args.texts:
        preds = predict(model, tokenizer, text, device, args.topk)
        labels = ", ".join(f"{name} ({p:.1%})" for name, p in preds)
        snippet = text if len(text) <= 60 else text[:57] + "..."
        print(f"{snippet}: {labels}")


if __name__ == "__main__":
    main()
