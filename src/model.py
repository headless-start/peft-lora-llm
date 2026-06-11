import torch
from transformers import (AutoModelForSequenceClassification, LlamaConfig,
                          LlamaForSequenceClassification)

from .lora import LoRALinear, inject_lora


def build_model(cfg, num_classes):
    """Build the LLM classifier in the configured mode: lora, head (linear probe) or full."""
    if cfg.model.pretrained:
        # fp32 master weights — amp handles the half-precision compute
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.backbone, num_labels=num_classes, dtype=torch.float32)
    else:
        # tiny random-init stand-in with the same q/k/v layout — used by the
        # smoke run, no downloads
        config = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=64,
                             num_labels=num_classes)
        model = LlamaForSequenceClassification(config)
    if model.config.pad_token_id is None:
        # decoder-only models pool the last non-pad token, so they need this set
        model.config.pad_token_id = model.config.eos_token_id

    mode = cfg.model.get("mode", "lora")
    n_lora = 0
    if mode == "lora":
        r = cfg.model.lora.r
        alpha = cfg.model.lora.alpha_factor * r
        targets = tuple(cfg.model.lora.get("placement", ["q", "v"]))
        n_lora = inject_lora(model, r, alpha, cfg.model.lora.dropout, targets)
        freeze_backbone(model)
    elif mode == "head":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.score.parameters():
            p.requires_grad_(True)
    elif mode != "full":
        raise ValueError(f"unknown mode: {mode}")
    return model, n_lora


def freeze_backbone(model):
    """Freeze everything, then re-enable the LoRA matrices and the classifier head."""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for name, p in m.named_parameters():
                if name.startswith("lora_"):
                    p.requires_grad_(True)
    for p in model.score.parameters():
        p.requires_grad_(True)


def trainable_state_dict(model):
    """State dict with just the trainable tensors (LoRA matrices + head) — a few MB, not 1.4 GB."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in keep}
