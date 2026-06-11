import math

import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen Linear plus a trainable low-rank delta (B · A · x · α/r).

    HF decoder blocks keep q_proj / k_proj / v_proj as separate Linears
    (unlike timm's fused qkv), so each target projection gets wrapped on
    its own. B is zero-initialised so training starts exactly from the
    pretrained model; the default q+v placement follows the placement
    study in the original LoRA paper.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        """Frozen projection output plus the scaled low-rank delta."""
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def inject_lora(model, r: int, alpha: int, dropout: float = 0.0, targets=("q", "v")):
    """Wrap every target attention projection (q_proj/k_proj/v_proj) with a LoRALinear."""
    names = {f"{t}_proj" for t in targets}
    sites = [(parent, name) for parent in model.modules()
             for name, child in parent.named_children()
             if name in names and isinstance(child, nn.Linear)]
    for parent, name in sites:
        setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha, dropout))
    return len(sites)
