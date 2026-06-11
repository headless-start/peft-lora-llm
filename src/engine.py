import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, device, optimizer=None, scaler=None, use_amp=False, limit_batches=None):
    """Run one epoch; trains when an optimizer is passed, otherwise evaluates. Returns (loss, acc)."""
    train = optimizer is not None
    model.train(train)
    loss_sum, correct, total = 0.0, 0, 0

    for i, batch in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        y = batch.pop("labels")

        with torch.set_grad_enabled(train), torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(**batch).logits
            loss = criterion(out, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total
