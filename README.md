# Parameter-Efficient Fine-Tuning of a Small Language Model (LoRA)

## 📌 Project Overview
This project demonstrates **parameter-efficient fine-tuning** of a small **decoder-only language model** for text classification using **LoRA** — strictly low-rank updates, no other PEFT method. A pretrained backbone is adapted to a new dataset by learning small low-rank deltas on the attention **query/value** projections, while the backbone itself stays frozen. This reaches near full fine-tuning accuracy while updating only a tiny fraction of the weights.

**Dataset**: AG News (4 news topics: World, Sports, Business, Sci/Tech).  
**Backbone**: `HuggingFaceTB/SmolLM2-360M` with a classification head, via `transformers`.  
**Goal**: Strong top-1 accuracy while training well under 5% of the model's parameters.

I built this as hands-on preparation for the PEFT/LoRA side of my thesis; everything here is a standalone prototype on public data and public weights.

![Dataset Samples](results/ag_news_samples.png)

---

## 🚀 Key Features
1. **Hand-Written LoRA**:
   - Low-rank matrices injected into the attention q/v projections (`B · A · x · α/r`, with `α = 2r` and `B` zero-initialised so training starts exactly from the pretrained model). HF decoder blocks keep q/k/v as separate Linears, so each projection is wrapped on its own.
   - Placement follows the original [LoRA paper (Hu et al., 2022)](https://arxiv.org/abs/2106.09685), whose placement study (§7.1) found adapting **q and v** the best use of a fixed parameter budget — k contributes least.
   - Only the LoRA matrices and the classifier head are trainable; the backbone is fully frozen.
2. **Rank Ablation**:
   - One command sweeps the LoRA rank over {4, 8, 16, 32} and plots accuracy and cost against rank.
3. **Tiny Checkpoints**:
   - Only the LoRA weights and head are saved — a few MB instead of the full 1.4 GB backbone. Inference rebuilds the model from public pretrained weights and loads the LoRA weights on top.
4. **Solid Training Recipe**:
   - AdamW with a 2-epoch linear warmup into cosine decay, mixed precision, on-the-fly tokenization.
5. **Configurable with Hydra**:
   - Data, model, and training settings live in `configs/` and can be overridden straight from the command line.
6. **Experiment Tracking**:
   - Metrics are logged to Weights & Biases in **offline** mode by default, so it runs without an account.

---

## 🔍 Findings
Training runs are in progress — results tables and figures land here once they finish.

---

## ⚙️ How to Run
Works on Linux, macOS and Windows.

```bash
git clone https://github.com/headless-start/peft-lora-llm.git
cd peft-lora-llm

python -m venv .venv
source .venv/bin/activate          # linux / macos
# .venv\Scripts\activate           # windows

pip install -r requirements.txt
```

For GPU training install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) first; the plain `pip install` gives you a CPU build on some platforms.

```bash
# full run on AG News (downloads the backbone and dataset on first use)
python train.py

# override anything from the command line
python train.py train.epochs=3 data.batch_size=16 model.lora.r=16
```

Sweep the LoRA rank (writes `results/ablation.json` and `results/ablation.png`):

```bash
python ablate.py                    # ranks 4, 8, 16, 32
python ablate.py --ranks 4,8
```

Sweep the LoRA placement over q/k/v subsets at fixed rank (writes `results/placement.json` and `results/placement.png`):

```bash
python ablate.py --placements q,k,v,qk,qv,qkv --ranks 8
```

Compare LoRA against the baselines — linear probe and full fine-tuning (writes `results/baselines.json` and `results/baselines.png`):

```bash
python baselines.py
```

Classify your own news snippets with a trained checkpoint:

```bash
python predict.py "Stocks rallied after the central bank held rates steady."
# Stocks rallied after the central bank held rates steady.: Business (99.1%), Sci/Tech (0.5%)
```

Quick smoke test (CPU, small random-init backbone, no downloads):

```bash
python train.py +experiment=smoke
```

Runs are logged to Weights & Biases offline by default; to sync to the cloud:

```bash
wandb login
python train.py wandb.mode=online
```

Training curves and `metrics.json` are written to `results/`; checkpoints go to `outputs/`.

---

## 🛠 System Requirements
### Dependencies
- Python 3.10+
- Libraries: `torch`, `transformers`, `datasets`, `hydra-core`, `wandb`, `matplotlib`
- Hardware: CUDA GPU recommended (a CPU smoke run is supported)

### Reproducibility
- Runs on Linux, macOS and Windows; all paths and commands are OS-agnostic.
- Seeds are fixed (`seed: 42`). Expect individual numbers to move slightly across reruns and library versions due to GPU non-determinism.
- On machines with little RAM, add `data.num_workers=0` to any command.

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
