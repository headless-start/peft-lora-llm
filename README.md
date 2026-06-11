# Parameter-Efficient Fine-Tuning of a Small Language Model (LoRA)

## 📌 Project Overview
This project demonstrates **parameter-efficient fine-tuning** of a small **decoder-only language model** for text classification using **LoRA** — strictly low-rank updates, no other PEFT method. A pretrained backbone is adapted to a new dataset by learning small low-rank deltas on the attention **query/value** projections, while the backbone itself stays frozen.

**Dataset**: AG News (4 news topics: World, Sports, Business, Sci/Tech).  
**Backbone**: `HuggingFaceTB/SmolLM2-360M` with a classification head, via `transformers`.  
**Goal**: Strong top-1 accuracy while training well under 5% of the model's parameters.

Work in progress — code first, results and figures land once the full runs finish.

---

## ⚙️ How to Run
```bash
git clone https://github.com/headless-start/peft-lora-llm.git
cd peft-lora-llm

python -m venv .venv
source .venv/bin/activate          # linux / macos
# .venv\Scripts\activate           # windows

pip install -r requirements.txt
python train.py
```

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
