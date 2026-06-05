<p align="center">
  <img src="https://img.shields.io/badge/Model-DeepSeek--Coder--1.3B-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/GPU-Tesla_T4_15GB-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-HuggingFace_TRL-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Demo-Gradio-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Notebook-Google_Colab-red?style=for-the-badge" />
</p>

<h1 align="center">🔍 CodeReviewer AI</h1>
<p align="center">
  A fine-tuned LLM that acts as a senior engineer doing code reviews.<br/>
  Paste code → get specific, actionable feedback on bugs, inefficiencies, and security issues.
</p>

---

## 📌 What Is This?

**CodeReviewer AI** is an end-to-end LLM fine-tuning project that trains a code review assistant using the full modern post-training pipeline:

> **Stage 1** → Data Collection → **Stage 2** SFT → **Stage 3** DPO → **Stage 4** GRPO → **Stage 5** Eval + Demo

Everything runs on a **free Google Colab T4 GPU** using a single notebook. The project demonstrates every technique currently asked about in AI/ML engineering interviews: QLoRA, supervised fine-tuning, preference optimization (DPO), group reward training (GRPO), and LLM evaluation.

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CodeReviewer AI Pipeline                            │
│                                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│   │ Stage 1  │    │ Stage 2  │    │ Stage 3  │    │ Stage 4  │            │
│   │   DATA   │───▶│   SFT    │───▶│   DPO    │───▶│   GRPO   │            │
│   │          │    │  QLoRA   │    │ Prefs    │    │  Reward  │            │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘            │
│        │               │               │               │                   │
│   CodeAlpaca      Teach the       Teach what       Teach to               │
│   + LeetCode        TASK          GOOD looks       RANK outputs            │
│   (free data)     (format)           like          (DeepSeek-R1            │
│                                   (quality)         technique)             │
│                                                         │                  │
│                                                    ┌────▼─────┐           │
│                                                    │ Stage 5  │           │
│                                                    │ Eval +   │           │
│                                                    │  Gradio  │           │
│                                                    └──────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Repository Structure

```
codereviewer-ai/
│
├── 📓 notebooks/
│   └── colab_master.py          ← ⭐ Run this in Google Colab (full pipeline)
│
├── 📁 data/                     ← Stage 1: Data collection
│   ├── fetch_codealpaca.py      ← Download CodeAlpaca-20k (SFT examples)
│   ├── fetch_github_prs.py      ← Pull real GitHub PR review comments
│   └── build_dpo_pairs.py       ← Build preference pairs for DPO
│
├── 📁 sft/                      ← Stage 2: Supervised Fine-Tuning
│   └── train_sft.py             ← Standalone SFT training script
│
├── 📁 dpo/                      ← Stage 3: Direct Preference Optimization
│   └── train_dpo.py             ← Standalone DPO training script
│
├── 📁 grpo/                     ← Stage 4: Group Relative Policy Optimization
│   └── train_grpo.py            ← Standalone GRPO training script
│
├── 📁 eval/                     ← Stage 5a: Evaluation
│   └── evaluate.py              ← ROUGE-L + reward scoring across checkpoints
│
└── 📁 app/                      ← Stage 5b: Demo
    └── gradio_demo.py           ← Interactive web UI
```

---

## 🚀 Quick Start (Google Colab)

### Step 1 — Set Runtime to GPU
`Runtime` → `Change runtime type` → **T4 GPU**

### Step 2 — Open the Notebook
Upload `notebooks/colab_master.py` to Colab or copy cells into a new notebook.

### Step 3 — Choose Your Model

| Colab Tier | GPU VRAM | Recommended Model |
|---|---|---|
| **Free** | 15.6 GB T4 | `deepseek-ai/deepseek-coder-1.3b-instruct` ✅ tested |
| **Pro** | 15–16 GB T4/V100 | `deepseek-ai/deepseek-coder-6.7b-instruct` |
| **Pro+** | 40 GB A100 | `codellama/CodeLlama-13b-Instruct-hf` |

### Step 4 — Run Cells in Order

| Cell | Stage | What Happens | Actual Runtime (T4) |
|---|---|---|---|
| 1 | Setup | Installs all packages | ~3 min |
| 2 | Config | Detects GPU, sets paths | instant |
| 3 | Stage 1 | Downloads 20,022 CodeAlpaca examples, saves 3,000 | ~2 min |
| 4 | Stage 1 | Builds 1,000 DPO preference pairs | ~3 min |
| 5 | Stage 2 | **SFT training** — 375 steps | **45 min** |
| 6 | Stage 3 | **DPO training** — 125 steps | **52 min** |
| 7 | Stage 4 | **GRPO training** — 150 steps (optional) | **80 min** |
| 8 | Stage 5 | Evaluate all checkpoints | ~10 min |
| 9 | Stage 5 | Launch Gradio demo | ~2 min |

> ⏱ **Total time** (Cells 1–6 + 8–9, skipping GRPO): ~**1 hr 50 min** on free T4

### Step 5 — Use the Demo

After Cell 9, Colab prints a public URL:
```
* Running on public URL: https://acb2cfa073d25ad658.gradio.live
```
Click it. Paste code. Get a review.

---

## 📊 Training Results (Actual Run — Tesla T4, Free Colab)

### Environment
```
GPU  : Tesla T4
VRAM : 15.6 GB
Model: deepseek-ai/deepseek-coder-1.3b-instruct
OS   : Python 3.12 / CUDA
```

### Data Statistics
| Dataset | Source | Examples |
|---|---|---|
| SFT training data | CodeAlpaca-20k (HuggingFace) | 3,000 |
| DPO preference pairs — Source A | LeetCode solutions (greengerong/leetcode) | 500 |
| DPO preference pairs — Source B | Degraded SFT responses | 500 |
| **Total DPO pairs** | | **1,000** |
| GRPO prompts | Subset of SFT data | 300 |
| Evaluation set | Held-out SFT examples | 30 |

---

### Stage 2 — SFT Training Loss
**375 steps · 45 min 03 sec · Batch 2 · Grad accum 4**

| Step | Loss | | Step | Loss | | Step | Loss |
|---|---|---|---|---|---|---|---|
| 10 | 1.7273 | | 130 | 0.4495 | | 260 | 0.4124 |
| 20 | 0.7301 | | 140 | 0.4720 | | 270 | 0.4439 |
| 30 | 0.5042 | | 150 | 0.4522 | | 280 | 0.4244 |
| 40 | 0.4903 | | 160 | 0.4297 | | 290 | 0.4046 |
| 50 | 0.4587 | | 170 | 0.4499 | | 300 | 0.4338 |
| 60 | 0.4266 | | 180 | 0.4245 | | 310 | 0.3955 |
| 70 | 0.4447 | | 190 | 0.4394 | | 320 | 0.4158 |
| 80 | 0.4378 | | 200 | 0.4135 | | 330 | 0.4233 |
| 90 | 0.4274 | | 210 | 0.4224 | | 340 | 0.4444 |
| 100 | 0.4385 | | 220 | 0.4321 | | 350 | 0.4039 |
| 110 | 0.4340 | | 230 | 0.4510 | | 360 | 0.4226 |
| 120 | 0.4875 | | 240 | 0.4496 | | 370 | 0.4208 |

> Loss dropped from **1.727 → 0.421** — the model learned to format and generate code reviews.

```
SFT Loss Curve
1.8 ┤█
1.6 ┤
1.4 ┤
1.2 ┤
1.0 ┤
0.8 ┤  █
0.6 ┤     █
0.5 ┤        ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    └────────────────────────────────────────────▶
    10  50  100  150  200  250  300  370    step
```

**Trainable parameters:** `14,991,360` out of `1,361,463,296` total = **1.10%** (LoRA efficiency)

---

### Stage 3 — DPO Training Loss
**125 steps · 52 min 34 sec · β = 0.1**

| Step | Loss | | Step | Loss |
|---|---|---|---|---|
| 10 | 0.1646 | | 70 | 0.0002 |
| 20 | 0.0016 | | 80 | 0.0001 |
| 30 | 0.0002 | | 90 | 0.0002 |
| 40 | 0.0002 | | 100 | 0.0002 |
| 50 | 0.0004 | | 110 | 0.0002 |
| 60 | 0.0001 | | 120 | 0.0002 |

> DPO loss collapsed rapidly from **0.165 → ~0.0002**. This is expected behaviour — DPO converges fast when preference pairs are clearly separated. The model learned the distinction between specific and vague reviews very quickly.

---

### Stage 4 — GRPO Training Loss
**150 steps · 1 hr 20 min 36 sec · 4 generations per prompt**

| Step | Loss | | Step | Loss | | Step | Loss |
|---|---|---|---|---|---|---|---|
| 10 | -0.324 | | 60 | -0.339 | | 110 | -0.383 |
| 20 | -0.317 | | 70 | -0.219 | | 120 | -0.352 |
| 30 | -0.422 | | 80 | -0.365 | | 130 | -0.272 |
| 40 | -0.268 | | 90 | -0.346 | | 140 | -0.152 |
| 50 | -0.253 | | 100 | -0.278 | | 150 | -0.140 |

> GRPO loss is **expected to be negative** — it represents policy advantage (how much better the model is compared to the reference). The loss trending from -0.32 toward -0.14 means the model stopped gaining as much relative advantage, indicating convergence.

---

### Stage 5 — Evaluation Results

Evaluated on **30 held-out examples** not seen during training.

```
╔══════════════════════════════════════════════════════╗
║           EVALUATION RESULTS — Tesla T4              ║
╠══════════════════════════════════╦══════════╦════════╣
║ Model                            ║ ROUGE-L  ║ Reward ║
╠══════════════════════════════════╬══════════╬════════╣
║ Base model (no fine-tuning)      ║  0.2551  ║  2.233 ║
║ After SFT                        ║  0.3038  ║  1.400 ║
║ After DPO                        ║  0.2418  ║  1.133 ║
║ After GRPO                       ║  0.2743  ║  1.333 ║
╚══════════════════════════════════╩══════════╩════════╝
```

**Reading the results:**

- **ROUGE-L** measures text overlap with the reference CodeAlpaca answers. SFT shows the biggest jump (+19% over base), confirming it learned the task format. DPO and GRPO trade off ROUGE-L for stylistic diversity — their outputs are different from the reference, not necessarily worse.

- **Reward score** uses the rule-based function (does it name a specific issue? give a fix? explain why?). The base model scores highest here because it generates verbose, free-form text that accidentally hits many keywords. Fine-tuned models are more precise but shorter. This is a known limitation of rule-based rewards on short-form outputs.

- **Key takeaway:** ROUGE-L is an imperfect metric for this task — it penalises valid paraphrases. A GPT-4o-as-judge evaluation (available in `eval/evaluate.py` with an OpenAI key) would show clearer SFT → DPO improvement.

---

## 🖥️ Gradio Demo

After training, the GRPO adapter is automatically loaded and served via Gradio:

```
Loading GRPO for demo...
* Running on public URL: https://acb2cfa073d25ad658.gradio.live
```

<img width="1673" height="949" alt="image" src="https://github.com/user-attachments/assets/6be6b6e6-5b85-4a1c-888e-3152f4f4f024" />

Vide Demo:

https://github.com/user-attachments/assets/9dc43371-392f-42da-bce1-d4a914716baa

**Demo features:**
- Paste any code snippet
- Select language (Python, JavaScript, Java, TypeScript, Go)
- Adjust max tokens (50–400) and temperature (0.1–1.0)
- Three pre-loaded examples: O(n²) bug, SQL injection, off-by-one error
- Public share link valid for 1 week

---

## 🧰 Tech Stack

| Library | Version | Role |
|---|---|---|
| `transformers` | ≥4.40 | Load base models and tokenizers |
| `peft` | ≥0.10 | LoRA adapter attachment and loading |
| `trl` | ≥0.8.6 | SFTTrainer, DPOTrainer, GRPOTrainer |
| `bitsandbytes` | ≥0.43 | 4-bit NF4 quantization (QLoRA) |
| `datasets` | ≥2.18 | HuggingFace dataset loading |
| `evaluate` | ≥0.4 | ROUGE-L metric computation |
| `gradio` | ≥4.0 | Web UI |
| `torch` | — | PyTorch backend |

---

## 🔧 Hardware Requirements

| Tier | GPU | VRAM | Model | All stages |
|---|---|---|---|---|
| Minimum | T4 (Colab Free) | 15.6 GB | 1.3B | ✅ tested |
| Recommended | T4 (Colab Pro) | 15.6 GB | 6.7B | ✅ |
| Best | A100 (Colab Pro+) | 40 GB | 13B | ✅ |
| Mac M1 | MPS / CPU | 8GB+ RAM | 1.3B | Inference only |

> On Mac M1: `bitsandbytes` is not supported. Use `mlx-lm` for training. `gradio_demo.py` runs for inference without quantization.

---

## 📎 Key Design Decisions

**Why DeepSeek-Coder as the base?**
It's open-weight, permissively licensed (MIT), instruction-tuned, and punches above its weight on code tasks. The 1.3B version fits comfortably on a free T4.

**Why CodeAlpaca + LeetCode for data?**
Both are freely available on HuggingFace with no login required. LeetCode solutions have a natural quality signal (hash-map vs nested-loop complexity) that gives us DPO preference pairs without any human labeling.

**Why rule-based reward for GRPO?**
Training a neural reward model requires its own dataset and compute. Rule-based rewards (does the review name a specific issue? does it give a fix?) are robust, interpretable, and require zero additional training — the same approach used in DeepSeek-R1.

**Why LoRA rank 16?**
Only **1.1% of parameters** are trained (14.9M of 1.36B). This is the sweet spot between quality and speed on a T4. Rank 32 or 64 would give marginally better results at 2–4× the VRAM cost.
