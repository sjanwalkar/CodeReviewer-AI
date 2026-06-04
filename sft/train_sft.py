"""
Stage 2 — Supervised Fine-Tuning (SFT) with QLoRA
===================================================
What this does:
  Takes a pretrained base model (DeepSeek-Coder) and teaches it to perform
  code reviews by training on our (code → review) pairs.

  "Supervised" means we show it the correct answer and adjust weights
  so it learns to produce similar answers.

Key concepts explained:
  ┌─────────────────────────────────────────────────────────────┐
  │  QLoRA = Quantization + LoRA                                │
  │                                                             │
  │  Quantization: compress model weights from 32-bit floats    │
  │  to 4-bit integers → model fits in ~4GB instead of ~14GB   │
  │                                                             │
  │  LoRA: instead of updating ALL weights (billions of params) │
  │  we add small "adapter" matrices (millions of params) and   │
  │  only train those. The base model is frozen.                │
  │                                                             │
  │  Together: we can fine-tune a 7B model on a single T4 GPU  │
  └─────────────────────────────────────────────────────────────┘

Libraries:
  transformers  — HuggingFace: loads the model & tokenizer
  peft          — adds LoRA adapter layers to the model
  trl           — SFTTrainer: handles the training loop for us
  bitsandbytes  — enables 4-bit quantization (Linux/Colab only)
  datasets      — loads our .jsonl training data
  torch         — PyTorch: the underlying deep learning framework
  wandb         — logs training metrics to a dashboard (optional)

Hardware tiers:
  Free Colab (T4 15GB)  → MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
  Colab Pro  (T4/V100)  → MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
  Colab Pro+ (A100)     → MODEL_NAME = "codellama/CodeLlama-13b-Instruct-hf"
  M1 Mac                → use mlx_sft.py instead (see notebooks/)
"""

# ── 0. Install (run once in Colab) ────────────────────────────────────────────
# !pip install transformers peft trl bitsandbytes datasets accelerate wandb --quiet

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# ── 1. Config — change MODEL_NAME to match your GPU ──────────────────────────
MODEL_NAME    = "deepseek-ai/deepseek-coder-1.3b-instruct"   # free Colab tier
OUTPUT_DIR    = "sft/checkpoints"
DATA_FILE     = "data/processed/sft_codealpaca.jsonl"
MERGED_DIR    = "sft/merged_model"

# Training hyperparameters
MAX_SEQ_LEN   = 1024     # max tokens per example (longer = more VRAM)
BATCH_SIZE    = 2        # examples per GPU step (lower = less VRAM)
GRAD_ACCUM    = 4        # simulate batch of BATCH_SIZE * GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS    = 1        # 1 epoch is enough to get a working model fast
LORA_R        = 16       # LoRA rank: higher = more params trained, more VRAM
LORA_ALPHA    = 32       # LoRA scaling factor (usually 2× rank)
LORA_DROPOUT  = 0.05

USE_WANDB     = False    # set True + login with: wandb.login() for tracking

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)

# ── 2. Load data ──────────────────────────────────────────────────────────────
print("Loading training data...")
examples = []
with open(DATA_FILE) as f:
    for line in f:
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"  {len(examples)} training examples loaded.")

# Convert list of {messages: [...]} dicts into a HuggingFace Dataset
# SFTTrainer expects a dataset with a column containing formatted text
dataset = Dataset.from_list(examples)
print(f"  Dataset columns: {dataset.column_names}")

# ── 3. Tokenizer ──────────────────────────────────────────────────────────────
# The tokenizer converts text → token IDs (integers) the model understands
print(f"\nLoading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Padding token: needed to make all sequences in a batch the same length
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"   # pad on right for causal LMs

# ── 4. Quantization config (4-bit) ───────────────────────────────────────────
# BitsAndBytesConfig tells the loader to compress weights to 4-bit
# nf4 (NormalFloat4) is the best 4-bit format for LLMs
# compute_dtype: even though weights are stored in 4-bit, actual
#   matrix multiplications happen in bfloat16 for numerical stability
print("Setting up 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # quantize the quantization constants too
)

# ── 5. Load model ─────────────────────────────────────────────────────────────
print(f"Loading model {MODEL_NAME} in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",          # automatically puts layers on GPU/CPU
    trust_remote_code=True,
)
model.config.use_cache = False                    # disable KV cache during training
model.config.pretraining_tp = 1

print(f"  Model loaded. Parameters: {model.num_parameters():,}")

# ── 6. LoRA config ────────────────────────────────────────────────────────────
# target_modules: which layer types to attach LoRA adapters to
# For most transformer models these are the attention projection layers
# r: rank of the adapter matrices — rank 16 is a good balance
print("Attaching LoRA adapters...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention layers
        "gate_proj", "up_proj", "down_proj",        # MLP layers
    ],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output will be something like:
#   trainable params: 4,194,304 || all params: 1,344,798,720 || trainable%: 0.31
# We only train 0.3% of params — that's the power of LoRA!

# ── 7. Format messages into a single string ───────────────────────────────────
# SFTTrainer's chat template support converts the messages list into
# the model's expected format automatically using the tokenizer's
# apply_chat_template method.

def format_chat(example):
    """Apply the model's chat template to the messages list."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

print("Formatting dataset with chat template...")
dataset = dataset.map(format_chat, remove_columns=["messages"])
print(f"  Sample formatted text (first 300 chars):\n  {dataset[0]['text'][:300]}\n")

# ── 8. Training arguments ─────────────────────────────────────────────────────
# SFTConfig extends TrainingArguments with SFT-specific options
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",         # learning rate decreases following a cosine curve
    warmup_ratio=0.03,                  # slowly ramp up LR for the first 3% of steps
    bf16=True,                          # bfloat16 mixed precision (faster, less memory)
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,                 # keep only 2 checkpoints to save disk space
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",          # column name in our dataset
    packing=False,                      # don't pack multiple examples into one sequence
    report_to="wandb" if USE_WANDB else "none",
)

# ── 9. Trainer ────────────────────────────────────────────────────────────────
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# ── 10. Train ─────────────────────────────────────────────────────────────────
print("\n🚀 Starting SFT training...")
print(f"   Model: {MODEL_NAME}")
print(f"   Examples: {len(dataset)}")
print(f"   Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"   LoRA rank: {LORA_R}\n")

trainer.train()

# ── 11. Save adapter ──────────────────────────────────────────────────────────
# We only save the LoRA adapter (small ~20MB), not the full model
print(f"\nSaving LoRA adapter to {OUTPUT_DIR}/final_adapter ...")
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")

print("\n✅ SFT training complete!")
print(f"   Adapter saved to: {OUTPUT_DIR}/final_adapter")
print("\nNext step: run dpo/train_dpo.py to apply preference optimization.")
