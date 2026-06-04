"""
Stage 3 — Direct Preference Optimization (DPO)
===============================================
What this does:
  Takes the SFT model (which knows HOW to do code reviews) and teaches it
  WHAT a good review looks like vs a bad one.

  DPO explained simply:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Old way (RLHF): train a separate "reward model" to score       │
  │  responses → use reinforcement learning to optimize the LLM     │
  │  → complex, unstable, needs lots of infrastructure              │
  │                                                                 │
  │  DPO (new way): mathematically shows that you can directly      │
  │  optimize the LLM on (chosen, rejected) pairs WITHOUT a         │
  │  separate reward model. Stable, simple, ~10 lines of code.      │
  │                                                                 │
  │  The loss function pushes the model to:                         │
  │    increase P(chosen | prompt)                                  │
  │    decrease P(rejected | prompt)                                │
  └─────────────────────────────────────────────────────────────────┘

  Input:  Our SFT-trained adapter + DPO preference pairs
  Output: A better adapter that prefers specific, helpful reviews
          over vague, generic ones

Libraries:
  trl           — DPOTrainer handles everything
  peft          — loads our LoRA adapter from Stage 2
  transformers  — model/tokenizer loading
  datasets      — loads our preference pairs
"""

# ── 0. Install ────────────────────────────────────────────────────────────────
# !pip install transformers peft trl bitsandbytes datasets accelerate --quiet

import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig

# ── 1. Config ─────────────────────────────────────────────────────────────────
MODEL_NAME       = "deepseek-ai/deepseek-coder-1.3b-instruct"
SFT_ADAPTER_DIR  = "sft/checkpoints/final_adapter"
DPO_OUTPUT_DIR   = "dpo/checkpoints"
DPO_DATA_FILE    = "data/processed/dpo_pairs.jsonl"

LEARNING_RATE    = 5e-5       # DPO uses a lower LR than SFT
BETA             = 0.1        # DPO's key hyperparameter:
                              #   low β  (0.1) = aggressively follow preferences
                              #   high β (0.5) = stay close to SFT model
                              # Think of β as "how much do we trust the preference data"
BATCH_SIZE       = 1
GRAD_ACCUM       = 8
NUM_EPOCHS       = 1
MAX_LENGTH       = 512
MAX_PROMPT_LEN   = 256

os.makedirs(DPO_OUTPUT_DIR, exist_ok=True)

# ── 2. Load DPO dataset ───────────────────────────────────────────────────────
# DPO dataset format: each example needs three fields:
#   prompt   — the question / code to review
#   chosen   — the better response
#   rejected — the worse response
print("Loading DPO preference pairs...")
pairs = []
with open(DPO_DATA_FILE) as f:
    for line in f:
        try:
            ex = json.loads(line)
            # validate all required fields are present
            if all(k in ex for k in ("prompt", "chosen", "rejected")):
                pairs.append(ex)
        except json.JSONDecodeError:
            continue

dataset = Dataset.from_list(pairs)
print(f"  {len(dataset)} preference pairs loaded.")
print(f"  Sample:\n    prompt:   {dataset[0]['prompt'][:80]}...")
print(f"    chosen:   {dataset[0]['chosen'][:80]}...")
print(f"    rejected: {dataset[0]['rejected'][:80]}...")

# ── 3. Quantization (same as SFT) ─────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── 4. Load tokenizer ─────────────────────────────────────────────────────────
print(f"\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    SFT_ADAPTER_DIR if os.path.exists(SFT_ADAPTER_DIR) else MODEL_NAME,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"   # DPO prefers left-padding

# ── 5. Load model + SFT adapter ───────────────────────────────────────────────
# DPO needs TWO copies of the model internally:
#   - policy model    : the one being trained (starts from SFT checkpoint)
#   - reference model : frozen copy of the SFT model (used to compute KL divergence)
#
# DPOTrainer handles creating the reference model automatically when
# we pass is_encoder_decoder=False and don't pass a ref_model.
# It freezes a copy of the policy model as the reference.
print(f"Loading base model {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

# Load the SFT LoRA adapter on top of the base model
if os.path.exists(SFT_ADAPTER_DIR):
    print(f"Loading SFT adapter from {SFT_ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR, is_trainable=True)
else:
    print("⚠️  SFT adapter not found — training DPO from base model.")
    print("    Run sft/train_sft.py first for best results.")
    # Add a fresh LoRA adapter if no SFT checkpoint exists
    from peft import get_peft_model, TaskType
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

# ── 6. DPO training config ────────────────────────────────────────────────────
dpo_config = DPOConfig(
    output_dir=DPO_OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    beta=BETA,
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_PROMPT_LEN,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
)

# ── 7. DPOTrainer ─────────────────────────────────────────────────────────────
# DPOTrainer automatically:
#   1. Creates a frozen reference model (copy of our model)
#   2. For each batch: runs BOTH models on chosen + rejected
#   3. Computes DPO loss: log P_policy(chosen)/P_ref(chosen)
#                                - log P_policy(rejected)/P_ref(rejected)
#   4. Backpropagates through only the policy model's LoRA adapters
print("\nInitializing DPOTrainer...")
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# ── 8. Train ──────────────────────────────────────────────────────────────────
print("\n🚀 Starting DPO training...")
print(f"   β (beta) = {BETA}  (lower = more aggressive preference learning)")
print(f"   Pairs: {len(dataset)}")
print(f"   Epochs: {NUM_EPOCHS}\n")

trainer.train()

# ── 9. Save ───────────────────────────────────────────────────────────────────
print(f"\nSaving DPO adapter to {DPO_OUTPUT_DIR}/final_adapter ...")
trainer.model.save_pretrained(f"{DPO_OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{DPO_OUTPUT_DIR}/final_adapter")

print("\n✅ DPO training complete!")
print("   The model now prefers specific, detailed reviews over vague ones.")
print("\nNext: run grpo/train_grpo.py (optional) or eval/evaluate.py")
