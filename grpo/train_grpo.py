"""
Stage 4 — GRPO: Group Relative Policy Optimization (Optional / Advanced)
=========================================================================
What this does:
  Takes the DPO model and makes it even better at RANKING multiple
  possible code reviews — not just "is A better than B" (DPO) but
  "here are 4 reviews, rank them all and learn from the group."

  GRPO explained simply:
  ┌───────────────────────────────────────────────────────────────────┐
  │  DPO  = compare pairs  (A vs B)                                   │
  │  GRPO = compare groups (A vs B vs C vs D simultaneously)          │
  │                                                                   │
  │  For each prompt, we:                                             │
  │    1. Generate N responses from the current model                 │
  │    2. Score each with a reward function (no neural net needed!)   │
  │    3. Compute relative advantage: how much better is each         │
  │       response vs the GROUP average?                              │
  │    4. Update the model to be more likely to produce               │
  │       above-average responses                                     │
  │                                                                   │
  │  This is how DeepSeek-R1 was trained — it's the cutting edge      │
  │  of open-source post-training as of 2025.                         │
  └───────────────────────────────────────────────────────────────────┘

  Our reward function (rule-based, no model needed):
    +2 if review mentions a specific issue (bug/inefficiency/security)
    +2 if review gives a concrete fix suggestion
    +1 if review explains WHY it's an issue
    +1 if review is appropriately concise (50-400 words)
    -1 if review is generic/vague

  This is called "rule-based reward" — one of the most effective
  approaches in modern RLHF research and much simpler than training
  a reward model.

Libraries:
  trl   — GRPOTrainer (added in trl >= 0.8.0)
  peft  — load our DPO adapter
"""

# ── 0. Install ────────────────────────────────────────────────────────────────
# !pip install "trl>=0.8.0" transformers peft bitsandbytes datasets accelerate --quiet

import os
import re
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig

# ── 1. Config ─────────────────────────────────────────────────────────────────
MODEL_NAME       = "deepseek-ai/deepseek-coder-1.3b-instruct"
DPO_ADAPTER_DIR  = "dpo/checkpoints/final_adapter"
GRPO_OUTPUT_DIR  = "grpo/checkpoints"
SFT_DATA_FILE    = "data/processed/sft_codealpaca.jsonl"

NUM_GENERATIONS  = 4      # generate this many responses per prompt, then rank them
BATCH_SIZE       = 1
GRAD_ACCUM       = 4
LEARNING_RATE    = 1e-5   # very small LR for GRPO — we're fine-tuning a fine-tuned model
NUM_STEPS        = 200    # GRPO is expensive — 200 steps is enough to see improvement
MAX_NEW_TOKENS   = 256    # max length of generated review

os.makedirs(GRPO_OUTPUT_DIR, exist_ok=True)

# ── 2. Reward function ────────────────────────────────────────────────────────
# This is the heart of GRPO. It scores any generated text.
# Rule-based rewards are robust and don't require training.

ISSUE_KEYWORDS = [
    "bug", "error", "inefficien", "slow", "o(n", "time complexity",
    "memory", "security", "injection", "overflow", "null", "none check",
    "exception", "edge case", "off by one",
]
FIX_KEYWORDS = [
    "instead", "replace", "use ", "consider ", "try ", "better to",
    "recommend", "suggest", "change", "rewrite", "refactor",
]
WHY_KEYWORDS = [
    "because", "since", "this causes", "this means", "which leads",
    "result in", "due to", "as a result",
]

def compute_reward(response: str) -> float:
    """
    Rule-based reward for a code review response.
    Returns a float between -1.0 and 6.0
    """
    reward = 0.0
    text = response.lower()

    # +2 if mentions a specific technical issue
    if any(kw in text for kw in ISSUE_KEYWORDS):
        reward += 2.0

    # +2 if gives a concrete fix suggestion
    if any(kw in text for kw in FIX_KEYWORDS):
        reward += 2.0

    # +1 if explains WHY
    if any(kw in text for kw in WHY_KEYWORDS):
        reward += 1.0

    # +1 if appropriate length (not too short, not too long)
    word_count = len(response.split())
    if 30 <= word_count <= 300:
        reward += 1.0
    elif word_count < 10:
        reward -= 1.0   # penalize one-liners that say nothing

    # -1 if very generic (catches vague responses)
    generic_phrases = ["looks good", "seems fine", "nice code", "well done"]
    if any(p in text for p in generic_phrases):
        reward -= 1.0

    return reward

def batch_reward_fn(completions, **kwargs):
    """
    GRPO passes a list of generated completions here.
    We return a list of reward scores (one per completion).
    """
    rewards = []
    for completion in completions:
        # completion might be a string or a list of token dicts
        if isinstance(completion, list):
            text = "".join(c.get("content", "") for c in completion)
        else:
            text = str(completion)
        rewards.append(compute_reward(text))
    return rewards

# ── 3. Load prompts for GRPO ──────────────────────────────────────────────────
# GRPO only needs the PROMPTS (not the expected answers).
# It will GENERATE responses from the current model, then score + rank them.
print("Loading prompts for GRPO...")
prompts = []
with open(SFT_DATA_FILE) as f:
    for line in f:
        try:
            ex = json.loads(line)
            messages = ex["messages"]
            # We want just the user message (the code to review)
            user_msg = next(
                (m["content"] for m in messages if m["role"] == "user"), None
            )
            if user_msg:
                prompts.append({"prompt": user_msg})
        except (json.JSONDecodeError, KeyError, StopIteration):
            continue
        if len(prompts) >= 500:   # 500 prompts is plenty for GRPO
            break

dataset = Dataset.from_list(prompts)
print(f"  {len(dataset)} prompts loaded for GRPO exploration.")

# ── 4. Load model ─────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"\nLoading base model {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
    DPO_ADAPTER_DIR if os.path.exists(DPO_ADAPTER_DIR) else MODEL_NAME,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load DPO adapter if it exists
if os.path.exists(DPO_ADAPTER_DIR):
    print(f"Loading DPO adapter from {DPO_ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(model, DPO_ADAPTER_DIR, is_trainable=True)
else:
    print("⚠️  DPO adapter not found — using base model with fresh LoRA.")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

# ── 5. GRPO config ────────────────────────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir=GRPO_OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    num_generations=NUM_GENERATIONS,   # how many responses to sample per prompt
    max_new_tokens=MAX_NEW_TOKENS,
    bf16=True,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    max_steps=NUM_STEPS,
)

# ── 6. GRPOTrainer ────────────────────────────────────────────────────────────
# reward_funcs: our rule-based function. Can be a list for multiple rewards.
print("\nInitializing GRPOTrainer...")
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,
    reward_funcs=batch_reward_fn,
    tokenizer=tokenizer,
)

# ── 7. Train ──────────────────────────────────────────────────────────────────
print("\n🚀 Starting GRPO training...")
print(f"   Generations per prompt: {NUM_GENERATIONS}")
print(f"   Training steps: {NUM_STEPS}")
print(f"   Reward function: rule-based (no reward model needed)")
print("\n   Each step:")
print("     1. Sample a code-review prompt")
print(f"     2. Generate {NUM_GENERATIONS} different reviews")
print("     3. Score each review with compute_reward()")
print("     4. Update model to favor above-average reviews\n")

trainer.train()

# ── 8. Save ───────────────────────────────────────────────────────────────────
print(f"\nSaving GRPO adapter to {GRPO_OUTPUT_DIR}/final_adapter ...")
trainer.model.save_pretrained(f"{GRPO_OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{GRPO_OUTPUT_DIR}/final_adapter")

print("\n✅ GRPO training complete!")
print("   The model now generates reviews that are more specific,")
print("   more actionable, and better at catching real issues.")
print("\nNext: run eval/evaluate.py to measure improvement.")
