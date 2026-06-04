"""
Stage 5 — Evaluation
=====================
What this does:
  Measures whether each stage of training actually improved the model.
  We compare three checkpoints:
    1. Base model (no fine-tuning)
    2. After SFT
    3. After DPO
    4. After GRPO (if run)

  Metrics we use:
  ┌─────────────────────────────────────────────────────────────────┐
  │  ROUGE-L  : measures text overlap with reference answers.       │
  │             Quick proxy for "does it say similar things."       │
  │             Score 0-1; higher is better.                        │
  │                                                                 │
  │  Reward score : run our rule-based reward function from GRPO    │
  │             on the generated reviews. Tells us if the review    │
  │             is specific, actionable, and explains WHY.          │
  │                                                                 │
  │  Win rate  : compare two models head-to-head on the same        │
  │             prompt. Which one scored higher? % of wins.         │
  │                                                                 │
  │  GPT-4o judge (optional): if you have an OpenAI key, we can    │
  │             ask GPT-4o to rate responses 1-5. Expensive but     │
  │             the most reliable automatic metric.                 │
  └─────────────────────────────────────────────────────────────────┘

Libraries:
  evaluate      — HuggingFace: ROUGE and other NLP metrics
  transformers  — load and run each checkpoint
  peft          — load LoRA adapters
  tabulate      — pretty-print comparison tables
"""

# ── 0. Install ────────────────────────────────────────────────────────────────
# !pip install evaluate transformers peft bitsandbytes datasets tabulate rouge_score --quiet

import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import evaluate

# ── 1. Config ─────────────────────────────────────────────────────────────────
MODEL_NAME       = "deepseek-ai/deepseek-coder-1.3b-instruct"
SFT_ADAPTER      = "sft/checkpoints/final_adapter"
DPO_ADAPTER      = "dpo/checkpoints/final_adapter"
GRPO_ADAPTER     = "grpo/checkpoints/final_adapter"
DATA_FILE        = "data/processed/sft_codealpaca.jsonl"

NUM_EVAL_SAMPLES = 50     # evaluate on 50 examples (fast but meaningful)
MAX_NEW_TOKENS   = 200

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")   # optional

# ── 2. Load eval data ─────────────────────────────────────────────────────────
print("Loading evaluation samples...")
eval_examples = []
with open(DATA_FILE) as f:
    for line in f:
        if len(eval_examples) >= NUM_EVAL_SAMPLES:
            break
        try:
            ex = json.loads(line)
            messages = ex["messages"]
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
            ref_ans  = next((m["content"] for m in messages if m["role"] == "assistant"), None)
            if user_msg and ref_ans:
                eval_examples.append({"prompt": user_msg, "reference": ref_ans})
        except (json.JSONDecodeError, KeyError, StopIteration):
            continue

print(f"  {len(eval_examples)} evaluation examples loaded.")

# ── 3. Reward function (same as GRPO stage) ───────────────────────────────────
ISSUE_KEYWORDS = [
    "bug", "error", "inefficien", "slow", "o(n", "time complexity",
    "memory", "security", "injection", "overflow", "null", "none check",
    "exception", "edge case", "off by one",
]
FIX_KEYWORDS   = ["instead", "replace", "use ", "consider ", "try ", "better to",
                   "recommend", "suggest", "change", "rewrite", "refactor"]
WHY_KEYWORDS   = ["because", "since", "this causes", "this means", "which leads",
                  "result in", "due to", "as a result"]

def compute_reward(text: str) -> float:
    reward = 0.0
    lower  = text.lower()
    if any(kw in lower for kw in ISSUE_KEYWORDS): reward += 2.0
    if any(kw in lower for kw in FIX_KEYWORDS):   reward += 2.0
    if any(kw in lower for kw in WHY_KEYWORDS):    reward += 1.0
    wc = len(text.split())
    if 30 <= wc <= 300: reward += 1.0
    elif wc < 10:        reward -= 1.0
    if any(p in lower for p in ["looks good", "seems fine"]): reward -= 1.0
    return reward

# ── 4. Model loading utility ──────────────────────────────────────────────────
def load_model_and_tokenizer(adapter_path=None):
    """Load base model, optionally with a LoRA adapter on top."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tok_path = adapter_path if (adapter_path and os.path.exists(adapter_path)) else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  Loaded adapter: {adapter_path}")
    else:
        print(f"  Using base model (no adapter at {adapter_path})")

    model.eval()
    return model, tokenizer

# ── 5. Generation utility ─────────────────────────────────────────────────────
def generate_review(model, tokenizer, prompt: str) -> str:
    """Generate a code review for the given prompt."""
    # Format as a simple user message (no chat template needed for eval)
    inputs = tokenizer(
        f"Review this code:\n{prompt}\n\nReview:",
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.3,       # low temperature = more deterministic/focused
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ── 6. Evaluate one checkpoint ────────────────────────────────────────────────
rouge_metric = evaluate.load("rouge")

def evaluate_checkpoint(name, adapter_path=None):
    """Run full evaluation for one model checkpoint."""
    print(f"\n{'='*50}")
    print(f"Evaluating: {name}")
    print(f"{'='*50}")

    model, tokenizer = load_model_and_tokenizer(adapter_path)

    predictions = []
    references  = []
    rewards     = []

    for i, ex in enumerate(eval_examples):
        if i % 10 == 0:
            print(f"  [{i+1}/{len(eval_examples)}] generating...", end="\r")
        generated = generate_review(model, tokenizer, ex["prompt"])
        predictions.append(generated)
        references.append(ex["reference"])
        rewards.append(compute_reward(generated))

    # ROUGE scores
    rouge_result = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    avg_reward = sum(rewards) / len(rewards)
    avg_rouge_l = rouge_result["rougeL"]

    print(f"\n  Results for {name}:")
    print(f"    ROUGE-L score : {avg_rouge_l:.4f}  (text overlap with reference)")
    print(f"    Avg reward    : {avg_reward:.4f}  (rule-based quality score)")
    print(f"    Min reward    : {min(rewards):.4f}")
    print(f"    Max reward    : {max(rewards):.4f}")

    # Free GPU memory before loading next model
    del model
    torch.cuda.empty_cache()

    return {
        "name":      name,
        "rouge_l":   avg_rouge_l,
        "reward":    avg_reward,
        "predictions": predictions,
    }

# ── 7. Run evaluation for all checkpoints ─────────────────────────────────────
checkpoints = [
    ("Base model (no fine-tuning)", None),
    ("After SFT", SFT_ADAPTER),
    ("After DPO", DPO_ADAPTER),
]

# Add GRPO only if the checkpoint exists
if os.path.exists(GRPO_ADAPTER):
    checkpoints.append(("After GRPO", GRPO_ADAPTER))

results = []
for name, adapter_path in checkpoints:
    result = evaluate_checkpoint(name, adapter_path)
    results.append(result)

# ── 8. Summary table ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("📊 EVALUATION SUMMARY")
print("="*60)
print(f"{'Model':<30} {'ROUGE-L':>10} {'Reward':>10}")
print("-"*60)
for r in results:
    print(f"{r['name']:<30} {r['rouge_l']:>10.4f} {r['reward']:>10.4f}")
print("="*60)

# Win rate: DPO vs SFT
if len(results) >= 3:
    sft_rewards  = [compute_reward(p) for p in results[1]["predictions"]]
    dpo_rewards  = [compute_reward(p) for p in results[2]["predictions"]]
    dpo_wins     = sum(d > s for d, s in zip(dpo_rewards, sft_rewards))
    win_rate     = dpo_wins / len(sft_rewards) * 100
    print(f"\nDPO win rate vs SFT: {win_rate:.1f}% ({dpo_wins}/{len(sft_rewards)} examples)")
    print("(Win = DPO-generated review scored higher by reward function)")

# ── 9. Optional: GPT-4o as judge ──────────────────────────────────────────────
if OPENAI_API_KEY and len(results) >= 2:
    print("\n--- GPT-4o Judge (optional) ---")
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        judge_scores = {"sft": [], "dpo": []}
        sample_size  = min(10, len(eval_examples))   # GPT-4o costs money, keep it small

        for i in range(sample_size):
            prompt    = eval_examples[i]["prompt"]
            sft_resp  = results[1]["predictions"][i]
            dpo_resp  = results[2]["predictions"][i]

            judge_prompt = f"""You are evaluating code review quality.

Code to review:
{prompt[:300]}

Review A:
{sft_resp[:300]}

Review B:
{dpo_resp[:300]}

Rate each review 1-5 on:
- Specificity (does it name exact issues?)
- Actionability (does it give a concrete fix?)
- Clarity (is it easy to understand?)

Respond with JSON only: {{"review_a": score, "review_b": score}}"""

            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=50,
            )
            text = resp.choices[0].message.content.strip()
            # Parse JSON response safely
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                scores = json.loads(match.group())
                judge_scores["sft"].append(scores.get("review_a", 3))
                judge_scores["dpo"].append(scores.get("review_b", 3))

        if judge_scores["sft"] and judge_scores["dpo"]:
            avg_sft = sum(judge_scores["sft"]) / len(judge_scores["sft"])
            avg_dpo = sum(judge_scores["dpo"]) / len(judge_scores["dpo"])
            print(f"  GPT-4o score — SFT: {avg_sft:.2f}/5 | DPO: {avg_dpo:.2f}/5")

    except ImportError:
        print("  openai package not installed. Run: pip install openai")
    except Exception as e:
        print(f"  GPT-4o judge error: {e}")

print("\n✅ Evaluation complete.")
print("   Results saved to console. Add wandb/MLflow logging for experiment tracking.")
