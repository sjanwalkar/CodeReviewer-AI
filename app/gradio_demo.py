"""
Stage 5 — Gradio Demo
======================
What this does:
  Launches a simple web UI where you can paste any code snippet and
  get a code review from your fine-tuned model.

  Gradio creates the entire UI with ~20 lines of code. It runs
  locally or in Colab (it auto-creates a public share link).

  We load the BEST available adapter in this priority order:
    GRPO → DPO → SFT → base model

Libraries:
  gradio        — instant web UI for ML models
  transformers  — model loading
  peft          — LoRA adapter loading
  torch         — inference
"""

# ── 0. Install ────────────────────────────────────────────────────────────────
# !pip install gradio transformers peft bitsandbytes accelerate --quiet

import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── 1. Config ─────────────────────────────────────────────────────────────────
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"

# Priority order: use the most trained adapter available
ADAPTER_CANDIDATES = [
    ("GRPO-tuned", "grpo/checkpoints/final_adapter"),
    ("DPO-tuned",  "dpo/checkpoints/final_adapter"),
    ("SFT-tuned",  "sft/checkpoints/final_adapter"),
]

SYSTEM_PROMPT = (
    "You are a senior software engineer performing a thorough code review. "
    "Identify bugs, inefficiencies, and style issues. Suggest specific improvements "
    "with clear explanations. Be concise but precise."
)

# ── 2. Load best available model ──────────────────────────────────────────────
def find_best_adapter():
    for label, path in ADAPTER_CANDIDATES:
        if os.path.exists(path):
            return label, path
    return "Base model", None

print("Loading model...")
adapter_label, adapter_path = find_best_adapter()
print(f"  Using: {adapter_label} ({adapter_path or 'no adapter'})")

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

tok_source = adapter_path if adapter_path else MODEL_NAME
tokenizer  = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if adapter_path:
    model = PeftModel.from_pretrained(model, adapter_path)

model.eval()
print(f"✅ Model ready ({adapter_label})")

# ── 3. Inference function ─────────────────────────────────────────────────────
def review_code(
    code_snippet: str,
    language: str,
    max_tokens: int,
    temperature: float,
):
    """Core function: takes code, returns review string."""
    if not code_snippet.strip():
        return "⚠️  Please paste some code to review."

    # Build the prompt
    user_message = (
        f"Please review the following {language} code and provide specific, "
        f"actionable feedback:\n\n```{language.lower()}\n{code_snippet.strip()}\n```"
    )

    # Apply chat template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback if model doesn't support chat template
        formatted = f"System: {SYSTEM_PROMPT}\n\nUser: {user_message}\n\nAssistant:"

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            do_sample=temperature > 0.1,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    review     = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return review if review else "⚠️  Model returned an empty response. Try adjusting temperature."

# ── 4. Gradio UI ──────────────────────────────────────────────────────────────
# Example snippets to pre-populate the demo
EXAMPLES = [
    [
        "def find_duplicates(arr):\n    duplicates = []\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):\n            if arr[i] == arr[j] and arr[i] not in duplicates:\n                duplicates.append(arr[i])\n    return duplicates",
        "Python",
        200,
        0.3,
    ],
    [
        "def get_user(user_id):\n    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n    return db.execute(query)",
        "Python",
        200,
        0.3,
    ],
    [
        "function calculateTotal(items) {\n  let total = 0;\n  for (let i = 0; i <= items.length; i++) {\n    total += items[i].price;\n  }\n  return total;\n}",
        "JavaScript",
        200,
        0.3,
    ],
]

with gr.Blocks(title="CodeReviewer AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 🔍 CodeReviewer AI
        **Model: {adapter_label}** | Paste code below to get a detailed review.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            code_input = gr.Textbox(
                label="Code to Review",
                placeholder="Paste your code here...",
                lines=15,
            )
            language = gr.Dropdown(
                choices=["Python", "JavaScript", "Java", "C++", "TypeScript", "Go", "Rust"],
                value="Python",
                label="Language",
            )

        with gr.Column(scale=2):
            review_output = gr.Textbox(
                label="Code Review",
                lines=15,
                interactive=False,
            )

    with gr.Row():
        max_tokens_slider = gr.Slider(
            minimum=50, maximum=500, value=200, step=10,
            label="Max response length (tokens)",
        )
        temperature_slider = gr.Slider(
            minimum=0.1, maximum=1.0, value=0.3, step=0.05,
            label="Temperature (lower = more focused)",
        )

    with gr.Row():
        submit_btn = gr.Button("🔍 Review Code", variant="primary")
        clear_btn  = gr.Button("🗑️  Clear")

    submit_btn.click(
        fn=review_code,
        inputs=[code_input, language, max_tokens_slider, temperature_slider],
        outputs=review_output,
    )
    clear_btn.click(
        fn=lambda: ("", ""),
        outputs=[code_input, review_output],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[code_input, language, max_tokens_slider, temperature_slider],
        label="Try these examples (click to load, then click Review Code)",
    )

    gr.Markdown(
        """
        ---
        **Tips:**
        - Lower temperature (0.1–0.3) = more consistent, focused reviews
        - Higher temperature (0.5–0.8) = more varied, creative suggestions
        - The model works best on Python code (most training data)
        """
    )

# ── 5. Launch ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # share=True creates a public link — useful for Colab
    # In Colab, Gradio can't open a local browser, so it generates a
    # temporary public URL (valid for 72 hours)
    demo.launch(
        share=True,          # set False if running locally with a browser
        server_port=7860,
        show_error=True,
    )
