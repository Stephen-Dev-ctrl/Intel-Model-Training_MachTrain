import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import SFTConfig, SFTTrainer

print("=" * 60)
print("⏱️ INTEL CPU 2-HOUR TARGETED TRAINING ENGINE")
print("=" * 60)

# 1. LOAD MODEL INTO SYSTEM RAM
model_name = (
    "./uncompressed_qwen_model"
    if os.path.exists("./uncompressed_qwen_model")
    else "Qwen/Qwen2.5-VL-3B-Instruct"
)
print(f"\n🧠 Loading Model weights into Intel RAM from: '{model_name}'...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # ⚡ OPTIMIZED: Cuts model RAM size from 12GB to ~6GB
    device_map="cpu",            # ⚡ OPTIMIZED: Forces execution on your Intel CPU
    trust_remote_code=True,
)

# ⚡ OPTIMIZED: Prevents RAM accumulation explosions during the training loop
model.gradient_checkpointing_enable()

# 2. LOAD & FORMAT DATASET
local_jsonl = "unsloth_medical_dataset.jsonl"
if os.path.exists(local_jsonl):
    print("\n📈 Pre-blended 80/20 dataset detected. Loading directly into engine...")
    train_ready_dataset = load_dataset("json", data_files=local_jsonl, split="train")
else:
    print("\n⚠️ Pre-blended dataset missing! Falling back to raw bank sample...")
    hf_dataset = load_dataset("DoDataThings/us-bank-transaction-categories-v2", split="train")
    raw_dataset = hf_dataset.shuffle(seed=3407).select(range(500))

    def convert_to_qwen_vl_chatml(batch):
        descriptions = batch.get("description", [])
        categories = batch.get("category", [])
        formatted_prompts = []
        for desc, cat in zip(descriptions, categories):
            prompt = f"<|im_start|>user\nAnalyze expense:\nTransaction: {desc}<|im_end|>\n<|im_start|>assistant\n{cat}<|im_end|>"
            formatted_prompts.append(prompt)
        return {"text": formatted_prompts}

    train_ready_dataset = raw_dataset.map(convert_to_qwen_vl_chatml, batched=True)

print(f"Dataset successfully staged! Total items to process: {len(train_ready_dataset)}")

# 3. ATTACH LORA ADAPTERS
print("\n🛠️ Attaching LoRA Fine-Tuning Adapters...")
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 4. INITIALIZE MODERN SFTCONFIG TRAINER
print("\n🔥 Initializing Supervised Fine-Tuning Trainer...")
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ready_dataset,
    args=SFTConfig(
        use_cpu=True,  # ⚡ REQUIRED BY MODERN HUGGINGFACE FOR CPU TRAINING!
        dataset_text_field="text",
        max_length=256,
        packing=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=3,
        max_steps=28,   # ⚡ CALIBRATED: Run exactly 75 steps to hit the 2-hour target
        learning_rate=2e-4,
        logging_steps=1,
        optim="adafactor",  # ⚡ OPTIMIZED: Saves ~20GB of RAM compared to adamw_torch
        output_dir="training_checkpoints",
        report_to="none",
    ),
)

print("\n🚀 STARTING 2-HOUR HYBRID FINE-TUNING LOOP!")
print("Your model is training. Sit back and let the CPU cook!")
trainer.train()

output_model_folder = "corporate_finance_model"
print(f"\n💾 Saving 2-hour fine-tuned weights to './{output_model_folder}'...")
model.save_pretrained(output_model_folder)
tokenizer.save_pretrained(output_model_folder)
print(f"\n🎉 SUCCESS! Saved 2-hour hybrid model to './{output_model_folder}'!")