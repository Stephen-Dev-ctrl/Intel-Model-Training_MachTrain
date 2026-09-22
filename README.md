WARNING! I AM NOT RESPONSIBLE FOR ANYTHING THAT HAPPENDS TO YOUR COMPUTER


# Intel-GPU-Targeted Fine-Tuning Engine

A memory-optimized PyTorch and Hugging Face TRL fine-tuning pipeline designed for training vision-language models on systems with constrained VRAM/RAM (e.g., Intel processors with integrated graphics).

## Features
- **Low Memory Footprint:** Loads weights in `bfloat16` and utilizes `adafactor` optimizer to prevent system RAM explosions.
- **LoRA Adapter Support:** Uses PEFT (`r=8`, `lora_alpha=16`) on `q_proj` and `v_proj` target modules to train less than 1% of total parameters.
- **Gradient Checkpointing:** Reduces intermediate tensor memory accumulation during backpropagation.
- **Auto-Formatting:** Automatically processes structured ChatML dataset prompts for Qwen2.5-VL.

## Quick Start

### 1. Clone Repository & Setup Environment
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
