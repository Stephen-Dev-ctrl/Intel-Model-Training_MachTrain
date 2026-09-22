WARNING! I AM NOT RESPONSIBLE FOR ANYTHING THAT HAPPENS TO YOUR COMPUTER


# Intel-GPU-Targeted Fine-Tuning Engine

A memory-optimised PyTorch and Hugging Face TRL fine-tuning pipeline designed for training vision-language models (You can change the code slightly for training text models; I am currently working on it to upload different versions) on systems with constrained VRAM/RAM (e.g., Intel processors with integrated graphics).

## Features
- **Low Memory Footprint:** Loads weights in `bfloat16` and utilizes `adafactor` optimiser to prevent system RAM explosions.
- **LoRA Adapter Support:** Uses PEFT (`r=8`, `lora_alpha=16`) on `q_proj` and `v_proj` target modules to train less than 1% of total parameters.
- **Gradient Checkpointing:** Reduces intermediate tensor memory accumulation during backpropagation.
- **Auto-Formatting:** Automatically processes structured ChatML dataset prompts.

## Quick Start

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/Stephen-Dev-ctrl/Intel-Model-Training_MachTrain.git
cd Intel-Model-Training_MachTrain
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
