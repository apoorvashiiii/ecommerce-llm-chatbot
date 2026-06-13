# Enterprise E-Commerce LLM Support Agent 🚀

An end-to-end MLOps pipeline designed to fine-tune a lightweight Large Language Model (**Qwen-2.5-1.5B-Instruct**) into a highly specialized customer support agent using Parameter-Efficient Fine-Tuning (**PEFT/LoRA**). 

This agent safely handles complex e-commerce inquiries (refunds, order tracking, returns) in a corporate, professional tone.

## 📊 Project Architecture
* **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Selected for optimization and memory efficiency)
* **Dataset:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (1,000 highly structured instruction-response pairs)
* **Fine-Tuning Method:** LoRA (Low-Rank Adaptation) via Hugging Face `trl` and `peft`
* **Experiment Tracking:** Live logging to **Weights & Biases (W&B)**

---

## 🛠️ Infrastructure & Optimization Engineering
During development, standard 4-bit (`bitsandbytes`) quantization hit structural system RAM deadlocks on resource-constrained cloud hardware (NVIDIA T4). 

**The Solution:** The pipeline was re-architected to bypass quantization layers by operating natively in **FP16 (`torch.float16`) precision**. Because the 1.5B model footprint is compact (~3GB), training was optimized to process a higher throughput batch size seamlessly without any memory leaks or GPU freezes.

### Hyperparameters:
* **LoRA Rank (r):** 16
* **LoRA Alpha:** 32
* **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
* **Learning Rate:** 2e-4
* **Batch Size:** 4 (with Gradient Accumulation steps = 2)

---

## 📈 Training Analytics & Results
The model was fine-tuned for 40 optimization steps. Metrics were streamed dynamically using Weights & Biases:

* **Training Loss:** Dropped sharply from **1.51** (random convergence) down to a stable **0.54** inside 40 steps, proving robust model convergence.
* **Mean Token Accuracy:** Reached a strong peak of **83.08%** in capturing corporate customer support idioms.

---

## 🤖 Model Inference Example
### **Prompt Given:**
> `Instruction: I received a broken item. How can I get a refund or replacement?`

### **Fine-Tuned Response:**
> `I understand that you've encountered a problem with the product and need assistance in getting a refund or replacement. We're here to guide you through the process, ensuring your satisfaction.`
> 
> `To initiate the refund or replacement procedure, please follow these steps:`
> `1. Contact Our Support Team: Reach out to our dedicated support team by calling us at [Customer Support Phone Number]...`

---

## 📂 Repository Layout
* `README.md` - Technical project breakdown.
* `*.ipynb` - Core PyTorch training pipeline and weight merging routines.

*Note: The raw fine-tuned 3GB model binaries are decoupled from this repository and securely hosted on the Hugging Face Hub.*
