# Reward Stealing Attack (ReSA)

Official implementation for the paper **"Reward Stealing Attack on Large Language Models"**.

ReSA recovers the latent safety reward from aligned LLMs via **Maximum Entropy IRL** and inverts it to induce harmful generation through **Reward-Guided Decoding**.

---

## ⚡ Quick Start

### 1. Installation

Install all required dependencies:

```bash
pip install -r requirements.txt

```

### 2. Train (Reward Extraction)

Recover the proxy safety reward model from an aligned LLM (treated as expert policy).

```bash
bash run/scripts/train.sh

```

> 
> **Tip:** Check `run/scripts/train.sh` to configure model paths and training parameters.
> 
> 

### 3. Attack (Adversarial Generation)

Perform inference-time attacks using Reward-Guided Decoding.

```bash
bash run/scripts/attack.sh

```

> 
> **Note:** The parameter `--reward_weight` controls attack strength. Empirically, `1.5` balances success rate and coherence.
> 
> 

### 4. Evaluation

Evaluate Attack Success Rate (ASR), Harmful Score (HS), GPT Score (GS) and linguistic quality (PPL-S).

```bash
python run/eval.py

```

---

## 📊 Supported Models

The codebase supports experimentation with the following model families:

* 
**Llama-3** (e.g., Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct)


* 
**Gemma** (e.g., Gemma-7B-Instruct, Gemma-2-27B-Instruct)


* 
**Qwen2.5** (e.g., Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct)
