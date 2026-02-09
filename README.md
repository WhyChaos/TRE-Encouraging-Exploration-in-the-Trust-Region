
# TRE: Encouraging Exploration in the Trust Region

[![arXiv](https://img.shields.io/badge/arXiv-2602.03635-b31b1b.svg)](https://arxiv.org/abs/2602.03635)

---

## 📖 Introduction

We propose **Trust Region Entropy (TRE)**, a novel exploration regularization technique for large language models (LLMs). Unlike standard entropy regularization—which indiscriminately spreads probability mass across the entire vocabulary and risks allocating mass to invalid tokens—TRE restricts exploration to a dynamically defined trust region of plausible tokens. This region can be constructed using either:

- **TRE-K**: Fixed top-K approach
- **TRE-P**: Nucleus-based threshold (top-p) approach

<p align="center">
  <img src="figs/tre_diagram_panel_A.png" width="70%" />
</p>

**Figure 1:** Overview of Trust Region Entropy (TRE). Standard entropy regularization (red) flattens the distribution and causes probability mass to leak into the tail. TRE (green) encourages diversity only within the trust region, avoiding tail risk and improving both stability and generation quality.

> 📄 For more details, please refer to the algorithm and mathematical formulation in our [paper](https://arxiv.org/abs/2602.03635).

---

## 🚀 Environment Setup

<details>
<summary>Click to expand</summary>

Our code is based on [verl 0.4.x](https://github.com/verl-project/verl/tree/v0.4.x). Please follow the instructions in the original [`README_verl.md`](README_verl.md) for environment installation.

</details>

---

## 🎯 Training

<details>
<summary>Click to expand</summary>

### Math Task

#### 1. Prepare the dataset
```bash
python experiments/math/data_preprocess/math_dataset.py
```

> **Note:** Please specify the correct model path for both actor and critic in the training scripts.

#### 2. Training Scripts

**Vanilla PPO and Entropy** (by modifying `ENTROPY_COEFF`)
```bash
bash experiments/math/1.5b_ppo.sh
bash experiments/math/7b_ppo.sh
```

**Forking-Tokens Baseline**
```bash
bash experiments/math/1.5b_ppo_forking_tokens.sh
bash experiments/math/7b_ppo_forking_tokens.sh
```

**KL-Cov Baseline**
```bash
bash experiments/math/1.5b_ppo_kl_cov.sh
bash experiments/math/7b_ppo_kl_cov.sh
```

**TRE-K (Ours)**
```bash
bash experiments/math/1.5b_ppo_tre_k.sh
bash experiments/math/7b_ppo_kl_cov.sh
```

**TRE-P (Ours)**
```bash
bash experiments/math/1.5b_ppo_tre_p.sh
bash experiments/math/7b_ppo_tre_p.sh
```

---

### Countdown Task

#### 1. Prepare the dataset
```bash
python experiments/countdown/data_preprocess/countdown.py
```

> **Note:** Please specify the correct model path for both actor and critic in the training scripts.

#### 2. Training Scripts

**Vanilla PPO and Entropy** (by modifying `ENTROPY_COEFF`)
```bash
bash experiments/countdown/1.5b_ppo.sh
bash experiments/countdown/7b_ppo.sh
```

**Forking-Tokens Baseline**
```bash
bash experiments/countdown/1.5b_ppo_forking_tokens.sh
bash experiments/countdown/7b_ppo_forking_tokens.sh
```

**KL-Cov Baseline**
```bash
bash experiments/countdown/1.5b_ppo_kl_cov.sh
bash experiments/countdown/7b_ppo_kl_cov.sh
```

**TRE-K (Ours)**
```bash
bash experiments/countdown/1.5b_ppo_tre_k.sh
bash experiments/countdown/7b_ppo_tre_k.sh
```

**TRE-P (Ours)**
```bash
bash experiments/countdown/1.5b_ppo_tre_p.sh
bash experiments/countdown/7b_ppo_tre_p.sh
```

---

### HH Task

#### 1. Reward Model Preparation

Use [TRL](https://huggingface.co/docs/trl/main/en/reward_trainer) to train a scalar reward model.

**Environment Setup**
```bash
pip install uv
uv pip install trl
```

**Train Reward Model (1.5B)**
```bash
accelerate config
accelerate launch --multi_gpu --num_processes=8 experiments/hh/reward/train_reward_model_1.5b.py
```

**Train Reward Model (7B)**
```bash
pip install deepspeed
accelerate launch --config_file experiments/hh/reward/deepspeed_config_zero3.yaml experiments/hh/reward/train_reward_model_7b.py
```

**Start Reward Model Server**

Set `HH_REWARD_MODEL_PATH` accordingly, then:
```bash
bash experiments/hh/reward/run_with_server.sh
```

**Test Reward Model**
```bash
export NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1,$NO_PROXY
export REWARD_SERVER_URL="http://localhost:8888/predict"
python experiments/hh/reward/test_reward_server.py
```

#### 2. Prepare the dataset
```bash
python experiments/hh/data_preprocess/hh.py
```

> **Note:** Please specify the correct model path for both actor and critic in the training scripts.

#### 3. Training Scripts

**Vanilla PPO and Entropy** (by modifying `ENTROPY_COEFF`)
```bash
bash experiments/hh/1.5b_ppo.sh
bash experiments/hh/7b_ppo.sh
```

**Forking-Tokens Baseline**
```bash
bash experiments/hh/1.5b_ppo_forking_tokens.sh
bash experiments/hh/7b_ppo_forking_tokens.sh
```

**KL-Cov Baseline**
```bash
bash experiments/hh/1.5b_ppo_kl_cov.sh
bash experiments/hh/7b_ppo_kl_cov.sh
```

**TRE-K (Ours)**
```bash
bash experiments/hh/1.5b_ppo_tre_k.sh
bash experiments/hh/7b_ppo_tre_k.sh
```

**TRE-P (Ours)**
```bash
bash experiments/hh/1.5b_ppo_tre_p.sh
bash experiments/hh/7b_ppo_tre_p.sh
```

</details>

---

## 💡 Implementation Details

<details>
<summary>Click to expand</summary>

You can view [this commit](https://github.com/WhyChaos/TRE-Encouraging-Exploration-in-the-Trust-Region/commit/5546077a7599f31ad618a1801337edd3d9986172) to see how we adapted [verl 0.4.x](https://github.com/verl-project/verl/tree/v0.4.x) for our experiments.

### Key Implementation Files

| Method | File | Line |
|--------|------|------|
| **KL-COV Baseline** | [`verl/trainer/ppo/core_algos.py`](verl/trainer/ppo/core_algos.py#L519) | 519 |
| **Forking Tokens Baseline** | [`verl/trainer/ppo/core_algos.py`](verl/trainer/ppo/core_algos.py#L706) | 706 |
| **TRE (Ours)** | [`verl/utils/torch_functional.py`](verl/utils/torch_functional.py#L125) | 125 |

</details>

---

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{huang2026treencouragingexplorationtrust,
      title={TRE: Encouraging Exploration in the Trust Region}, 
      author={Chao Huang and Yujing Lu and Quangang Li and Shenghe Wang and Yan Wang and Yueyang Zhang and Long Xia and Jiashu Zhao and Zhiyuan Sun and Daiting Shi and Tingwen Liu},
      year={2026},
      eprint={2602.03635},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.03635}
}
```