# Gloss-Free Sign Language Translation: An Unbiased Evaluation of Progress in the Field

**Ozge Mercanoglu Sincan, Jian He Low, Sobhan Asasi, Richard Bowden**

**Paper:** [Computer Vision and Image Understanding (CVIU) 2025](https://doi.org/10.1016/j.cviu.2025.104498) | [Preprint (Open Access)](https://openresearch.surrey.ac.uk/esploro/fulltext/journalArticle/Gloss-Free-Sign-Language-Translation-An-Unbiased/991040066602346?repId=12224845200002346&mId=13224845190002346&institution=44SUR_INST)

---

Sign Language Translation aims to convert sign language videos into spoken language text. Reported improvements in the literature can stem from many factors beyond algorithmic novelty, e.g., backbones, preprocessing, training schedules, and evaluation conventions. This repository evaluates five recent gloss-free Sign Language Translation (SLT) methods — **GFSLT-VLP**, **SignCL**, **Sign2GPT**, **Fla-LLM**, and **C2RL** — by reimplementing the key innovations in a unified codebase under consistent conditions.


The following key contributions are implemented:

| Method | Key idea | Paper | Code |
|--------|----------|-----------|-----------|
| **GFSLT-VLP (Zhou et al., 2023)** | Visual-language pretraining | [ICCV'23](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_Gloss-Free_Sign_Language_Translation_Improving_from_Visual-Language_Pretraining_ICCV_2023_paper.html) | [Github](https://github.com/zhoubenjia/GFSLT-VLP) |
| **SignCL (Ye et al., 2024)** | Contrastive learning on adjacent frames | [NeurIPS'24](https://www.proceedings.com/content/079/079017-3411open.pdf) | [Github](https://github.com/JinhuiYE/SignCL) |
| **Sign2GPT (Wong et al., 2024)*** | Pseudo-gloss pretraining | [ICLR'24](https://iclr.cc/virtual/2024/poster/18847) | [Github](https://github.com/ryanwongsa/Sign2GPT) |
| **Fla-LLM (Chen et al., 2024)** | Lightweight translation (Light-T) in pretraining | [LREC-COLING'24](https://aclanthology.org/2024.lrec-main.620/)  | |
| **C2RL (Chen et al., 2025)** | Combine [CiCO loss](https://github.com/FangyunWei/SLRT/tree/main/CiCo) + Light-T |[IEEE TCSVT'25](https://ieeexplore.ieee.org/document/10933970) |  |

<figure>
  <img src="./assets/overview.jpg" alt="Pipeline diagram" width="700"/>
  <figcaption>Overview and training objectives of compared methods.</figcaption>
</figure>

> *Sign2GPT uses a fundamentally different architecture (DINOv2, XGLM, LoRA adapters) compared to the mBART-based framework used for the other methods. Adapting its pseudo-gloss pretraining strategy into our unified framework yielded significantly lower performance. Implementation is not included in this repository. Please refer to the [official codebase](https://github.com/ryanwongsa/Sign2GPT).

---

## Current Project: iSign Workflow

> **Note on Repository Scope:** This repository is based on the benchmark codebase for gloss-free Sign Language Translation and retains original support code and baseline configs for **Phoenix-2014T** and **CSL-Daily**. However, the current active project workflow focuses on the **iSign** dataset (English sign language translation).

### External Dependencies & Storage Policy
To ensure repository portability and conform to GitHub file size limitations, **no large data or model binaries are tracked in GitHub**:
- **iSign LMDB Dataset:** External dependency stored locally (NOT tracked in Git).
- **Prepared mBART Assets (`MBart_trimmed`, `mytran`):** External model artifacts prepared locally (NOT tracked in Git).
- **Training Checkpoints:** Stored in an external directory outside the repository (NOT tracked in Git).

### Environment Requirements
- **Python:** 3.9 (tested on Python 3.9.25)
- **PyTorch:** 2.1.2+cu118
- **Torchvision:** 0.16.2+cu118
- **Transformers:** 4.32.0
- **Tokenizers:** 0.13.3
- **Hugging Face Hub:** 0.23.4
- **SentencePiece:** 0.1.97
- **CUDA:** 11.8

Dependencies can be installed via `pip install -r assets/requirements.txt`.

### Required Environment Variables
The configuration dynamically resolves paths using environment variables:
- `ISIGN_LMDB_ROOT`: Root directory containing the iSign LMDB dataset and `labels/` subdirectory (`labels.train`, `labels.dev`, `labels.test`).
- `MBART_MODELS_ROOT`: Directory containing prepared mBART assets (`MBart_trimmed` and `mytran`).

**Example for current machine (Windows PowerShell):**
```powershell
$env:ISIGN_LMDB_ROOT = "D:/iSign_LMDB"
$env:MBART_MODELS_ROOT = "E:/FINAL YEAR PROJECT/GLOSS FREE SLT/mbart_models"
```

**Example for Linux / Bash:**
```bash
export ISIGN_LMDB_ROOT="/path/to/iSign_LMDB"
export MBART_MODELS_ROOT="/path/to/mbart_models"
```

### Configuration
The portable iSign configuration is located at:
- `configs/isign/config1.yaml`

Always explicitly pass `--config configs/isign/config1.yaml` when running iSign training or evaluation workflows.

### iSign Smoke Tests
Before running full training, verify environment setup and model construction using the smoke tests:

1. **Model & Architecture Smoke Test:**
   ```bash
   python scripts/isign_model_smoke_test.py
   ```
2. **One-Sample End-to-End Pipeline Smoke Test:**
   ```bash
   python scripts/isign_one_sample_smoke_test.py
   ```

### Stage 1: Vision–Language Pretraining (VLP) on iSign
To launch Stage 1 VLP training with iSign:
```bash
python train_vlp.py \
  --config configs/isign/config1.yaml \
  --model_type gfslt \
  --output_dir "E:/iSign_Checkpoints/vlp" \
  --batch-size 8 \
  --checkpoint-interval 250
```

### Checkpointing & Resuming Training
- **Checkpoint Saving:** Mid-epoch checkpoints are saved periodically via `--checkpoint-interval <N>` (e.g. every 250 completed batches), updating `latest_checkpoint.pth` and saving epoch snapshots to `--output_dir`.
- **Resuming:** To resume an interrupted training session, pass `--resume`:
  ```bash
  python train_vlp.py \
    --config configs/isign/config1.yaml \
    --model_type gfslt \
    --output_dir "E:/iSign_Checkpoints/vlp" \
    --resume "E:/iSign_Checkpoints/vlp/latest_checkpoint.pth"
  ```
  The checkpoint manager restores model parameters, optimizer, learning rate schedulers, epoch counter, batch step, sampler permutation, and RNG state.

---

## Installation

**Prerequisites:** Python 3.9, a CUDA-enabled GPU.

### Option A — Conda

```bash
# 1. Create environment
conda create -n sltbaselines python=3.9 -y
conda activate sltbaselines

# 2. Install PyTorch (adjust the CUDA version to match your driver)
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# 3. Install remaining dependencies
pip install -r assets/requirements.txt
```

### Option B — Docker

```bash
# Build the image (from the repository root)
docker build -t sltbaselines:cuda11.7 -f Code/assets/Dockerfile Code/assets

# Run with GPU access
docker run --gpus all -it --shm-size=24gb \
  -v "$(pwd)"/Code:/Code \
  -v "$(pwd)"/mbart_models:/mbart_models \
  -w /Code \
  sltbaselines:cuda11.7 \
  /bin/bash
```

---

## Data preparation steps
1. Download datasets.

2. Build LMDB databases for fast data loading (optional):

   ```bash
   python scripts/create_lmdb.py \
     --src_dir /path/to/Phoenix-2014T/frames \
     --dst_path /path/to/Phoenix_lmdb
   ```

3. Prepare mBART models. The configs expect a trimmed mBART tokenizer and model under `../mbart_models/`. Refer to [pretrain_models](https://github.com/zhoubenjia/GFSLT-VLP/blob/main/pretrain_models/README.md).

---

## Training

Training follows a two-stage pipeline:
1. **Stage 1 — Vision–Language Pretraining (VLP):** Learn aligned visual and textual representations.
2. **Stage 2 — SLT Fine-tuning:** Fine-tune the encoder–decoder for translation using the Stage 1 checkpoint.

### Stage 1: Vision–Language Pretraining

GFSLT, CICO, and SignCL use `train_vlp.py`, Fla-LLM and C2RL use `train_slt.py` for pretraining.

```bash
# GFSLT
python train_vlp.py --config ./configs/phoenix/config1.yaml --model_type gfslt

# CICO (cross-lingual contrastive learning)
python train_vlp.py --config ./configs/phoenix/config1.yaml --model_type cico

# SignCL
python train_vlp.py --config ./configs/phoenix/config1.yaml --model_type signcl

# Fla-LLM  (lightweight translation: gfslt model without pretraining)
python train_slt.py --config ./configs/phoenix/config1.yaml --model_type gfslt --finetune=""

# C2RL  (CiCO loss + lightweight translation)
python train_slt.py --config ./configs/phoenix/config1.yaml --model_type c2rl --finetune=""
```

### Stage 2: SLT Fine-tuning

Fine-tune from the best Stage 1 checkpoint using `train_slt.py`:

```bash
# GFSLT
python train_slt.py --config ./configs/phoenix/config1.yaml \
  --finetune <CHECKPOINT_PATH> --model_type gfslt

# CICO
python train_slt.py --config ./configs/phoenix/config1.yaml \
  --finetune <CHECKPOINT_PATH> --model_type cico

# SignCL
python train_slt.py --config ./configs/phoenix/config1.yaml \
  --finetune <CHECKPOINT_PATH> --model_type signcl

# Fla-LLM  (frozen visual encoder, 12-layer mBART decoder)
python train_slt.py --config ./configs/phoenix/flallm_config1_stage2.yaml \
  --finetune <CHECKPOINT_PATH> --model_type flallm \
  --frozenFeatureExtractor

# C2RL
python train_slt.py --config ./configs/phoenix/flallm_config1_stage2.yaml \
  --finetune <CHECKPOINT_PATH> --model_type c2rl \
  --frozenFeatureExtractor
```


### Model-specific hyperparameters

| Model | Parameter | 
|-------|-----------|
| SignCL | `--zipf_factor`, `--signcl_warmup_epochs`, `--signcl_decay_rate`   |
| Fla-LLM / C2RL | `--frozenFeatureExtractor`, `--lr_llm_adapter`  |
---

## Evaluation

```bash
python train_slt.py \
  --config ./configs/phoenix/config1.yaml \
  --model_type gfslt \
  --eval \
  --resume <CHECKPOINT_PATH>
```

**Dataset-specific conventions** (following prior work):
- **Phoenix-2014T:** Dot (` .`) is appended to each sentence.
- **CSL-Daily:** Character-level BLEU.

---

## Results

All models are trained on a **single NVIDIA A100 GPU** with batch size 8. All models are trained and evaluated three times with random seeds (0, 42, 100). Our analysis reveals that many reported performance gains diminish under consistent experimental setups, demonstrating that implementation details significantly impact results.

<figure>
  <img src="./assets/results.png" alt="Results" width="700"/>
  <figcaption>Reported vs. reproduced results on Phoenix-2014T and CSL-Daily.</figcaption>
</figure>

We provide test set predictions and reference translations for the best-performing run of each model on both datasets in the [`outputs/`](outputs/) directory. Each subdirectory contains the predicted sentences (`tmp_pres.txt`), ground truth references (`tmp_refs.txt`), and evaluation scores (`scores.txt`).

---

## Citation

```bibtex
@article{sincan2025gloss,
  title   = {Gloss-free Sign Language Translation: An unbiased evaluation of progress in the field},
  journal = {Computer Vision and Image Understanding},
  volume  = {261},
  pages   = {104498},
  year    = {2025},
  issn    = {1077-3142},
  doi     = {https://doi.org/10.1016/j.cviu.2025.104498},
  author  = {Ozge Mercanoglu Sincan and Jian He Low and Sobhan Asasi and Richard Bowden}
}
```

---

## Acknowledgements

This codebase is built upon [GFSLT-VLP](https://github.com/zhoubenjia/GFSLT-VLP). We thank all authors who 
released their code, which served as useful references for our reimplementations, and the C2RL authors for kindly sharing part of their implementation.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
