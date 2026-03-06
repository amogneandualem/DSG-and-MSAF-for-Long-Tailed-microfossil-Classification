# DSG-MSAF for Long-Tailed Microfossil Classification

This repository contains the official implementation of the paper:

**"Decaying Synthetic Guidance and Multi‑Scale Adaptive Fusion for Long‑Tailed Microfossil Classification"**  
*Amogne Andualem Ayalew, MD Yahia Shawon, Muhammad Arsalan, Zhengde Zhang, Fazhi Qi*  
(Submitted to *Engineering Applications of Artificial Intelligence*)

---

## 📌 Overview

Automated fossil classification is challenging due to long‑tailed label distributions and limited annotated data. We propose:

- **Decaying Synthetic Guidance (DSG)**: A training strategy that progressively reduces the influence of synthetic data during training (from 0.8 to 0.2), enabling early diversity exploration and later refinement on real data.
- **Multi‑Scale Adaptive Fusion (MSAF)**: A lightweight attention module that aggregates hierarchical features from a Vision Transformer backbone to improve fine‑grained representation.
- **Controlled synthetic generation** using FLUX.1 with Canny edge conditioning and class‑specific prompts.
- **CLIP‑based dual evaluation** to filter low‑quality or morphologically inconsistent synthetic images before training.

Our framework achieves up to **94.6% accuracy** on a radiolarian microfossil dataset and improves rare‑class F1 scores by **20%** over conventional training.

---

## 🗂️ Repository Structure

```
.
├── Data processing and Clip model validation/
│   ├── flux_A800_final.py          # FLUX-based synthetic image generation
│   └── clip_evaluation.ipynb       # CLIP dual evaluation and filtering
├── Model Training/
│   └── DINOV3/
│       └── MSAF_DSG_FINAL/
│           └── train_dinov3_msaf_dsg_final.py   # DINOv3 training with MSAF & DSG
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```


## ⚙️ Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```
Main dependencies:

Python 3.9+

PyTorch 2.0+

torchvision

transformers

diffusers

accelerate

opencv-python

scikit-learn

tqdm

Jupyter (for the notebook)

A GPU with at least 24 GB memory is recommended for training and synthetic image generation.

## 🚀 Usage
1. Synthetic Image Generation
Run the FLUX generation script:

```
cd "Data processing and Clip model validation."
python flux_A800_final.py
```
What this script does:

Loads real images from a specified class.

Extracts Canny edges and blends them with the original image.

Uses FLUX.1 (with Canny conditioning) and class‑specific prompts to generate new specimens.

Saves raw synthetic images to an output folder.

2. CLIP Filtering
Open and run the Jupyter notebook:

```
cd "Data processing and Clip model validation."
jupyter notebook clip_evaluation.ipynb
```
The notebook:

Computes text‑to‑image similarity (with class prompts) and image‑to‑image similarity (with real images) using a CLIP ViT‑B/32 model.

Retains only images that pass both thresholds (≥0.85 and ≥0.80 by default).

Moves accepted images to a curated training folder.

3. Training with DSG and MSAF
Train the DINOv3 backbone with MSAF and DSG:
```
cd "Model Training/DINOV3/MSAF_DSG_FINAL."
python train_dinov3_msaf_dsg_final.py
```
Key features:

Uses a pretrained DINOv3 ViT‑B/14 backbone.

MSAF aggregates [CLS] tokens from the last four layers via learnable attention.

DSG applies an exponential decay schedule to the synthetic loss weight (0.8 → 0.2 over 50 epochs).

Mini‑batches contain 50% real and 50% curated synthetic images.

Logs and checkpoints are saved to logs/ and saved_models/ (excluded from Git).

## 📊 Results

| Backbone       | Real‑Only | Naïve Mixing | DSG (Ours) |
|----------------|-----------|--------------|------------|
| ConvNeXtV2     | 90.2%     | 91.4%        | **92.1%**  |
| DINOv3 + MSAF  | 91.6%     | 93.2%        | **94.6%**  |
| InternImage‑L  | 88.0%     | 89.5%        | **90.3%**  |

- Rare‑class F1 improvement: **+20%** over real‑only training.
- DSG consistently outperforms static synthetic–real mixing across all architectures.

---

## 📝 Citation

If you find this code useful for your research, please cite our paper:

```
bibtex
@article{ayalew2025decaying,
  title={
  Decaying Synthetic Guidance and Multi-Scale Adaptive Fusion for Long-Tailed Microfossil Classification},
  author={Ayalew, Amogne Andualem and Shawon, MD Yahia and Arsalan, Muhammad and Zhang, Zhengde and Qi, Fazhi},
  journal={Engineering Applications of Artificial Intelligence},
  year={2025},
  note={submitted}
}
```
##📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

##🙏 Acknowledgments
This work was supported by the UCAS‑IHEP Computing Center and the High Energy Physics Artificial Intelligence platform (HepAI). We thank all contributors for providing computational resources and technical support.

##📬Contact
For questions or issues, please open an issue on GitHub or contact:

Amogne Andualem Ayalew – amogneandualem@gmail.com

