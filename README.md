# DSG-MSAF for Long-Tailed Microfossil Classification

This repository contains the official implementation of the paper:

**"Decaying Synthetic Guidance and Multi‑Scale Adaptive Fusion for Long‑Tailed Microfossil Classification"**  
*Amogne Andualem Ayalew, MD Yahia Shawon, Muhammad Arsalan, Zhengde Zhang, Fazhi Qi*  
(Submitted to *Engineering Applications of Artificial Intelligence*)

---

## 📌 Overview

Automated fossil classification is challenging due to long‑tailed distributions and limited annotated data. We propose:

- **Decaying Synthetic Guidance (DSG)**: A training strategy that progressively reduces the influence of synthetic data during training (from 0.8 to 0.2), enabling early diversity exploration and later refinement on real data.
- **Multi‑Scale Adaptive Fusion (MSAF)**: A lightweight attention module that aggregates hierarchical features from a Vision Transformer backbone to improve fine‑grained representation.
- **Controlled synthetic generation** using FLUX.1 with Canny edge conditioning, guided by class‑specific prompts.
- **CLIP‑based dual evaluation** to filter low‑quality or morphologically inconsistent synthetic images before training.

The framework achieves up to **94.6% accuracy** on a radiolarian microfossil dataset and improves rare‑class F1 scores by **20%** over conventional training.

---

#🗂️ Repository Structure