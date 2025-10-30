# 🧠 <span style="color:#e74c3c; font-weight:bold;">Depthwise-Dilated Convolutional Adapters for Medical Object Tracking and Segmentation Using the Segment Anything Model 2 (DD-SAM2)</span>

[![Paper](https://img.shields.io/badge/Paper-DD--SAM2-blue?style=flat-square)](https://www.arxiv.org/abs/2507.14613)
[![arXiv](https://img.shields.io/badge/arXiv-2507.14613-b31b1b?style=flat-square)](https://www.arxiv.org/abs/2507.14613)
[![Journal](https://img.shields.io/badge/Journal-MLST-orange?style=flat-square)](https://iopscience.iop.org/article/10.1088/2632-2153/ae13d1/meta)

---

## 📁 Repository Structure

> Please refer to the following key scripts and modules for implementation details:

- **[`sam2/adapter_ap.py`](https://github.com/apple1986/DD-SAM2/blob/main/sam2/adapter_ap.py)** — contains the core `DD_Adapter` implementation.  
- **`train_xx.py`** — training script for DD-SAM2.  
- **`test_xx.py`** — evaluation and inference script.  
- **`save_seg_result_xx.py`** — script for saving segmentation and tracking results.

---

## 🧩 About DD-SAM2

Our framework builds upon **SAM2** and **MedSAM2**, integrating **depthwise-dilated convolutional adapters** to enhance feature representation for **medical object tracking and segmentation** tasks.

---

## 📚 Citation

If you find this work helpful, please cite our paper along with **SAM2** and **MedSAM2**.

```bibtex
@article{xu2025depthwise,
  title   = {Depthwise-dilated convolutional adapters for medical object tracking and segmentation using the Segment Anything Model 2},
  author  = {Xu, Guoping and Kabat, Christopher and Zhang, You},
  journal = {Machine Learning: Science and Technology},
  year    = {2025}
}
```

**Paper Links:**  
- [arXiv Preprint](https://www.arxiv.org/abs/2507.14613)  
- [MLST Journal Version](https://iopscience.iop.org/article/10.1088/2632-2153/ae13d1/meta)

---

## ⚖️ Acknowledgements

Our implementation is based on:
- [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/segment-anything)  
- [MedSAM2](https://github.com/chaoningzhang/MEDSAM2)

Please cite these works if you use DD-SAM2 in your research.
