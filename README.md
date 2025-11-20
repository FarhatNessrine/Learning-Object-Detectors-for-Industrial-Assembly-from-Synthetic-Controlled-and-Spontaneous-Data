# Learning-Object-Detectors-for-Industrial-Assembly-from-Synthetic-Controlled-and-Spontaneous-Data

This repository accompanies the paper **"Detectors-for-Industrial-Assembly"**. It provides a reproducible pipeline for training and evaluating object detection models using a dataset of real and synthetic images from tidal turbine assembly operations.

Key components of the repository include:
- Scripts for training, evaluation, and inference of object detection models (YOLO-based)
- A CVAT setup for semi-automatic annotation guide
- Scripts for data preparation: data split, Conversion from unity perception annotation to YOLO format
- Instructions for environment setup and dependencies

- > ⚠️ **Note:** Full datasets are hosted on Hagging Face due to their size. 
---
## 📁 Repository Structure

```bash
TTA-Sim2Real/
├── training/                 # Training scripts and YOLO configuration
├── evaluation/               # Evaluation scripts and metrics
├── inference/                # Inference scripts
├── Checkpoints/              # Models checkpoints
├── dataset_preparation/      # Annotation format conversion and split scripts
├── cvat_tutorial.md          # Step-by-step CVAT setup and usage guide
├── requirements.txt          # Python requirements for YOLO training/inference
├── LICENSE
└── README.md

📦 Dataset Access
The complete real and synthetic dataset is hosted on Hugging face:

🔗 Download from: https://huggingface.co/datasets/NeFr25/TTA_Tidal_Turbine_Assembly_Visual_Dataset

🚀 Training
YOLO training scripts are included in the training/ directory.

📊 Evaluation
Run evaluation using the model predictions and ground truth annotations, the YOLO evaluation scripts are included in evaluation/ directory.

🔍 Inference
TH eenference scripts are included in inference/ directory.

✍️ CVAT Annotation
See cvat_tutorial.md for:
