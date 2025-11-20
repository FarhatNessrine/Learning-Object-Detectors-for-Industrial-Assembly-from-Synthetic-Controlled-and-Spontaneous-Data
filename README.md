# Learning-Object-Detectors-for-Industrial-Assembly-from-Synthetic-Controlled-and-Spontaneous-Data

This repository accompanies the paper **"Detectors-for-Industrial-Assembly-from-Synthetic-Controlled-and-Spontaneous-Data"**. 
It provides a fully reproducible pipeline for training, fine-tuning, and evaluating YOLO-based object detectors using mixed real and synthetic data from a tidal turbine assembly case study.

The proposed TTA-S2R pipeline enables scalable model adaptation from synthetic CAD-generated imagery to real-world industrial footage, supporting controlled and spontaneous scenarios while preserving privacy.


🔧 Key Components

-End-to-end training, evaluation, and inference scripts for YOLOv8, YOLOv9, and YOLO-World

-Configuration files for sequential fine-tuning (synthetic → controlled real)

-Tools for data preparation, including:
    Conversion from Unity Perception to YOLO and COCO formats
    Automated data splitting and augmentation utilities

-Guide for semi-automatic annotation using CVAT with AI-assisted labeling

-Environment setup instructions for full reproducibility

- > ⚠️ **Note:** Full datasets are hosted on Hagging Face due to their size. 
---
## 📁 Repository Structure

```bash
TTA-Sim2Real/
├── training/                 # Training scripts and YOLO configuration
├── evaluation/               # Evaluation scripts and metrics
├── inference/                # Inference scripts           
├── dataset_preparation/      # Annotation format conversion and split scripts
├── cvat_tutorial.md          # Step-by-step CVAT setup and usage guide
├── requirements.txt          # Python requirements for YOLO training/inference
├── LICENSE
└── README.md

📦 Dataset Access

The TTA Dataset (Tidal Turbine Assembly) includes:
     Synthetic data generated from CAD models
     Controlled real data collected under consistent lighting and camera conditions
     Spontaneous real data from real assembly operations for sim-to-real evaluation

⚠️ Due to anonymization requirements, dataset links will be released publicly available for the camera-ready version.

## 📦 Models Checkpoints
To ensure reproducibility and maintain anonymity during the review process, all pretrained and finetuned models used in the **TTA-S2R (Tidal Turbine Assembly – Sim2Real)** study are hosted on an **anonymous Hugging Face repository**.  

These models correspond to the final checkpoints for each training configuration used in the experiments — spanning controlled real data, synthetic domain-randomized data, and synthetic + controlled fine-tuning.  

| Model | Training Regime | Description | Download |
|--------|------------------|-------------|-----------|
| **YOLOv8s** | Pretrained YOLOv8s | Baseline prtetrained on COCO dataset | [Download](https://huggingface.co/NeFr25/TTA-Sim2Real-YOLO-Models/resolve/main/weights/yolov8s.pt) |
| **YOLOv9s** | Pretrained YOLOv9s | Baseline prtetrained on COCO dataset | [Download](https://huggingface.co/NeFr25/TTA-Sim2Real-YOLO-Models/resolve/main/weights/yolov9s.pt) |
| **YOLO-Worldv2** | Pretrained YOLO-Worldv2 | Baseline prtetrained on Objects365 dataset and others  | [Download](https://huggingface.co/NeFr25/TTA-Sim2Real-YOLO-Models/resolve/main/weights/yolov8m-worldv2.pt) |
| **YOLOv8** | Controlled only | Baseline trained on controlled real data | [Download](https://huggingface.co/NeFr25/TTA-Sim2Real-YOLO-Models/resolve/main/weights/best_model_8_controlled.pt) |
| **YOLOv8** | Synthetic only | Trained purely on synthetic domain-randomized images | [Download](https://huggingface.co/NeFr25/TTA-Sim2Real-YOLO-Models/resolve/main/weights/best_model_8_synthetic.pt) |
| **YOLOv8** | Synthetic + Controlled | Fine-tuned mixed-domain model | [Download](https://huggingface.co/NeFr25/TTA-Sim2Real-YOLO-Models/resolve/main/weights/best_model_8_synthetic_controlled.pt) |
| **YOLOv9** | Controlled only | Baseline trained on controlled real data | [Download](https://huggingface.co/NeFr25/TTA-Sim2Real-YOLO-Models/resolve/main/weights/best_model_9_controlled.pt) |
| **YOLOv9** | Synthetic only | Trained purely on synthetic data | [Download](https://huggingface.co/NeFr25/TTA-Sim2Real-YOLO-Models/resolve/main/weights/best_model_9_synthetic.pt) |
| **YOLOv9** | Synthetic + Controlled | Fine-tuned mixed-domain model | [Download](https://huggingface.co/NeFr25/TTA-Sim2Real-YOLO-Models/resolve/main/weights/best_model_9_synthetic_controlled.pt) |
| **YOLO-World** | Controlled only | Trained on controlled real data with text-prompt capability | [Download](https://huggingface.co/NeFr25/TTA-Sim2Real-YOLO-Models/resolve/main/weights/best_model_yoloworld_controlled.pt) |
| **YOLO-World** | Synthetic only | Trained on synthetic data using text-based generalization | [Download](https://huggingface.co/NeFr25/TTA-Sim2Real-YOLO-Models/resolve/main/weights/best_model_yoloworld_synthetic.pt) |
| **YOLO-World** | Synthetic + Controlled | Best-performing model combining both sources | [Download](https://huggingface.co/NeFr25/TTA-Sim2Real-YOLO-Models/resolve/main/weights/best_model_yoloworld_synthetic_controlled.pt) |

---

🚀 Training

Train YOLO-based detectors using the provided configuration files:
    python train.py --model yolov9 --data configs/tta_data.yaml

Supports:
Pretraining on synthetic data
Sequential fine-tuning on controlled real data
Evaluation on spontaneous real images

📊 Evaluation

Evaluate trained models on annotated spontaneous data:
    python evaluate.py --weights best_model.pt --data configs/tta_test.yaml

Metrics include:
    Precision, Recall, mAP@0.5, mAP@0.5–0.95

🔍 Inference

Run inference on videos or images:
    python infer_video.py --weights best_model.pt --source sample_video.mkv

For YOLO-World, text prompts can be used to enhance zero-shot generalization across object colors and assembly states.

✍️ CVAT Annotation
See cvat_tutorial.md for:
