import os
import torch
import cv2
from ultralytics import YOLOWorld
from ultralytics.utils.metrics import ConfusionMatrix
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIGURATION ===
DEVICE = 0  # GPU index
WEIGHTS_PATH = "./best_model_yoloworld_synthetic.pt"
DATA_YAML = "./data.yaml"
REAL_IMAGES_DIR = "./images"
VIDEO_PATH = "./test_video.mkv"
OUTPUT_DIR = "./inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD MODEL ===
model = YOLOWorld(WEIGHTS_PATH)
model.to(DEVICE)
model.eval()

# === STEP 1: Evaluate on annotated real dataset (no prompts) ===
print("\n Evaluating model on annotated test set (without text prompts)...")

# Direct validation using dataset defined in YAML
metrics = model.val(
    data=DATA_YAML,
    split="test" if "test" in DATA_YAML else "val",
    imgsz=640,
    conf=0.60,
    iou=0.5,
    device=DEVICE,
    verbose=True
)

print("\n Evaluation Results:")
print(f"Precision: {metrics.box.p.mean():.4f}")
print(f"Recall: {metrics.box.r.mean():.4f}")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5-0.95: {metrics.box.map:.4f}")

# === STEP 2: Generate confusion matrix ===
print("\n Generating confusion matrix...")

# YOLOWorld internally computes confusion matrices, but we’ll visualize it
cm = metrics.confusion_matrix.matrix if hasattr(metrics, "confusion_matrix") else None

if cm is not None:
    labels = metrics.names
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix - YOLO-World on Real Test Set")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f" Confusion matrix saved to: {cm_path}")
else:
    print(" Confusion matrix not available in metrics object.")

# === STEP 3: Inference on individual test images (optional visualization) ===
print("\n Running visualized inference on test images...")
for img_file in tqdm(os.listdir(REAL_IMAGES_DIR)):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    img_path = os.path.join(REAL_IMAGES_DIR, img_file)
    results = model.predict(source=img_path, device=DEVICE, conf=0.35, save=False, verbose=False)
    annotated_frame = results[0].plot()
    out_path = os.path.join(OUTPUT_DIR, f"det_{img_file}")
    cv2.imwrite(out_path, annotated_frame)

print(f" Annotated test images saved in: {OUTPUT_DIR}")

# === STEP 4: Inference on video with TEXT PROMPTS (zero-shot) ===
print("\n Running video inference with text prompts (zero-shot)...")

text_prompts = [
    "multi-color tidal turbine",
    "blue body assembled",
    "blue body not assembled",
    "black hub assembled",
    "red hub assembled",
    "hub not assembled",
    "blue rear cap assembled",
    "blue rear cap not assembled",
    "tidal turbine",
    "assembled hub",
    "unassembled hub",
    "assembled body",
    "unassembled body",
    "rear cap",
    "blue part",
    "red part",
    "black part",
    "grey part"
]

# Register text prompts for zero-shot inference
model.set_classes(text_prompts)

# Run inference on video
model.predict(
    source=VIDEO_PATH,
    project=OUTPUT_DIR,
    name="video_results",
    save=True,
    show=False,
    conf=0.60,
    device=DEVICE,
)

print(f" Video results saved in: {os.path.join(OUTPUT_DIR, 'video_results')}")
print("\n Complete: Evaluation metrics, confusion matrix, and video inference generated.")
