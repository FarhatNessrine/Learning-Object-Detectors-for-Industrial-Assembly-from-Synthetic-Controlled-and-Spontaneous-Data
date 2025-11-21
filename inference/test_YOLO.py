import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIGURATION ===
DEVICE = 0  # GPU index
MODEL_PATH = "./best_model_9_controlled_synthetic.pt"  # path to YOLO weights
DATA_YAML = "./data.yaml" # path to data yaml
REAL_IMAGES_DIR = "./Test" # path to test set images

# Automatically create output folder based on model name
MODEL_NAME = os.path.splitext(os.path.basename(MODEL_PATH))[0]
OUTPUT_DIR = os.path.join("/mnt/storage/admindi/home/nfarhat/object_detection/inference_results", MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n Starting evaluation for model: {MODEL_NAME}")
print(f"Results will be saved in: {OUTPUT_DIR}")

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)
model.to(DEVICE)
model.eval()

# === STEP 1: Evaluate model on test set ===
metrics = model.val(
    data=DATA_YAML,
    split="Test" if "Test" in DATA_YAML else "val",
    imgsz=640,
    conf=0.25,
    iou=0.5,
    device=DEVICE,
    verbose=True,
    save_json=True,  # for COCO-style metrics
    project=OUTPUT_DIR,
    name="val_results"
)

print("\n Evaluation Results:")
print(f"Precision: {metrics.box.p.mean():.4f}")
print(f"Recall: {metrics.box.r.mean():.4f}")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5-0.95: {metrics.box.map:.4f}")

# === STEP 2: Confusion Matrix Visualization ===
print("\n Generating confusion matrix...")

cm = metrics.confusion_matrix.matrix if hasattr(metrics, "confusion_matrix") else None

if cm is not None:
    labels = metrics.names
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"Confusion Matrix - {MODEL_NAME}")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f" Confusion matrix saved to: {cm_path}")
else:
    print(" Confusion matrix not available in metrics object.")

# === STEP 3: Inference Visualization on Test Images ===
print("\n Running inference and saving annotated test images...")

for img_file in tqdm(os.listdir(REAL_IMAGES_DIR)):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    img_path = os.path.join(REAL_IMAGES_DIR, img_file)
    results = model.predict(source=img_path, device=DEVICE, conf=0.35, save=False, verbose=False)
    annotated_frame = results[0].plot()
    out_path = os.path.join(OUTPUT_DIR, f"det_{img_file}")
    cv2.imwrite(out_path, annotated_frame)

print(f" Annotated inference images saved to: {OUTPUT_DIR}")

# === STEP 4: Save Metrics Summary to Text File ===
summary_path = os.path.join(OUTPUT_DIR, "metrics_summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Precision: {metrics.box.p.mean():.4f}\n")
    f.write(f"Recall: {metrics.box.r.mean():.4f}\n")
    f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
    f.write(f"mAP@0.5-0.95: {metrics.box.map:.4f}\n")
print(f" Metrics summary saved to: {summary_path}")

print(f"\n Completed evaluation for {MODEL_NAME}. Results available in: {OUTPUT_DIR}")
