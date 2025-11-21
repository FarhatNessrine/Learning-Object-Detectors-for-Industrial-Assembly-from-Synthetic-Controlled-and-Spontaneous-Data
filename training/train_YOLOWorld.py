
import os
import time
import shutil
import torch
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch


# === Configuration ===
DEVICE = 0  # GPU
DATASET_DIR = "./yolo_synth_data" # path contains YOLO format data
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml") # This YAML should point to your YOLO format dataset
WEIGHTS_PATH = "./best_model_yoloworld_controlled.pt" 
PROJECT_NAME = "./runs_yoloworld_controlled_synthetic"
PLOTS_DIR = "./analysis_plots_yoloworld_controlled_synthetic"
os.makedirs(PROJECT_NAME, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# === Training Parameters ===
epochs = 100
optimizer = "Adam"
lr0 = 0.0005
run_name = f"e{epochs}_opt{optimizer}_lr{lr0:.4f}"

# === Initialize Model ===
# Load a pretrained YOLO-World model 
model = YOLOWorld(WEIGHTS_PATH)
start_time = time.time()


data_config = dict(
    train=dict(
        yolo_data=[DATA_YAML], # Your custom dataset in YOLO format
        # grounding_data=[
        #     dict(
        #         img_path="path/to/grounding_images",
        #         json_file="path/to/grounding_annotations.json",
        #     ),
        # ],
    ),
    val=dict(yolo_data=[DATA_YAML]), # Using the same for validation for simplicity, adjust if you have a separate val.yaml
)

# === Train Model ===
results = model.train(
    data=data_config,
    epochs=epochs,
    optimizer=optimizer,
    lr0=lr0,
    freeze=10,
    project=PROJECT_NAME,
    name=run_name,
    save=True,
    patience=40,
    dropout=0.3,
    augment=True,
    batch=16,
    workers=2,
    plots=True,
    imgsz=640,
    device=DEVICE,
    trainer=WorldTrainerFromScratch # Specify the custom trainer for YOLO-World
)

# === Record Time ===
train_time = time.time() - start_time

# === Evaluate Model ===
val_metrics = model.val(data=DATA_YAML)
mAP50 = val_metrics.box.map50
print(f"\nValidation mAP@50: {mAP50:.4f}")

# === Save Model ===
model_dir = os.path.join(PROJECT_NAME, run_name, "weights", "best.pt")
best_model_path = os.path.join(PROJECT_NAME, "best_model_yoloworld_controlled_synthetic.pt")
best_run_name = run_name

if os.path.exists(model_dir):
    shutil.copy(model_dir, best_model_path)
    print(f" Best model saved to: {best_model_path}")
else:
    print(f" Model not saved, best.pt not found at: {model_dir}")

# === Save Run Info ===
with open("best_run_yoloworld_controlled_synthetic.txt", "w") as f:
    f.write(best_run_name)

# === Print Summary ===
print("\n Best Model Summary:")
print(f"Run Name       : {best_run_name}")
print(f"mAP@50         : {mAP50:.4f}")
print(f"Training Time  : {train_time:.2f} sec")