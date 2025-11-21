Training Scripts Overview

This folder contains Python scripts for fine-tuning YOLOv8,YOLOv9 and YOLO-World models on real and synthetic data from the TTA dataset. 

Dataset Classes:
 
✅Controlled real data: 12K Sample Set (7 classes)
Used for full assembly monitoring, this dataset includes all components and their assembly states:
Tidal-turbine  
Body-assembled  
Body-not-assembled  
Hub-assembled  
Hub-not-assembled  
Rear-cap-assembled  
Rear-cap-not-assembled 

✅ Synthetic Data: 4800 Sample Set (7 classes)
Auto-labeled images generated using Unity’s Perception Package, following the same class structure as the 12k sample set:
Body-assembled  
Body-not-assembled  
Hub-assembled  
Hub-not-assembled  
Rear-cap-assembled  
Rear-cap-not-assembled 
Tidal-turbine 
These images are used for ablation studies and domain adaptation experiments.

🛠 Available Scripts

train_YOLO.py => Fine-tunes YOLOv8/v9 on the 12k-sample synthetic dataset, and then on the 12k-sample controlled real dataset
train_YOLOWorld.py => Fine-tunes YOLO-World on the 12k-sample synthetic dataset, and then on the 12k-sample controlled real dataset
