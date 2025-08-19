# computer-vision-surgical-applications-project

This repository contains the full pipeline for developing and evaluating a 2D keypoint detection system for articulated surgical instruments.  
The project is structured into three phases:  

1. **Synthetic Data Generation** – using BlenderProc to create synthetic annotated datasets of surgical tools.  
2. **Model Training & Inference** – training YOLO-Pose models on synthetic data and running inference on real surgical videos.  
3. **Domain Adaptation / Refinement** – refining the model on unlabeled real surgical videos using pseudo-labeling and self-training.

---

##  Environment Setup
Clone the repository:
```bash
git clone https://github.com/yarden077/computer-vision-surgical-applications-project
```
## Install dependencies:
```bash
pip install -r requirements.txt
```
## Reproducing Results
The workflow is divided into 4 main steps:

1. Synthetic Data Generation

Run the BlenderProc script to generate synthetic datasets with annotations:
```bash
python synthetic_data_generator.py
```
2. Training on Synthetic Data
Train YOLO-Pose on the generated dataset:
```bash
yolo pose train model=yolov8m-pose.pt data=your_dataset.yaml epochs=100 imgsz=640
```
3. Refinement with Real Unlabeled Data
Use pseudo-labeling to adapt the synthetic-trained model to real surgical videos:
```bash
python pseudo_label.py
```
This will generate pseudo-labels for the real dataset and allow you to retrain/refine the model.

4. Inference
•	On images:
```bash
python predict.py
```

•	On videos:
 ```bash
 python video.py --source path/to/video.mp4
```
The predictions (bounding boxes + keypoints) will be saved under runs/pose/predict*/.

##  Final Model Weights
•	Phase 2 (Synthetic-only model): [Download here]()
•	Phase 3 (Refined model): [Download here]()

## Repository Structure
 ```bash
computer-vision-surgical-applications-project/
│── examples/                     # Example predictions and results
│── weights/                      # Folder for saving trained weights
│── predict.py                    # Run inference on images
│── video.py                      # Run inference on videos
│── synthetic_data_generator.py   # Generate synthetic data with BlenderProc
│── requirements.txt              # Environment dependencies
│── README.md                     # Project documentation (this file)
```


