
from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "/home/student/project_2D/new_yolo_runs/pose_refine2/weights/best.pt"
model = YOLO(MODEL_PATH)

video_path = '/datashare/project/vids_test/4_2_24_A_1.mp4'
OUT_DIR = "/home/student/project_2D/video_pred/refine_model/"
os.makedirs(OUT_DIR, exist_ok=True)

# Read input video to get properties
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define MP4 writer 
mp4_path = os.path.join(OUT_DIR, "4_2_24_A_1_pred.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

# Run YOLO inference frame-by-frame
for result in model.predict(source=video_path, stream=True, conf=0.6, vid_stride=1):
    frame = result.plot()  
    out.write(frame)

cap.release()
out.release()
print(f"Saved MP4: {mp4_path}")