# infer_pose.py
import os
import glob
from pathlib import Path

import torch
from ultralytics import YOLO
import cv2
import numpy as np

# -------- paths --------
WEIGHTS = "/home/student/project_2D/new_yolo_runs/pose_refine2/weights/best.pt"
FRAMES_DIR = "/home/student/project_2D/sampled_frames/frames"
OUT_DIR = "/home/student/project_2D/sampled_frames/frames/annotations"

os.makedirs(OUT_DIR, exist_ok=True)

# -------- model --------
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(WEIGHTS)

# -------- frames --------
frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")))
if not frames:
    raise FileNotFoundError(f"No frames found matching {FRAMES_DIR}/frame_*.jpg")

# Class ID to name mapping
CLASS_NAMES = {
    1: "Needle Holder",
    2: "Tweezers"
}

# -------- helpers --------
def kp_visibility_flags(conf_arr, k_len, th=0.5):
    """Convert per-kp confidences into YOLO-style visibility flags."""
    if conf_arr is None:
        return [2] * k_len
    out = []
    for c in conf_arr:
        if np.isnan(c):
            out.append(0)
        elif c >= th:
            out.append(2)
        else:
            out.append(1)
    return out

def draw_keypoints(img, kps_xy, vis_flags, kp_style="rg"):
    """Draw keypoints with strong contrast and larger circles."""
    H, W = img.shape[:2]
    RED   = (0, 0, 255)
    GREEN = (0, 255, 0)
    CYAN  = (255, 255, 0)
    PURP  = (255, 0, 255)
    ORAN  = (0, 165, 255)
    PALETTE = [RED, GREEN, ORAN, CYAN, PURP]

    radius_fill = 6
    radius_edge = 9
    edge_color = (0, 0, 0)
    edge_thickness = 2

    K = kps_xy.shape[0]
    for i in range(K):
        if vis_flags[i] == 0:
            continue
        x, y = kps_xy[i]
        cx = int(np.clip(x * W, 0, W - 1))
        cy = int(np.clip(y * H, 0, H - 1))

        if kp_style == "rg" and K == 5:
            color = RED if i < 3 else GREEN
        else:
            color = PALETTE[i % len(PALETTE)]

        cv2.circle(img, (cx, cy), radius_edge, edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius_fill, color, thickness=-1, lineType=cv2.LINE_AA)

def draw_bbox(img, box_xyxy, cls_id):
    """Draw a class-colored bounding box with a larger name label above it."""
    COLORS = {
        1: (0, 0, 255),    # red
        2: (0, 255, 0)     # green
    }
    color = COLORS.get(int(cls_id), (255, 255, 255))
    x1, y1, x2, y2 = map(int, box_xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)

    label = CLASS_NAMES.get(int(cls_id), f"Class {cls_id}")
    font_scale = 0.8
    font_thickness = 2
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    y1_txt = max(0, y1 - 10)
    cv2.rectangle(img, (x1, y1_txt - th - 6), (x1 + tw + 6, y1_txt), color, -1)
    cv2.putText(img, label, (x1 + 3, y1_txt - 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                thickness=font_thickness, lineType=cv2.LINE_AA)

# -------- predict --------
results_iter = model.predict(
    source=frames,
    stream=True,
    conf=0.5,
    device=device,
    save=False,
    verbose=False,
)

for res in results_iter:
    h, w = res.orig_shape
    stem = Path(res.path).stem

    img = cv2.imread(res.path)
    if img is None:
        img = res.plot()

    vis_out = os.path.join(OUT_DIR, f"{stem}_pred.jpg")
    lbl_out = os.path.join(OUT_DIR, f"{stem}.txt")
    lines = []

    boxes = res.boxes
    kps = res.keypoints

    if boxes is not None and len(boxes) > 0:
        cxcywh_n = boxes.xywhn.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        kps_xy_n = None
        kps_conf = None
        if kps is not None and kps.xyn is not None:
            kps_xy_n = kps.xyn.cpu().numpy()
            if getattr(kps, "conf", None) is not None:
                kps_conf = kps.conf.cpu().numpy()

        for i in range(cxcywh_n.shape[0]):
            cls_i = clses[i]
            cx, cy, bw, bh = cxcywh_n[i].tolist()

            parts = [str(cls_i), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]

            draw_bbox(img, xyxy[i], cls_i)

            if kps_xy_n is not None:
                xy = kps_xy_n[i]
                K = xy.shape[0]
                conf_i = kps_conf[i] if kps_conf is not None else None
                v_flags = kp_visibility_flags(conf_i, K, th=0.5)

                for (xk, yk), vk in zip(xy, v_flags):
                    parts += [f"{xk:.6f}", f"{yk:.6f}", str(int(vk))]

                draw_keypoints(img, xy, v_flags, kp_style="rg")

            lines.append(" ".join(parts))

    cv2.imwrite(vis_out, img)
    with open(lbl_out, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote: {vis_out} and {lbl_out}")

print("Done.")