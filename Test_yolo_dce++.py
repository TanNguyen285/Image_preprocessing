import os
import glob
import time
import torch
import numpy as np
import cv2

from PIL import Image
from ultralytics import YOLO
import model_dce


# ==============================
# 1. CONFIG
# ==============================

INPUT_PATH = "test_0"
OUTPUT_PATH = "DCE_YOLO_Results"
ZERO_DCE_WEIGHT = "Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth"
YOLO_WEIGHT = r"C:\Users\LagCT\Desktop\Image Pre-processing\runs\detect\yolov26_trained\weights\best.pt"

DEVICE = "cpu"
SCALE_FACTOR = 12

TARGET_W = 640
TARGET_H = 480

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ==============================
# 2. LOAD MODELS
# ==============================

def load_models():
    DCE_net = model_dce.enhance_net_nopool(12).to(DEVICE)
    DCE_net.load_state_dict(torch.load(ZERO_DCE_WEIGHT, map_location=DEVICE))
    DCE_net.eval()

    yolo_model = YOLO(YOLO_WEIGHT).to(DEVICE)

    return DCE_net, yolo_model


# ==============================
# 3. WARMUP (FIXED)
# ==============================

def warmup(DCE_net, yolo_model):
    # đảm bảo đúng bội số 12
    H = (TARGET_H // SCALE_FACTOR) * SCALE_FACTOR
    W = (TARGET_W // SCALE_FACTOR) * SCALE_FACTOR

    dummy = torch.zeros(1, 3, H, W).to(DEVICE)

    with torch.no_grad():
        for _ in range(3):
            enhanced, _ = DCE_net(dummy)
            img = (enhanced[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            _ = yolo_model(img, verbose=False)


# ==============================
# 4. PREPROCESS
# ==============================

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)

    h, w = img.shape[:2]

    scale = min(TARGET_W / w, TARGET_H / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img_resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)

    x_offset = (TARGET_W - new_w) // 2
    y_offset = (TARGET_H - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

    img_np = canvas.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    # crop về bội số 12
    _, _, H, W = tensor.shape
    H = (H // SCALE_FACTOR) * SCALE_FACTOR
    W = (W // SCALE_FACTOR) * SCALE_FACTOR
    tensor = tensor[:, :, :H, :W]

    return tensor.to(DEVICE)


# ==============================
# 5. DCE INFERENCE
# ==============================

def enhance_image(DCE_net, input_tensor):
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    enhanced, _ = DCE_net(input_tensor)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    return enhanced, (t2 - t1) * 1000


# ==============================
# 6. YOLO INFERENCE
# ==============================

def detect_image(yolo_model, enhanced_tensor):
    img = (enhanced_tensor.squeeze(0)
           .permute(1,2,0)
           .cpu()
           .numpy() * 255).astype(np.uint8)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    results = yolo_model(img, verbose=False)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    return results, (t2 - t1) * 1000


# ==============================
# 7. SAVE
# ==============================

def save_result(results, filename):
    plotted = results[0].plot()
    img = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    Image.fromarray(img).save(os.path.join(OUTPUT_PATH, filename))


# ==============================
# 8. MAIN
# ==============================

def main():
    DCE_net, yolo_model = load_models()

    print("Warming up models...")
    warmup(DCE_net, yolo_model)
    print("Warmup done.\n")

    test_list = sorted(glob.glob(os.path.join(INPUT_PATH, "**/*.*"), recursive=True))
    test_list = [f for f in test_list if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    print(f"{'File Name':<25} | {'DCE++ (ms)':<12} | {'YOLO (ms)':<12}")
    print("-" * 55)

    total_dce, total_yolo = 0, 0

    with torch.no_grad():
        for image_path in test_list:
            filename = os.path.basename(image_path)

            input_tensor = preprocess_image(image_path)

            enhanced, dce_time = enhance_image(DCE_net, input_tensor)
            results, yolo_time = detect_image(yolo_model, enhanced)

            save_result(results, filename)

            total_dce += dce_time
            total_yolo += yolo_time

            print(f"{filename[:24]:<25} | {dce_time:>10.2f} | {yolo_time:>10.2f}")

    n = len(test_list)

    if n > 0:
        print("-" * 55)
        avg_dce = total_dce / n
        avg_yolo = total_yolo / n
        avg_total = avg_dce + avg_yolo

        print(f"{'AVERAGE':<25} | {avg_dce:>10.2f} | {avg_yolo:>10.2f} ms/ảnh")
        print(f"Tổng 1 ảnh: {avg_total:.2f} ms")
        print(f"FPS: {1000/avg_total:.2f}")


if __name__ == "__main__":
    main()