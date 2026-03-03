import os
import glob
import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from model_sci import Finetunemodel


# ================= CONFIG =================

INPUT_PATH = r"C:\Users\LagCT\Desktop\Image Pre-processing\test_1"
OUTPUT_PATH = r"C:\Users\LagCT\Desktop\Image Pre-processing\SCI_YOLO_Results"
SCI_WEIGHT = r"C:\Users\LagCT\Desktop\Image Pre-processing\SCI-2022+2025\CVPR\weights\medium.pt"
YOLO_WEIGHT = r"C:\Users\LagCT\Desktop\Image Pre-processing\runs\detect\yolov26_trained\weights\best.pt"

DEVICE = "cpu"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ================= LOAD MODELS =================

def load_models():
    sci_model = Finetunemodel(SCI_WEIGHT).to(DEVICE).eval()
    yolo_model = YOLO(YOLO_WEIGHT).to(DEVICE)
    return sci_model, yolo_model


# ================= WARMUP =================

def warmup(sci_model, yolo_model):
    dummy = torch.zeros(1, 3, 480, 640).to(DEVICE)

    with torch.no_grad():
        for _ in range(3):
            _, r = sci_model(dummy)
            img = (r[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            _ = yolo_model(img, verbose=False)


# ================= PREPROCESS =================

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    target_w, target_h = 640, 480
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_rgb = np.transpose(img_rgb, (2, 0, 1))

    return torch.from_numpy(img_rgb).unsqueeze(0).to(DEVICE)


# ================= SCI =================

def enhance_image(model, input_tensor):
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    _, r = model(input_tensor)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    r = r[0].detach().cpu().numpy()
    r = np.transpose(r, (1, 2, 0))
    r = np.clip(r, 0, 1)
    r = (r * 255).astype(np.uint8)

    return r, (t2 - t1) * 1000


# ================= YOLO =================

def detect_image(model, enhanced_rgb):
    enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    results = model(enhanced_bgr, verbose=False)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    return results, (t2 - t1) * 1000


# ================= SAVE =================

def save_result(results, output_path, filename):
    plotted = results[0].plot()
    save_path = os.path.join(output_path, filename)
    cv2.imwrite(save_path, plotted)


# ================= MAIN =================

def main():
    sci_model, yolo_model = load_models()

    print("Warming up models...")
    warmup(sci_model, yolo_model)
    print("Warmup done.\n")

    test_list = sorted(glob.glob(os.path.join(INPUT_PATH, "**/*.*"), recursive=True))
    test_list = [f for f in test_list if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    print(f"{'File Name':<25} | {'SCI (ms)':<12} | {'YOLO (ms)':<12}")
    print("-" * 55)

    total_sci, total_yolo = 0, 0

    with torch.no_grad():
        for image_path in test_list:
            filename = os.path.basename(image_path)

            input_tensor = preprocess_image(image_path)
            enhanced, sci_time = enhance_image(sci_model, input_tensor)
            results, yolo_time = detect_image(yolo_model, enhanced)

            save_result(results, OUTPUT_PATH, filename)

            total_sci += sci_time
            total_yolo += yolo_time

            print(f"{filename[:24]:<25} | {sci_time:>10.2f} | {yolo_time:>10.2f}")

    n = len(test_list)

    if n > 0:
        print("-" * 55)
        avg_sci = total_sci / n
        avg_yolo = total_yolo / n
        avg_total = avg_sci + avg_yolo

        print(f"{'AVERAGE':<25} | {avg_sci:>10.2f} | {avg_yolo:>10.2f} ms/ảnh")
        print(f"Tổng 1 ảnh: {avg_total:.2f} ms")
        print(f"FPS: {1000/avg_total:.2f}")


if __name__ == "__main__":
    main()