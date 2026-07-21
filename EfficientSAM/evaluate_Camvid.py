import os
import time
import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from efficient_sam.build_efficient_sam import (
    build_efficient_sam_vits
)

from prompt_utils import (
    generate_box_prompt,
    generate_click_prompt,
    build_click_prompt,
    select_random_object,
    resize_prediction
)

from metrics import (
    compute_iou,
    compute_dice,
    compute_precision,
    compute_recall
)

# =====================================================
# Configuration
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)

MODEL_PATH = (
    "/workspace/EfficientSAM/weights/"
    "efficient_sam_vits.pt"
)

ORIGINAL_DIR = (
    "/workspace/EfficientSAM/datasets/camvid/test"
)

ENHANCED_DIR = (
    "/workspace/EfficientSAM/datasets/camvid/test_enhanced"
)

LABEL_DIR = (
    "/workspace/EfficientSAM/datasets/camvid/test_labels"
)

OUTPUT_DIR = (
    "/workspace/EfficientSAM/outputs/camvid"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

transform = transforms.ToTensor()


# =====================================================
# Load EfficientSAM
# =====================================================

print("Loading EfficientSAM ViT-S...")

model = build_efficient_sam_vits()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

if "model" in checkpoint:
    checkpoint = checkpoint["model"]

model.load_state_dict(
    checkpoint
)

model.to(DEVICE)
model.eval()

print("Model loaded.")

# =====================================================
# GPU Warm-up
# =====================================================

dummy_image = torch.zeros(
    (1,3,512,512),
    device=DEVICE
)

dummy_points = torch.tensor(
    [[[
        [10,10],
        [100,100]
    ]]],
    dtype=torch.float,
    device=DEVICE
)

dummy_labels = torch.tensor(
    [[[2,3]]],
    dtype=torch.float,
    device=DEVICE
)

with torch.no_grad():
    model(
        dummy_image,
        dummy_points,
        dummy_labels
    )


    

def load_dataset(image_dir):

    files = sorted([
        f
        for f in os.listdir(image_dir)
        if f.endswith(".png")
    ])

    return files



# =====================================================
# Load One Sample
# =====================================================

def load_sample(image_dir, filename):

    image = np.array(

        Image.open(

            os.path.join(
                image_dir,
                filename
            )

        ).convert("RGB")

    )

    label_name = filename.replace(
        ".png",
        "_L.png"
    )

    label = np.array(

        Image.open(

            os.path.join(
                LABEL_DIR,
                label_name
            )

        ).convert("RGB")

    )

    return image, label


# =====================================================
# EfficientSAM Prediction
# =====================================================

def predict_mask(
        image,
        input_points,
        input_labels
):


    image_tensor = transform(
        image
    ).to(DEVICE)

    input_points = input_points.to(DEVICE)
    input_labels = input_labels.to(DEVICE)

    # -------------------------------------
    # Accurate latency measurement
    # -------------------------------------

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        predicted_logits, predicted_iou = model(
            image_tensor.unsqueeze(0),
            input_points,
            input_labels
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    latency_ms = (end - start) * 1000

    # -------------------------------------
    # EfficientSAM output format
    # -------------------------------------

    if predicted_logits.dim() == 5:
        predicted_logits = predicted_logits.squeeze(1)

    if predicted_iou.dim() == 3:
        predicted_iou = predicted_iou.squeeze(1)

    # -------------------------------------
    # Choose highest confidence mask
    # -------------------------------------

    best_index = torch.argmax(
        predicted_iou,
        dim=-1
    )[0]

    pred_prob = torch.sigmoid(
        predicted_logits[0, best_index]
    )

    pred_mask = (
        pred_prob.cpu().numpy() > 0.5
    ).astype(np.uint8)

    confidence = float(
        predicted_iou[0, best_index]
        .cpu()
        .item()
    )

    return (
        pred_mask,
        latency_ms,
        confidence
    )



# =====================================================
# Box Prompt Prediction
# =====================================================

def evaluate_box(
        image,
        gt_mask
):

    points, labels = generate_box_prompt(
        gt_mask
    )

    if points is None:
        return None, 0

    pred, latency, _ = predict_mask(
        image,
        points,
        labels
    )

    pred = resize_prediction(
        pred,
        gt_mask
    )

    return pred, latency




# =====================================================
# One Click Prediction
# =====================================================

def evaluate_click1(
        image,
        gt_mask
):

    points = generate_click_prompt(
        gt_mask,
        num_points=1
    )

    if points is None:
        return None, 0

    input_points, input_labels = build_click_prompt(
        points
    )

    pred, latency, _ = predict_mask(
        image,
        input_points,
        input_labels
    )

    pred = resize_prediction(
        pred,
        gt_mask
    )

    return pred, latency




# =====================================================
# Evaluate Dataset
# =====================================================

def evaluate_dataset(
        dataset_name,
        image_dir
):

    print("\n===================================")
    print(f"Evaluating {dataset_name}")
    print("===================================")

    files = load_dataset(image_dir)

    results = []

    total_box_iou = 0
    total_click1_iou = 0

    total_box_dice = 0
    total_box_precision = 0
    total_box_recall = 0
    total_box_latency = 0

    total_click1_dice = 0
    total_click1_precision = 0
    total_click1_recall = 0
    total_click1_latency = 0

    valid_images = 0

        # =====================================================
    # Iterate Images
    # =====================================================

    for filename in tqdm(
        files,
        desc=dataset_name
    ):

        try:

            # -----------------------------
            # Load image and label
            # -----------------------------
            image, label = load_sample(
                image_dir,
                filename
            )

            h, w = image.shape[:2]

            # -----------------------------
            # Select object for evaluation
            # -----------------------------
            gt_mask, class_color = select_random_object(
                label
            )

            # Skip image if no valid object
            if gt_mask is None:
                continue

            # =================================================
            # Box Prompt
            # =================================================
            pred_box, latency = evaluate_box(
                image,
                gt_mask
            )

            if pred_box is None:
                continue

            box_iou = compute_iou(gt_mask, pred_box)
            box_dice = compute_dice(gt_mask, pred_box)
            box_precision = compute_precision(gt_mask, pred_box)
            box_recall = compute_recall(gt_mask, pred_box)
            # =================================================
            # One-Click Prompt
            # =================================================
            pred_click1, latency1 = evaluate_click1(
                image,
                gt_mask
            )

            if pred_click1 is None:
                continue

            click1_iou = compute_iou(gt_mask, pred_click1)
            click1_dice = compute_dice(gt_mask, pred_click1)
            click1_precision = compute_precision(gt_mask, pred_click1)
            click1_recall = compute_recall(gt_mask, pred_click1)
       

            # =================================================
            # Accumulate Results
            # =================================================
            total_box_iou += box_iou
            total_box_dice += box_dice
            total_box_precision += box_precision
            total_box_recall += box_recall
            total_box_latency += latency

            total_click1_iou += click1_iou
            total_click1_dice += click1_dice
            total_click1_precision += click1_precision
            total_click1_recall += click1_recall
            total_click1_latency += latency1

            valid_images += 1

            # =================================================
            # Save Per-image Results
            # =================================================
            results.append({
                "Image": filename,

                "Box IoU": box_iou * 100,
                "Box Dice": box_dice * 100,
                "Box Precision": box_precision * 100,
                "Box Recall": box_recall * 100,
                "Box Latency (ms)": latency,

                "1-Click IoU": click1_iou * 100,
                "1-Click Dice": click1_dice * 100,
                "1-Click Precision": click1_precision * 100,
                "1-Click Recall": click1_recall * 100,
                "1-Click Latency (ms)": latency1

            })
        except Exception as e:

            print(
                f"Skip {filename}: {e}"
            )

            continue


    # =====================================================
    # Dataset Summary
    # =====================================================

    if valid_images == 0:

        raise RuntimeError(
            f"No valid images found in {dataset_name}"
        )


    mean_box_iou = total_box_iou / valid_images * 100
    mean_click1_iou = total_click1_iou / valid_images * 100

    mean_box_dice = total_box_dice / valid_images * 100
    mean_box_precision = total_box_precision / valid_images * 100
    mean_box_recall = total_box_recall / valid_images * 100
    mean_box_latency = total_box_latency / valid_images
    mean_box_fps = 1000 / mean_box_latency

    mean_click1_dice = total_click1_dice / valid_images * 100
    mean_click1_precision = total_click1_precision / valid_images * 100
    mean_click1_recall = total_click1_recall / valid_images * 100
    mean_click1_latency = total_click1_latency / valid_images
    mean_click1_fps = 1000 / mean_click1_latency


    print("\n==============================")
    print(dataset_name)
    print("==============================")
    print(f"Images Evaluated : {valid_images}")

    print("\n----- Box Prompt -----")
    print(f"mIoU          : {mean_box_iou:.2f}%")
    print(f"Dice          : {mean_box_dice:.2f}%")
    print(f"Precision     : {mean_box_precision:.2f}%")
    print(f"Recall        : {mean_box_recall:.2f}%")
    print(f"Latency (ms)  : {mean_box_latency:.2f}")
    print(f"FPS           : {mean_box_fps:.2f}")

    print("\n----- 1-Click Prompt -----")
    print(f"mIoU          : {mean_click1_iou:.2f}%")
    print(f"Dice          : {mean_click1_dice:.2f}%")
    print(f"Precision     : {mean_click1_precision:.2f}%")
    print(f"Recall        : {mean_click1_recall:.2f}%")
    print(f"Latency (ms)  : {mean_click1_latency:.2f}")
    print(f"FPS           : {mean_click1_fps:.2f}")

    summary = pd.DataFrame([{

        "Dataset": dataset_name,
        "Images": valid_images,

        "Box mIoU": mean_box_iou,
        "Box Dice": mean_box_dice,
        "Box Precision": mean_box_precision,
        "Box Recall": mean_box_recall,
        "Box Latency (ms)": mean_box_latency,
        "Box FPS": mean_box_fps,

        "1-Click mIoU": mean_click1_iou,
        "1-Click Dice": mean_click1_dice,
        "1-Click Precision": mean_click1_precision,
        "1-Click Recall": mean_click1_recall,
        "1-Click Latency (ms)": mean_click1_latency,
        "1-Click FPS": mean_click1_fps

    }])

    results_df = pd.DataFrame(results)

    return summary, results_df

# =====================================================
# Run Evaluation
# =====================================================
np.random.seed(42)
summary_original, detail_original = evaluate_dataset(
    "Original",
    ORIGINAL_DIR
)
np.random.seed(42)
summary_enhanced, detail_enhanced = evaluate_dataset(
    "Enhanced",
    ENHANCED_DIR
)

# ----------------------------------------
# Save summaries
# ----------------------------------------

summary = pd.concat(
    [
        summary_original,
        summary_enhanced
    ],
    ignore_index=True
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "evaluation_summary.csv"
    ),
    index=False
)

# ----------------------------------------
# Save detailed results
# ----------------------------------------

detail_original.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "original_results.csv"
    ),
    index=False
)

detail_enhanced.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "enhanced_results.csv"
    ),
    index=False
)

print("\nResults saved to:")
print(OUTPUT_DIR)