import os
import time
import random

import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from pycocotools.coco import COCO

from efficient_sam.build_efficient_sam import (
    build_efficient_sam_vitt,
    build_efficient_sam_vits,
)


# Configuration

SEED = 42 # Fixed seed

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_DIR = (
    "/workspace/EfficientSAM/"
    "datasets/coco/val2017"
)

ANNOTATION_FILE = (
    "/workspace/EfficientSAM/"
    "datasets/coco/annotations/"
    "instances_val2017.json"
)

OUTPUT_DIR = (
    "/workspace/EfficientSAM/outputs"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "coco_reproduction.csv"
)


# Evaluate all valid COCO instances
MAX_INSTANCES = None


# Reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Models
# Load EfficientSAM-Ti and EfficientSAM-S
MODELS = {
    "EfficientSAM-Ti": build_efficient_sam_vitt(),
    "EfficientSAM-S": build_efficient_sam_vits(),
}

# Convert image to tensor
transform = transforms.ToTensor()


# Box prompt
# Tightest bounding box from GT mask
def generate_box_prompt(mask):

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None, None

    x1 = float(np.min(xs))
    y1 = float(np.min(ys))

    # Exclusive upper boundary
    x2 = float(np.max(xs) + 1)
    y2 = float(np.max(ys) + 1)

    input_points = torch.tensor(
        [[[
            [x1, y1],
            [x2, y2],
        ]]],
        dtype=torch.float32,
    )

    # EfficientSAM:
    # 2 = top-left box point
    # 3 = bottom-right box point
    input_labels = torch.tensor(
        [[[2, 3]]],
        dtype=torch.float32,
    )

    return input_points, input_labels


# Click prompt
# Uniform random sampling inside GT mask
def generate_click_prompt(
    mask,
    num_points,
    rng,
):

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None, None

    if len(xs) < num_points:
        return None, None

    selected_ids = rng.choice(
        len(xs),
        size=num_points,
        replace=False,
    )

    points = np.stack(
        [
            xs[selected_ids],
            ys[selected_ids],
        ],
        axis=1,
    ).astype(np.float32)

    input_points = torch.from_numpy(
        points
    ).reshape(
        1,
        1,
        num_points,
        2,
    )

    # All sampled clicks are foreground clicks
    input_labels = torch.ones(
        (
            1,
            1,
            num_points,
        ),
        dtype=torch.float32,
    )

    return input_points, input_labels


# Calculate mask IoU
# Measure overlap between GT and predicted masks
def compute_iou(
    gt_mask,
    pred_mask,
):

    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    intersection = np.logical_and(
        gt_mask,
        pred_mask,
    ).sum()

    union = np.logical_or(
        gt_mask,
        pred_mask,
    ).sum()

    if union == 0:
        return 1.0

    return float(
        intersection / union
    )


# Model inference
# Select most confident predicted mask

# Run model prediction
def predict_mask(
    model,
    image,
    input_points,
    input_labels,
):

    image_np = np.asarray(image)

    image_tensor = transform(
        image_np
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    input_points = input_points.to(
        DEVICE
    )

    input_labels = input_labels.to(
        DEVICE
    )


    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


    start_time = time.perf_counter()


    with torch.inference_mode():

        predicted_logits, predicted_iou = model(
            image_tensor,
            input_points,
            input_labels,
        )


    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


    elapsed_time = (
        time.perf_counter()
        - start_time
    )


    # Expected EfficientSAM output:
    # predicted_logits:
    # [B, num_queries, num_masks, H, W]

    # predicted_iou:
    # [B, num_queries, num_masks]

    mask_logits = predicted_logits[
        0,
        0,
    ]

    mask_scores = predicted_iou[
        0,
        0,
    ]


    # evaluate most confident mask
    best_mask_index = torch.argmax(
        mask_scores
    ).item()


    selected_logits = mask_logits[
        best_mask_index
    ]


    pred_mask = (
        torch.sigmoid(
            selected_logits
        ) >= 0.5
    )


    pred_mask = (
        pred_mask
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


    confidence = float(
        mask_scores[
            best_mask_index
        ]
        .detach()
        .cpu()
        .item()
    )


    return (
        pred_mask,
        elapsed_time,
        confidence,
    )


# Resize predicted mask
# Match prediction size with the GT mask
def resize_mask_to_gt(
    pred_mask,
    gt_mask,
):

    if pred_mask.shape == gt_mask.shape:
        return pred_mask

    pred_mask = cv2.resize(
        pred_mask,
        (
            gt_mask.shape[1],
            gt_mask.shape[0],
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return pred_mask


# Warm up GPU before timing
# Reduce first-run GPU timing overhead
def warmup_model(model):

    if DEVICE.type != "cuda":
        return

    print("Running GPU warm-up...")

    dummy_image = torch.zeros(
        (
            1,
            3,
            512,
            512,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    dummy_points = torch.tensor(
        [[[
            [10.0, 10.0],
            [100.0, 100.0],
        ]]],
        dtype=torch.float32,
        device=DEVICE,
    )

    dummy_labels = torch.tensor(
        [[[2, 3]]],
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.inference_mode():

        for _ in range(5):

            model(
                dummy_image,
                dummy_points,
                dummy_labels,
            )

    torch.cuda.synchronize()


# Evaluate box, 1-click, and 3-click prompts
def evaluate(
    model_name,
    model,
):

    print("\n")
    print("=" * 70)
    print(
        f"Evaluating: {model_name}"
    )
    print(
        f"Device: {DEVICE}"
    )
    print("=" * 70)


    model = model.to(
        DEVICE
    )

    model.eval()


    # Separate deterministic RNG
    # Reset for every model so Ti and S receive
    # identical random click prompts
    rng = np.random.default_rng(
        SEED
    )


    coco = COCO(
        ANNOTATION_FILE
    )


    ann_ids = coco.getAnnIds()

    anns = coco.loadAnns(
        ann_ids
    )


    # Optional debugging limit
    if MAX_INSTANCES is not None:

        anns = anns[
            :MAX_INSTANCES
        ]


    print(
        f"Loaded annotations: {len(anns)}"
    )


    warmup_model(
        model
    )


    box_ious = []
    click1_ious = []
    click3_ious = []

    inference_times = []

    skipped_crowd = 0
    skipped_invalid = 0
    skipped_missing_image = 0


    progress_bar = tqdm(
        anns,
        desc=model_name,
    )


    for ann in progress_bar:

        # Skip crowd annotations
        # Exclude crowd regions from instance evaluation
        if ann.get(
            "iscrowd",
            0,
        ) == 1:

            skipped_crowd += 1

            continue


        # Validate segmentation
        # Ignore annotations without segmentation data
        if "segmentation" not in ann:

            skipped_invalid += 1

            continue


        # Load image
        img_info = coco.loadImgs(
            ann["image_id"]
        )[0]


        image_path = os.path.join(
            IMAGE_DIR,
            img_info["file_name"],
        )


        if not os.path.exists(
            image_path
        ):

            skipped_missing_image += 1

            continue


        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        # Ground-truth instance mask
        # Convert COCO annotation into a binary mask
        gt_mask = coco.annToMask(
            ann
        ).astype(
            np.uint8
        )


        if gt_mask.sum() == 0:

            skipped_invalid += 1

            continue


        # Box prompt
        (
            box_points,
            box_labels,
        ) = generate_box_prompt(
            gt_mask
        )


        if box_points is None:

            skipped_invalid += 1

            continue


        (
            pred_box,
            box_latency,
            _,
        ) = predict_mask(
            model,
            image,
            box_points,
            box_labels,
        )


        pred_box = resize_mask_to_gt(
            pred_box,
            gt_mask,
        )


        box_iou = compute_iou(
            gt_mask,
            pred_box,
        )


        # 1 click
        # Evaluate using one foreground point
        (
            click1_points,
            click1_labels,
        ) = generate_click_prompt(
            gt_mask,
            num_points=1,
            rng=rng,
        )


        if click1_points is None:

            skipped_invalid += 1

            continue


        (
            pred_click1,
            _,
            _,
        ) = predict_mask(
            model,
            image,
            click1_points,
            click1_labels,
        )


        pred_click1 = resize_mask_to_gt(
            pred_click1,
            gt_mask,
        )


        click1_iou = compute_iou(
            gt_mask,
            pred_click1,
        )


        # 3 clicks
        # Evaluate using three foreground points
        (
            click3_points,
            click3_labels,
        ) = generate_click_prompt(
            gt_mask,
            num_points=3,
            rng=rng,
        )


        if click3_points is None:

            skipped_invalid += 1

            continue


        (
            pred_click3,
            _,
            _,
        ) = predict_mask(
            model,
            image,
            click3_points,
            click3_labels,
        )


        pred_click3 = resize_mask_to_gt(
            pred_click3,
            gt_mask,
        )


        click3_iou = compute_iou(
            gt_mask,
            pred_click3,
        )


        # Save metrics
        # Store IoU and box inference time
        box_ious.append(
            box_iou
        )

        click1_ious.append(
            click1_iou
        )

        click3_ious.append(
            click3_iou
        )

        inference_times.append(
            box_latency
        )


        progress_bar.set_postfix(
            {
                "Box": (
                    f"{np.mean(box_ious) * 100:.2f}"
                ),
                "1Click": (
                    f"{np.mean(click1_ious) * 100:.2f}"
                ),
                "3Click": (
                    f"{np.mean(click3_ious) * 100:.2f}"
                ),
            }
        )


    # Final results
    valid_count = len(
        box_ious
    )


    if valid_count == 0:

        raise RuntimeError(
            "No valid COCO instances were evaluated."
        )


    mean_box_iou = (np.mean(box_ious) * 100)
    mean_click1_iou = (np.mean(click1_ious) * 100)
    mean_click3_iou = (np.mean(click3_ious) * 100)
    mean_latency_ms = (np.mean(inference_times)* 1000)

    fps = (
        1000.0 / mean_latency_ms
        if mean_latency_ms > 0
        else 0.0
    )


    print("\n")
    print("-" * 70)

    print(f"Results: {model_name}")

    print("-" * 70)

    print(f"Valid instances : {valid_count}")
    print(f"Box mIoU       : {mean_box_iou:.2f}")
    print(f"1-click mIoU   : {mean_click1_iou:.2f}")
    print(f"3-click mIoU   : {mean_click3_iou:.2f}")
    print(f"Latency        : {mean_latency_ms:.2f} ms")
    print(f"FPS            : {fps:.2f}")
    print("-" * 70)
    print(
        f"Skipped crowd  : {skipped_crowd}"
    )

    print(
        f"Skipped invalid: {skipped_invalid}"
    )

    print(
        f"Missing images : {skipped_missing_image}"
    )

    print("-" * 70)


    return {

        "Model": model_name,

        "Evaluated Instances":
            valid_count,

        "Box mIoU":
            round(
                mean_box_iou,
                2,
            ),

        "1-click mIoU":
            round(
                mean_click1_iou,
                2,
            ),

        "3-click mIoU":
            round(
                mean_click3_iou,
                2,
            ),

        "Latency(ms)":
            round(
                mean_latency_ms,
                2,
            ),

        "FPS":
            round(
                fps,
                2,
            ),
    }



# Evaluate all models and save results
def main():

    print("=" * 70)

    print("EfficientSAM Evaluation")

    print("COCO Reproduction")

    print("=" * 70)

    print(f"Device: {DEVICE}")

    print(f"Seed: {SEED}")

    print(f"Images: {IMAGE_DIR}")

    print(f"Annotations: {ANNOTATION_FILE}")

    print(f"Max instances: {MAX_INSTANCES}")


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )


    results = []


    for model_name, model in MODELS.items():

        result = evaluate(
            model_name,
            model,
        )

        results.append(
            result
        )


        # Save after every model
        # Prevent loss if later evaluation crashes

        partial_df = pd.DataFrame(
            results
        )

        partial_df.to_csv(
            OUTPUT_CSV,
            index=False,
        )


        # Free GPU memory

        del model

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()


    # Final table
    df = pd.DataFrame(
        results
    )


    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )


    # Final reproduction results
    print("\n")
    print("=" * 70)
    print("FINAL COCO EVALUATION RESULTS")
    print("=" * 70)

    print(
        df.to_string(
            index=False
        )
    )

    print("=" * 70)
    print(f"Saved to: {OUTPUT_CSV}")
    print("=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()