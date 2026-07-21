# metrics.py
import numpy as np

# IoU
# Calculate overlap between GT and prediction
def compute_iou(gt_mask, pred_mask):
    
    # Convert masks to boolean
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)
    
    # Calculate overlapping pixels
    intersection = np.logical_and(
        gt,
        pred
    ).sum()
    
    # Calculate combined pixels
    union = np.logical_or(
        gt,
        pred
    ).sum()
    
    # Return IoU score
    return intersection / (union + 1e-6)


# Dice score
# Calculate Dice similarity score
def compute_dice(
        gt_mask,
        pred_mask):
 
    # Convert masks to boolean
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)
    
    # Calculate overlapping pixels
    intersection = np.logical_and(
        gt,
        pred
    ).sum()


    # Return Dice score
    return (
        2 * intersection
    ) / (
        gt.sum()
        + pred.sum()
        + 1e-6
    )


# Precision
# Calculate segmentation precision
def compute_precision(
        gt_mask,
        pred_mask):
    
    # Convert masks to boolean
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)
    
    # Calculate true positives
    tp = np.logical_and(
        gt,
        pred
    ).sum()

    # Calculate false positives
    fp = np.logical_and(
        np.logical_not(gt),
        pred
    ).sum()
    
    # Return precision score
    return tp / (tp + fp + 1e-6)


# Recall
# Calculate segmentation recall
def compute_recall(
        gt_mask,
        pred_mask):
    
    # Convert masks to boolean
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)
    
    # Calculate true positives
    tp = np.logical_and(
        gt,
        pred
    ).sum()
    
    # Calculate false negatives
    fn = np.logical_and(
        gt,
        np.logical_not(pred)
    ).sum()
    
    # Return recall score
    return tp / (tp + fn + 1e-6)


# FPS
# Convert latency to frames per second
def latency_to_fps(
        latency_ms):
    
    # Prevent invalid division
    if latency_ms <= 0:
        return 0
    
    # Calculate FPS
    return 1000.0 / latency_ms


# Average metrics
# Calculate average evaluation metrics
def summarize_metrics(results):

    summary = {}

    summary["Box mIoU"] = np.mean(
        [r["box"] for r in results]
    )

    summary["1-click mIoU"] = np.mean(
        [r["click1"] for r in results]
    )

    # summary["3-click mIoU"] = np.mean(
    #     [r["click3"] for r in results]
    # )

    summary["Latency(ms)"] = np.mean(
        [r["latency"] for r in results]
    )

    summary["FPS"] = latency_to_fps(
        summary["Latency(ms)"]
    )

    return summary
