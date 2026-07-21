# prompt_utils.py

import numpy as np
import torch
import cv2

# Bounding box prompt
def generate_box_prompt(mask):
    
    # Find foreground coordinates
    ys, xs = np.where(mask > 0)
    
    # Check for empty mask
    if len(xs) == 0:
        return None, None
    
    # Get top-left coordinates
    x1 = np.min(xs)
    y1 = np.min(ys)
    
    # Get bottom-right coordinates
    x2 = np.max(xs)
    y2 = np.max(ys)
    
    # Convert box points to tensor
    input_points = torch.tensor(
        [[[
            [x1, y1],
            [x2, y2]
        ]]],
        dtype=torch.float32
    )

    # EfficientSAM:
    # 2 = top-left
    # 3 = bottom-right
    input_labels = torch.tensor(
        [[[2, 3]]],
        dtype=torch.float32
    )
    
    # Return box prompt
    return input_points, input_labels


# Random positive clicks
# Sample foreground points from mask
def generate_click_prompt(mask,
                          num_points=1,
                          seed=42):
    
    # Find foreground coordinates
    ys, xs = np.where(mask > 0)
    
    # Check for empty mask
    if len(xs) == 0:
        return None
    
    # Set random seed
    np.random.seed(seed)
    
    # Limit clicks to available pixels
    num_points = min(
        num_points,
        len(xs)
    )
    
    # Randomly select pixel indices
    ids = np.random.choice(
        len(xs),
        num_points,
        replace=False
    )
    
    # Store click coordinates
    points = []
    
    # Build click point list
    for i in ids:
        points.append(
            [xs[i], ys[i]]
        )
    # Return selected points
    return points


# Convert click list to EfficientSAM tensors
def build_click_prompt(points):
    
    # Convert points to tensor
    input_points = torch.tensor(
        [[points]],
        dtype=torch.float32
    )
    
    # Label all clicks as foreground
    input_labels = torch.ones(
        (
            1,
            1,
            len(points)
        ),
        dtype=torch.float32
    )
    # Return click prompt tensors
    return input_points, input_labels



# Create binary ground truth
def create_binary_mask(
        label_image,
        class_color):
  
    # Find pixels matching class color
    mask = np.all(
        label_image == class_color,
        axis=2
    )
    
    # Convert mask to binary format
    return mask.astype(np.uint8)


# Random object selection
def select_random_object(label_image,
                         ignore_black=True,
                         seed=42):
   
    # Find unique class colors
    colors = np.unique(
        label_image.reshape(-1, 3),
        axis=0
    )
    
    # Store valid classes
    valid = []
    
    # Check each class color
    for c in colors:
        
        # Ignore black background
        if ignore_black:

            if np.all(c == [0, 0, 0]):
                continue
        
        # Create temporary class mask
        mask = np.all(
            label_image == c,
            axis=2
        )

        # Keep sufficiently large objects
        if np.sum(mask) > 100:
            valid.append(c)
    
    # Check for valid classes
    if len(valid) == 0:
        return None, None
    
    # Set random seed
    np.random.seed(seed)
    
    # Randomly select one class
    selected = valid[
        np.random.randint(
            len(valid)
        )
    ]
    # Create selected class mask
    binary = create_binary_mask(
        label_image,
        selected
    )
    
    # Return mask and class color
    return binary, selected


# Centroid
def mask_centroid(mask):
    
    # Find foreground coordinates
    ys, xs = np.where(mask > 0)
    
    # Check for empty mask
    if len(xs) == 0:
        return None
    
    # Calculate center coordinates
    cx = int(xs.mean())
    cy = int(ys.mean())
    
    # Return centroid point
    return [cx, cy]


# Resize Prediction
def resize_prediction(pred_mask,
                      gt_mask):
    
    # Check mask dimensions
    if pred_mask.shape != gt_mask.shape:
        
        # Resize prediction to GT size
        pred_mask = cv2.resize(
            pred_mask,
            (
                gt_mask.shape[1],
                gt_mask.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Return resized prediction
    return pred_mask