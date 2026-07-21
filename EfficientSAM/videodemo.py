# videodemo.py
import os
import cv2
import time
import torch
import numpy as np
from torchvision import transforms
from efficient_sam.build_efficient_sam import (
    build_efficient_sam_vits,
)

# Configuration
VIDEO_PATH = (
    "/workspace/EfficientSAM/"
    "videos/testvideo.mp4"
)
OUTPUT_VIDEO_PATH = (
    "/workspace/EfficientSAM/"
    "videos/demotestvideo.mp4"
)
MODEL_PATH = (
    "/workspace/EfficientSAM/weights/"
    "efficient_sam_vits.pt"
)
WINDOW_NAME = "EfficientSAM Video Object Tracking"
MAX_TRACKING_POINTS = 100
MIN_TRACKING_POINTS = 5
BOX_PADDING = 15
MIN_MASK_AREA = 50


# Device

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")

# Load EfficientSAM-S model
print("Loading EfficientSAM-S...")

# Build the pretrained EfficientSAM-S architecture
model = build_efficient_sam_vits()

# Load pretrained weights
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
)

# Some checkpoints store weights inside the model
if (
    isinstance(checkpoint, dict)
    and "model" in checkpoint
):
    checkpoint = checkpoint["model"]

# Load weights into the model
model.load_state_dict(checkpoint)

# Move model to GPU (if available)
model.to(DEVICE)
model.eval()
print("EfficientSAM-S loaded successfully.")


# Image transform
transform = transforms.ToTensor()

# Video input
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
    )
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps <= 0:
    video_fps = 30.0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video loaded: {VIDEO_PATH}")
print(f"Video FPS: {video_fps:.2f}")
print(f"Video Resolution: {frame_width} x {frame_height}")
print(f"Total Frames: {total_frames}")


# Video output 
os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# Downscale the saved video by 50% to 3840x1080 to ensure the codec can process and save it properly
out_width = frame_width
out_height = frame_height // 2
video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    fourcc,
    video_fps,
    (out_width, out_height), 
)
if not video_writer.isOpened():
    raise RuntimeError(
        f"Cannot create output video: {OUTPUT_VIDEO_PATH}"
    )
print(f"Output video: {OUTPUT_VIDEO_PATH}")


# Global variables 
PROMPT_POINT = "POINT"
PROMPT_BOX = "BOX"
prompt_mode = PROMPT_POINT
# User prompt inputs
selected_point = None
box_points = []
selected_box = None
initial_segmentation_required = False
def create_state():
    # Return the computed result
    return {
        "mask": None,
        "box": None,
        "tracking_points": None,
        "previous_gray": None,
        "tracking_active": False,
        "confidence": 0.0,
        "latency": 0.0,
        "tracking_prompt_point": None,
        "frames_since_update": 0
    }
original_state = create_state()
enhanced_state = create_state()

# Lucas-kanade optical flow
LK_PARAMS = dict(
    winSize=(31, 31),
    maxLevel=4,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    ),
)

# Feature detection
FEATURE_PARAMS = dict(
    maxCorners=MAX_TRACKING_POINTS,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7,
)
# Image quality measurements
def get_brightness(gray):
    return np.mean(gray)
def get_contrast(gray):
    return np.std(gray)
def get_blur(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# Gamma correction
def gamma_correction(image, gamma=1.3):
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)

# CLAHE
def clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe_filter.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Bilateral filter
def bilateral(image):
    return cv2.bilateralFilter(image, 7, 50, 50)

# Sharpen
def sharpen(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 2)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
# ================================
# Adaptive image enhancement
# ================================
def adaptive_enhance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = get_brightness(gray)
    contrast = get_contrast(gray)
    blur = get_blur(gray)
    operations = []
    enhanced = image.copy()
    if brightness < 70:
        enhanced = gamma_correction(enhanced, gamma=1.4)
        operations.append("GAMMA")
    if contrast < 35:
        enhanced = clahe(enhanced)
        operations.append("CLAHE")
    if blur < 80:
        enhanced = bilateral(enhanced)
        enhanced = sharpen(enhanced)
        operations.append("SHARPEN")
    return enhanced, operations, brightness, contrast, blur

# Prepare image for EfficientSAM-S
def prepare_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = transform(rgb_frame).unsqueeze(0)
    return image_tensor.to(DEVICE)
# EfficientSAM point segmentation
def predict_point(frame, x, y):
    image_tensor = prepare_image(frame)
    input_points = torch.tensor(
        [[[[float(x), float(y)]]]],
        dtype=torch.float32,
        device=DEVICE,
    )
    input_labels = torch.ones((1, 1, 1), dtype=torch.float32, device=DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.inference_mode():
        predicted_logits, predicted_iou = model(image_tensor, input_points, input_labels)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    latency = (time.perf_counter() - start_time) * 1000.0
    mask_logits = predicted_logits[0, 0]
    mask_scores = predicted_iou[0, 0]
    best_mask_index = torch.argmax(mask_scores).item()
    mask = torch.sigmoid(mask_logits[best_mask_index]) > 0.5
    mask = mask.cpu().numpy().astype(np.uint8)
    confidence = float(mask_scores[best_mask_index].cpu().item())
    return mask, confidence, latency
# =========================================
# EfficientSAM bounding box segmentation
# =========================================
def predict_box(frame, x1, y1, x2, y2):
    image_tensor = prepare_image(frame)
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    input_points = torch.tensor(
        [[[[float(left), float(top)], [float(right), float(bottom)]]]],
        dtype=torch.float32,
        device=DEVICE,
    )
    input_labels = torch.tensor([[[2, 3]]], dtype=torch.float32, device=DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.inference_mode():
        predicted_logits, predicted_iou = model(image_tensor, input_points, input_labels)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    latency = (time.perf_counter() - start_time) * 1000.0
    mask_logits = predicted_logits[0, 0]
    mask_scores = predicted_iou[0, 0]
    best_mask_index = torch.argmax(mask_scores).item()
    mask = torch.sigmoid(mask_logits[best_mask_index]) > 0.5
    mask = mask.cpu().numpy().astype(np.uint8)
    confidence = float(mask_scores[best_mask_index].cpu().item())
    return mask, confidence, latency

# -----------------
# RESIZE MASK
# -----------------
def resize_mask(mask, frame):
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(
            mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask
# -------------------------------------------------
# Calculate bounding box from segmentation mask
# -------------------------------------------------
def calculate_mask_box(mask, frame):
    if mask is None:
        return None
    height, width = frame.shape[:2]
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    
    if contour_area < MIN_MASK_AREA:
        return None
        
    x, y, box_width, box_height = cv2.boundingRect(largest_contour)
    x1 = max(0, x - BOX_PADDING)
    y1 = max(0, y - BOX_PADDING)
    x2 = min(width - 1, x + box_width + BOX_PADDING)
    y2 = min(height - 1, y + box_height + BOX_PADDING)
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    return (x1, y1, x2, y2)
# -------------------------------------------------
# # Mask overlap
# -------------------------------------------------
def overlay_mask(frame, mask, color=(0, 255, 0)):
    result = frame.copy()
    overlay = np.zeros_like(result)
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    
    mask_bool = mask.astype(bool)
    result[mask_bool] = (
        0.5 * result[mask_bool] + 0.5 * overlay[mask_bool]
    ).astype(np.uint8)
    return result
# -------------------------------------------------
# # Detect tracking point
# -------------------------------------------------
def detect_tracking_points(frame, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    points = cv2.goodFeaturesToTrack(gray, mask=mask_uint8, **FEATURE_PARAMS)
    return points

# -------------------------------------------------
# # Initialize object tracking
# -------------------------------------------------
def initialize_tracking(frame, state):
    state["tracking_points"] = detect_tracking_points(frame, state["mask"])
    state["previous_gray"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if state["tracking_points"] is None or len(state["tracking_points"]) < MIN_TRACKING_POINTS:
        state["tracking_active"] = False
    else:
        state["tracking_active"] = True
# -------------------------------------------------
# # Filters point using segmentation mask
# -------------------------------------------------
def filter_points_by_mask(points, mask):
    if points is None:
        return None
    valid_points = []
    height, width = mask.shape
    for point in points:
        x = int(point[0])
        y = int(point[1])
        if 0 <= x < width and 0 <= y < height and mask[y, x] > 0:
            valid_points.append([float(x), float(y)])
            
    if len(valid_points) == 0:
        return None
    return np.array(valid_points, dtype=np.float32)

# -------------------------------------------------
# # Calculate tracking prompt
# -------------------------------------------------
def calculate_tracking_prompt(points, mask):
    if points is None or mask is None:
        return None
    if len(points) == 0:
        return None
        
    valid_points = filter_points_by_mask(points, mask)
    if valid_points is None:
        return None
        
    center_x = np.mean(valid_points[:, 0])
    center_y = np.mean(valid_points[:, 1])
    
    distances = (
        (valid_points[:, 0] - center_x) ** 2
        + (valid_points[:, 1] - center_y) ** 2
    )
    best_index = np.argmin(distances)
    x, y = valid_points[best_index]
    return (int(x), int(y))
# -------------------------------------------------
# Find valid point inside mask
# -------------------------------------------------
def find_mask_prompt(mask):
    if mask is None:
        return None
    mask_uint8 = (mask.astype(np.uint8) * 255)
    if cv2.countNonZero(mask_uint8) == 0:
        return None
    distance_map = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    _, _, _, max_location = cv2.minMaxLoc(distance_map)
    return max_location
# ==========================================================
# Update multi-point optical flow tracking
# ==========================================================
def update_tracking(frame, state):
    if not state["tracking_active"] or state["tracking_points"] is None:
        return
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
        state["previous_gray"],
        current_gray,
        state["tracking_points"],
        None,
        **LK_PARAMS,
    )
    if new_points is None or status is None:
        state["tracking_active"] = False
        return
    status = status.reshape(-1)
    good_new = new_points[status == 1].reshape(-1, 2)
    old_points = state["tracking_points"].reshape(-1, 2)
    good_old = old_points[status == 1]
    if len(good_new) < MIN_TRACKING_POINTS:
        state["tracking_active"] = False
        return
    height, width = frame.shape[:2]
    good_new[:, 0] = np.clip(good_new[:, 0], 0, width - 1)
    good_new[:, 1] = np.clip(good_new[:, 1], 0, height - 1)
    state["tracking_points"] = good_new.reshape(-1, 1, 2).astype(np.float32)
    state["previous_gray"] = current_gray
    # Calculate median movement of tracked points
    if len(good_new) > 0 and len(good_old) > 0:
        dx = np.median(good_new[:, 0] - good_old[:, 0])
        dy = np.median(good_new[:, 1] - good_old[:, 1])
        if np.isnan(dx): dx = 0
        if np.isnan(dy): dy = 0
    else:
        dx, dy = 0, 0
    # Fast approximate update: shift box and mask by dx, dy
    if state["box"] is not None:
        x1, y1, x2, y2 = state["box"]
        x1 = max(0, int(x1 + dx))
        y1 = max(0, int(y1 + dy))
        x2 = min(width - 1, int(x2 + dx))
        y2 = min(height - 1, int(y2 + dy))
        state["box"] = (x1, y1, x2, y2)
    if state["mask"] is not None:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        state["mask"] = cv2.warpAffine(state["mask"], M, (width, height), flags=cv2.INTER_NEAREST)
    state["frames_since_update"] += 1
    # Only invoke EfficientSAM every 30 frames or if tracking quality degrades significantly
    SAM_UPDATE_INTERVAL = 30
    if state["frames_since_update"] >= SAM_UPDATE_INTERVAL or len(good_new) < (MAX_TRACKING_POINTS // 2):
        if prompt_mode == PROMPT_BOX and state["box"] is not None:
            x1, y1, x2, y2 = state["box"]
            new_mask, confidence, latency = predict_box(frame, x1, y1, x2, y2)
            new_mask = resize_mask(new_mask, frame)
            if np.count_nonzero(new_mask) >= MIN_MASK_AREA:
                state["mask"] = new_mask
                state["confidence"] = confidence
                state["latency"] = latency
                state["box"] = calculate_mask_box(state["mask"], frame)
        else:
            tracking_prompt = calculate_tracking_prompt(good_new, state["mask"])
            if tracking_prompt is None:
                tracking_prompt = find_mask_prompt(state["mask"])
            if tracking_prompt is not None:
                x, y = tracking_prompt
                state["tracking_prompt_point"] = (x, y)
                new_mask, confidence, latency = predict_point(frame, x, y)
                new_mask = resize_mask(new_mask, frame)
                if np.count_nonzero(new_mask) >= MIN_MASK_AREA:
                    state["mask"] = new_mask
                    state["confidence"] = confidence
                    state["latency"] = latency
                
                state["box"] = None
        state["frames_since_update"] = 0
        # Filter the tracked points against the *newly refreshed* mask
        filtered_points = filter_points_by_mask(good_new, state["mask"])
        if filtered_points is None or len(filtered_points) < MIN_TRACKING_POINTS:
            initialize_tracking(frame, state)
        else:
            state["tracking_points"] = filtered_points.reshape(-1, 1, 2).astype(np.float32)

# -------------------------------------------------
# Mouse callback
# -------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global selected_point, box_points, selected_box
    global initial_segmentation_required, original_state, enhanced_state
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    # Map the x coordinate to a single frame width
    # If x > frame_width, it means the user clicked on the enhanced (right) side.
    mapped_x = x % frame_width
    if prompt_mode == PROMPT_POINT:
        original_state = create_state()
        enhanced_state = create_state()
        
        selected_point = (mapped_x, y)
        initial_segmentation_required = True
    elif prompt_mode == PROMPT_BOX:
        if len(box_points) == 0:
            original_state = create_state()
            enhanced_state = create_state()
            box_points.append((mapped_x, y))
        elif len(box_points) == 1:
            box_points.append((mapped_x, y))
            x1 = min(box_points[0][0], box_points[1][0])
            y1 = min(box_points[0][1], box_points[1][1])
            x2 = max(box_points[0][0], box_points[1][0])
            y2 = max(box_points[0][1], box_points[1][1])
            selected_box = (x1, y1, x2, y2)
            box_points = []
            initial_segmentation_required = True
# -------------------------------------------------
# Create window
# -------------------------------------------------
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 480)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
# -------------------------------------------------
# Main variables
# -------------------------------------------------
paused = False
last_frame = None
print("\n==============================================")
print(" EfficientSAM Dual Video Object Tracking")
print("==============================================")
print("Left Click : Select Object (on either pane)")
print("P          : Point Prompt Mode")
print("B          : Box Prompt Mode")
print("C          : Clear Tracking")
print("SPACE      : Pause / Resume")
print("Q          : Quit and Save Video")
print("==============================================\n")
# -------------------------------------------------
# Main video loop
# -------------------------------------------------
prev_write_time = time.perf_counter()
frame_interval = 1.0 / video_fps  # seconds per frame at output fps
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        last_frame = frame.copy()
    else:
        if last_frame is None:
            continue
        frame = last_frame.copy()
    # -------------------------------------------------
    # Process original pane
    # -------------------------------------------------
    original_display = frame.copy()
    
    if initial_segmentation_required:
        if prompt_mode == PROMPT_POINT and selected_point is not None:
            o_mask, o_conf, o_lat = predict_point(frame, selected_point[0], selected_point[1])
            original_state["mask"] = resize_mask(o_mask, frame)
            original_state["confidence"] = o_conf
            original_state["latency"] = o_lat
        elif prompt_mode == PROMPT_BOX and selected_box is not None:
            o_mask, o_conf, o_lat = predict_box(frame, *selected_box)
            original_state["mask"] = resize_mask(o_mask, frame)
            original_state["confidence"] = o_conf
            original_state["latency"] = o_lat
            
        if original_state["mask"] is not None:
            original_state["box"] = calculate_mask_box(original_state["mask"], frame)
            initialize_tracking(frame, original_state)
    elif not paused:
        update_tracking(frame, original_state)
    # -------------------------------------------------
    # Process enhanced pane
    # -------------------------------------------------
    (
        enhanced_frame,
        enhancement_operations,
        brightness,
        contrast,
        blur,
    ) = adaptive_enhance(frame)
    enhanced_display = enhanced_frame.copy()
    
    if initial_segmentation_required:
        if prompt_mode == PROMPT_POINT and selected_point is not None:
            e_mask, e_conf, e_lat = predict_point(enhanced_frame, selected_point[0], selected_point[1])
            enhanced_state["mask"] = resize_mask(e_mask, enhanced_frame)
            enhanced_state["confidence"] = e_conf
            enhanced_state["latency"] = e_lat
        elif prompt_mode == PROMPT_BOX and selected_box is not None:
            e_mask, e_conf, e_lat = predict_box(enhanced_frame, *selected_box)
            enhanced_state["mask"] = resize_mask(e_mask, enhanced_frame)
            enhanced_state["confidence"] = e_conf
            enhanced_state["latency"] = e_lat
            
        if enhanced_state["mask"] is not None:
            enhanced_state["box"] = calculate_mask_box(enhanced_state["mask"], enhanced_frame)
            initialize_tracking(enhanced_frame, enhanced_state)
    elif not paused:
        update_tracking(enhanced_frame, enhanced_state)
        
    initial_segmentation_required = False
        
    # -------------------------------------------------
    # Drawing utilities
    # -------------------------------------------------
    def draw_ui(display_frame, state, title, color_mask, is_enhanced=False):
        # Draw mask
        if state["mask"] is not None:
            display_frame = overlay_mask(display_frame, state["mask"], color=color_mask)
        # Draw bounding box if in BOX mode
        if prompt_mode == PROMPT_BOX and state["box"] is not None:
            bx1, by1, bx2, by2 = state["box"]
            cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            cv2.putText(display_frame, "Tracked Object", (bx1, max(25, by1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        
        # Optical flow points are still tracked internally in state["tracking_points"]
        # Draw prompt point 
        if state["tracking_prompt_point"] is not None:
            cv2.circle(display_frame, state["tracking_prompt_point"], 5, (0, 255, 255), -1)
        elif prompt_mode == PROMPT_POINT and selected_point is not None:
            cv2.circle(display_frame, selected_point, 5, (0, 0, 255), -1)
                        
        if prompt_mode == PROMPT_BOX and len(box_points) == 1:
            cv2.circle(display_frame, box_points[0], 5, (0, 255, 255), -1)
            
        # Draw UI Elements
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        # Title (ORIGINAL / ENHANCED)
        cv2.putText(display_frame, title, (20, 40), font, font_scale, (0, 255, 255), font_thickness)
        
        # # Latency
        # cv2.putText(display_frame, f"Latency: {state['latency']:.2f} ms", (20, 75), font, font_scale, (0, 255, 0), font_thickness)
        
        # Enhancement info on RESULTS pane
        if is_enhanced and enhancement_operations:
            enhancement_str = "+".join(enhancement_operations)
            cv2.putText(display_frame, enhancement_str, (20, 75), font, font_scale, (255, 255, 0), font_thickness)
            
        # Instructions on ORIGINAL pane (bottom)
        if not is_enhanced:
            controls = [
                "Left Click : Select Object",
                "P : Point Prompt",
                "B : Bounding Box",
                "SPACE : Pause",
                "Q : Quit AND SAVE"
            ]
            h = display_frame.shape[0]
            for i, c_text in enumerate(controls):
                y_pos = h - 20 - (len(controls) - 1 - i) * 30
                cv2.putText(display_frame, c_text, (20, y_pos), font, 0.7, (255, 255, 255), 2)
        return display_frame
        
    # Draw both sides
    original_display = draw_ui(original_display, original_state, "ORIGINAL", (0, 255, 0), is_enhanced=False)
    enhanced_display = draw_ui(enhanced_display, enhanced_state, "ENHANCE", (255, 0, 255), is_enhanced=True)
    # Combine into dual pane
    display = np.hstack((original_display, enhanced_display))
    now = time.perf_counter()
    elapsed = now - prev_write_time
    frames_to_write = max(1, int(round(elapsed / frame_interval)))
    write_display = cv2.resize(display, (out_width, out_height))
    for _ in range(frames_to_write):
        video_writer.write(write_display)
    prev_write_time = now
    cv2.imshow(WINDOW_NAME, display)
    delay = max(1, int(1000 / video_fps))
    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        print("Q pressed. Saving video and exiting...")
        break
    elif key == ord("p"):
        prompt_mode = PROMPT_POINT
        selected_point = None
        box_points = []
        selected_box = None
        original_state = create_state()
        enhanced_state = create_state()
        initial_segmentation_required = False
        print("Prompt mode changed to POINT.")
    elif key == ord("b"):
        prompt_mode = PROMPT_BOX
        selected_point = None
        box_points = []
        selected_box = None
        original_state = create_state()
        enhanced_state = create_state()
        initial_segmentation_required = False
        print("Prompt mode changed to BOUNDING BOX.")
    elif key == ord("c"):
        selected_point = None
        box_points = []
        selected_box = None
        original_state = create_state()
        enhanced_state = create_state()
        initial_segmentation_required = False
        print("Tracking cleared.")
    elif key == 32:
        paused = not paused
        print("Video paused." if paused else "Video resumed.")

# -------------------------------------------------
# Cleanup
# -------------------------------------------------
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("\n==============================================")
print("Program terminated.")
print(f"Demo video saved to: {OUTPUT_VIDEO_PATH}")
print("==============================================")