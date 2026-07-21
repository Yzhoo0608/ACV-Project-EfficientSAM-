import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm



# Paths
INPUT_DIR = "/workspace/EfficientSAM/datasets/camvid/test"

OUTPUT_DIR = "/workspace/EfficientSAM/datasets/camvid/test_enhanced"

COMPARE_DIR = os.path.join(
    OUTPUT_DIR,
    "comparisons"
)

# Crreate output directory
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# Crreate comparison directory
os.makedirs(
    COMPARE_DIR,
    exist_ok=True
)


# ---------------------------
# Image quality measurements
# ---------------------------

# Measure average brightness
def get_brightness(gray):
    return np.mean(gray)

# Measure image contrast
def get_contrast(gray):
    return np.std(gray)

# Measure image blur
def get_blur(gray):
    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


# ---------------------------
# # Enhancement functions
# ---------------------------

# Adjust image brightness
def gamma_correction(image, gamma=1.3):

    inv_gamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)

# Improve local contrast
def clahe(image):
    # Convert BGR to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l, a, b = cv2.split(lab)

    # Create CLAHE filter
    clahe_filter = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # Enhance lightness channel
    l = clahe_filter.apply(l)
    
    # Merge LAB channels
    lab = cv2.merge((l, a, b))
    
    # Convert back to BGR
    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

# Reduce noise while preserving edges
def bilateral(image):

    return cv2.bilateralFilter(
        image,
        7,
        50,
        50
    )

# Sharpen image
def sharpen(image):
    # Create blurred image
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        2
    )
    # Apply sharpening
    return cv2.addWeighted(
        image,
        1.5,
        blurred,
        -0.5,
        0
    )


# ---------------------------
# Adaptive enhancement
# ---------------------------
def adaptive_enhance(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )
    # Calculate image quality values
    brightness = get_brightness(gray)
    contrast = get_contrast(gray)
    blur = get_blur(gray)
    
    # Store applied operations
    operations = []
    
    # Copy original image
    enhanced = image.copy()

    # --------------------------
    # Dark image
    # --------------------------

    if brightness < 70:

        enhanced = gamma_correction(
            enhanced,
            gamma=1.4
        )

        operations.append("Gamma")

    # --------------------------
    # Low Contrast
    # --------------------------
    
    # Apply CLAHE
    if contrast < 35:

        enhanced = clahe(
            enhanced
        )

        operations.append("CLAHE")

    # --------------------------
    # Blurry
    # --------------------------

    if blur < 80:

        enhanced = bilateral(
            enhanced
        )

        enhanced = sharpen(
            enhanced
        )

        operations.append("Sharpen")
    
    # Return enhancement results
    return (
        enhanced,
        operations,
        brightness,
        contrast,
        blur
    )



# --------------------------
# Process Dataset
# --------------------------
# Get all PNG images
files = sorted([
    f
    for f in os.listdir(INPUT_DIR)
    if f.endswith(".png")
])

# Print total images
print(f"Found {len(files)} images")

# Store image records
records = []

# Enhancement counters
num_changed = 0
num_unchanged = 0

# Process each image
for filename in tqdm(files):
    
    # Create input path
    input_path = os.path.join(
        INPUT_DIR,
        filename
    )
    
    # Load image
    image = cv2.imread(
        input_path
    )

    # Apply adaptive enhancement
    enhanced, ops, b, c, blur = adaptive_enhance(
        image
    )
    
    # Save enhanced image
    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(
        output_path,
        enhanced
    )

    # Save comparison only if modified

    if len(ops) > 0:

        comparison = np.hstack([
            image,
            enhanced
        ])
        
        # Add original label
        cv2.putText(
            comparison,
            "Original",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )
        
        # Add enhanced label
        cv2.putText(
            comparison,
            "Enhanced",
            (image.shape[1]+20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )
        
        # Create comparison path
        compare_path = os.path.join(
            COMPARE_DIR,
            filename
        )
        
        # Save comparison image
        cv2.imwrite(
            compare_path,
            comparison
        )
        
        # Count enhanced image
        num_changed += 1

    else:
        # Count unchanged image
        num_unchanged += 1
    
    # Store image measurements
    records.append({

        "Image": filename,

        "Brightness": round(b,2),

        "Contrast": round(c,2),

        "Blur": round(blur,2),

        "Gamma": "Yes" if "Gamma" in ops else "No",

        "CLAHE": "Yes" if "CLAHE" in ops else "No",

        "Sharpen": "Yes" if "Sharpen" in ops else "No",

        "Enhanced": "Yes" if len(ops)>0 else "No"

    })

# --------------------------
# Save CSV
# --------------------------

df = pd.DataFrame(records)

csv_path = os.path.join(
    OUTPUT_DIR,
    "enhancement_report.csv"
)

df.to_csv(
    csv_path,
    index=False
)

# --------------------------
# Summary
# --------------------------

print("\n========================================")
print("Adaptive enhancement completed")
print("========================================")
print(f"Total images      : {len(files)}")
print(f"Enhanced images   : {num_changed}")
print(f"Unchanged images  : {num_unchanged}")
print()
print("Enhanced images:")
print(OUTPUT_DIR)

print()
print("Comparison images:")
print(COMPARE_DIR)

print()
print("CSV report:")
print(csv_path)

print("========================================")