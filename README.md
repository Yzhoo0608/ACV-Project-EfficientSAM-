# EfficientSAM Reproduction
This repository reproduces and evaluates the **EfficientSAM** framework.
The project includes:
- EfficientSAM-Ti and EfficientSAM-S inference
- COCO 2017 validation evaluation
- CamVid test dataset evaluation
- Adaptive image enhancement comparison
- EfficientSAM video object tracking demo with interactive point/box prompts
---
## Prerequisites: OpenCV GUI Setup for Video Demo
The video demo uses `cv2.imshow()` and `cv2.setMouseCallback()`. Because we are running inside Docker, Docker needs access to your Windows display to show the video windows.
**Step 1: Install VcXsrv**
1. Download and install [VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/).
2. Launch **XLaunch**.
3. Select **Multiple windows** and click Next.
4. Select **Start no client** and click Next.
5. **Important:** Check the **Disable access control** box.
6. Finish and keep VcXsrv running in the background.
**Step 2: Find Windows IP Address**
1. Open PowerShell and run:
   ```powershell
   ipconfig
   ```
2. Find your IPv4 address (e.g., `192.168.0.6`). Write this down for the Docker setup step.
---
## Step 1: Clone Repository & Project Setup
1. Open PowerShell and navigate to your working directory:
   ```powershell
   cd C:\ACV_Project
   ```
2. Clone the official EfficientSAM repository:
   ```powershell
   git clone https://github.com/yformer/EfficientSAM.git
   ```
3. Enter the project:
   ```powershell
   cd EfficientSAM
   ```
### Expected Project Structure
```text
ACV_Project
│
├── Dockerfile
├── docker-compose.yml
├── README.md
│
└── EfficientSAM
    │
    ├── efficient_sam/
    │   └── EfficientSAM source code
    │
    ├── weights/
    │   ├── efficient_sam_vits.pt.zip
    │   └── efficient_sam_vits.pt
    │
    ├── datasets/
    │   ├── coco/
    │   └── camvid/
    │
    ├── outputs/
    │   └── evaluation results
    │
    ├── EfficientSAM_inference.py
    ├── reproduce.py
    │
    ├── camvid_loader.py
    ├── create_enhanced_camvid.py
    ├── evaluate_camvid.py
    │
    ├── prompt_utils.py
    ├── metrics.py
    │
    └── videodemo.py
```
---
## Step 2: Prepare Weights and Datasets
*Make sure you are in the `C:\ACV_Project\EfficientSAM` directory before doing this.*
### 1. Model Weights Preparation
The ViT-S checkpoint is provided as a compressed file in the cloned repository. You must unzip it before running inference:
```bash
unzip weights/efficient_sam_vits.pt.zip -d weights/
```
### 2. COCO 2017 Validation Dataset (for `reproduce.py`)
1. Download the COCO 2017 validation images from Kaggle: [COCO 2017 Val Images](https://www.kaggle.com/datasets/xthink/coco-2017-val-images)
2. Extract the images into: `datasets/coco/val2017/`
3. Download the COCO annotations: [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
4. Extract and move the `instances_val2017.json` file into `datasets/coco/annotations/`.

**Final COCO structure:**
```text
datasets/
└── coco/
    ├── val2017/
    │   └── (image files .jpg)
    └── annotations/
        └── instances_val2017.json
```
### 3. CamVid Test Dataset (for CamVid scripts)
This project only uses the CamVid **test split** for evaluation.
1. Download the dataset from Kaggle: [CamVid Dataset](https://www.kaggle.com/datasets/carlolepelaars/camvid)
2. Extract it and place it in the `datasets/camvid/` directory.
---
## Evaluation Dependencies
The CamVid evaluation requires additional utility files:
- `prompt_utils.py`
- `metrics.py`
These files provide:
### `prompt_utils.py`
Contains functions for generating EfficientSAM prompts:
- Bounding box prompts
- Random click prompts
- Click prompt formatting
- Random object selection from segmentation labels

Used by:
```python
evaluate_camvid.py
```
### `metrics.py`
Contains evaluation metrics:
- IoU
- Dice coefficient
- Precision
- Recall

Used to compare:
- Original CamVid images
- Enhanced CamVid images
---
## Step 3: Docker Environment Setup
**1. Update `docker-compose.yml`**
Open your `docker-compose.yml` file and add the `DISPLAY` environment variable using the IPv4 address you found earlier:
```yaml
environment:
  - PYTHONUNBUFFERED=1
  - DISPLAY=192.168.0.6:0   # Replace with YOUR IPv4 address
```
**2. Build the Docker Image**
Open PowerShell in `C:\ACV_Project` (where your docker-compose file is) and run:
```powershell
docker compose build
```
**3. Start the Docker Container**
Run the container in the background:
```powershell
docker compose up -d
```
*(Optional) Verify it is running by typing `docker ps`. You should see `acv-project`.*
**4. Enter the Docker Container**
```powershell
docker exec -it acv-project bash
```
Once inside, navigate to the EfficientSAM workspace:
```bash
cd /workspace/EfficientSAM
```
---
## Step 4: Execution Pipeline
Run these commands inside the Docker container (`/workspace/EfficientSAM`) in the following order:

**1. EfficientSAM Inference**
- Run a basic inference test using the EfficientSAM models (Ti and S) on a sample image.
```bash
python EfficientSAM_inference.py
```
**2. Reproduce Paper Results**

- Evaluate the models on the COCO dataset to reproduce the original paper results as closely as possible.
```bash
python reproduce.py
```
**3. CamVid Loader**
- Verify the CamVid dataset loading mechanism.
```bash
python camvid_loader.py
```
**4. Create Enhanced CamVid Dataset**
- Perform adaptive image preprocessing (gamma correction, CLAHE, bilateral filtering, sharpening) on the CamVid test dataset to improve image quality for evaluation.
```bash
python create_enhanced_camvid.py
```
**5. Evaluate CamVid**
- Evaluate the EfficientSAM model on both the original and the enhanced CamVid datasets to compare performance (IoU, Dice, Precision, Recall).
Before running evaluation, make sure these files exist:
- `prompt_utils.py`
- `metrics.py`

Run:
```bash
python evaluate_camvid.py
```
Results will be saved to:
```
outputs/camvid/
```
Generated files:
```text
outputs/camvid/
│
├── evaluation_summary.csv
├── original_results.csv
└── enhanced_results.csv
└── enhancement_report.csv
```
- `enhancement-report.csv`: This file is used to verify and analyze what preprocessing techniques were applied before evaluation.


**6. Video Demo**
Run a video demonstration of object tracking using the EfficientSAM model. *(This requires the XLaunch GUI setup from earlier to be running).*
```bash
python videodemo.py
```
