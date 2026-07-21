FROM python:3.11

WORKDIR /workspace

# Install system dependencies for OpenCV and graphics libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 1. Install base math and imaging packages first
RUN pip install --no-cache-dir \
    numpy \
    opencv-python \
    matplotlib \
    scikit-image

# 2. Force install PyTorch compiled specifically for your laptop's NVIDIA driver
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu118

CMD ["bash"]