# MIA-Bench — PyTorch (CUDA) runtime image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies and Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the repository
COPY . .

CMD ["bash"]
