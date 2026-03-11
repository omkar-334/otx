#!/bin/bash
# Setup script for OTX on macOS (Apple Silicon)

set -e

# Install otx in editable mode without resolving deps.
# since decord==0.6.0 has no macOS ARM wheels so the installation fails.
#  -> video I/O uses cv2.VideoCapture instead.
echo "Installing otx (editable, no-deps)"
cd lib
uv pip install -e ".[dev]" --no-deps
cd ..

# Downgrade setuptools before anything else (v82+ removed pkg_resources needed by anomalib)
echo "Fixing setuptools for pkg_resources"
uv pip install "setuptools<81"

# Install core OTX dependencies
echo "Installing core dependencies"
uv pip install \
    datumaro==1.10.0 \
    lightning==2.4.0 \
    openvino==2025.2 \
    torchmetrics==1.6.0 \
    omegaconf==2.3.0 \
    "anomalib[core]==1.1.3" \
    onnx==1.17.0 \
    onnxconverter-common==1.16.0

# Tracker dependencies
echo "Installing tracker dependencies"
uv pip install numpy scipy lap cython-bbox

# Notebook / training dependencies
echo "Installing notebook & training dependencies"
uv pip install \
    jupyter \
    ipykernel \
    ipywidgets \
    matplotlib \
    pycocotools

echo "Setup done!"
