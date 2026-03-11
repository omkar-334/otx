#!/bin/bash
# Setup script for OTX on macOS (Apple Silicon)

set -e

# Install otx in editable mode (without resolving deps, since decord has no macOS ARM wheels)
cd lib
uv pip install -e ".[dev]" --no-deps
cd ..

# install core dependencies
echo "Installing core dependencies..."
uv pip install \
    datumaro==1.10.0 \
    lightning==2.4.0 \
    openvino==2025.2 \
    torchmetrics==1.6.0 \
    omegaconf==2.3.0 \
    "anomalib[core]==1.1.3" \
    onnx==1.17.0 \
    onnxconverter-common==1.16.0


uv pip install "setuptools<81"

# Tracker dependencies
uv pip install numpy scipy lap cython-bbox

echo "Setup done!"
