#!/bin/bash
set -e

# Upgrade pip to avoid older versions that may fail
python3 -m pip install --upgrade pip

# Install base Python dependencies
pip install --prefer-binary -r requirements.txt

# Attempt to install optional GPU libraries. Installation failures are ignored
pip install cupy || echo "cupy installation failed; falling back to NumPy"
pip install fftx || echo "fftx installation failed; using numpy.fft"
