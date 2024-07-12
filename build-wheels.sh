#!/bin/bash
set -ex

# Prepare the environment
export CIBW_PLATFORM=linux
export CIBW_ARCHS="x86_64"
export CIBW_MANYLINUX_X86_64_IMAGE="unaidedelf/cibw_rust:x86_64"
export CIBW_MANYLINUX_AARCH64_IMAGE="unaidedelf/cibw_rust:aarch64"



# Install Python and pip if not already installed
sudo apt update && sudo apt install python3-pip python3-dev -y

# Install cibuildwheel
python3 -m pip install cibuildwheel

# Run cibuildwheel to build the wheels
cibuildwheel --output-dir wheelhouse

# Repair the wheels using auditwheel
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" -w wheelhouse/repaired/
done
