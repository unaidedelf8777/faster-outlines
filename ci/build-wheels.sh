set -ex

export CIBW_PLATFORM=linux
export CIBW_ARCHS="x86_64"
export CIBW_MANYLINUX_X86_64_IMAGE="unaidedelf/cibw_rust:x86_64"
export CIBW_MANYLINUX_AARCH64_IMAGE="unaidedelf/cibw_rust:aarch64"

sudo apt update && sudo apt install python3-pip python3-dev -y

python3 -m pip install cibuildwheel

cibuildwheel --output-dir wheelhouse

for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" -w wheelhouse/repaired/
done