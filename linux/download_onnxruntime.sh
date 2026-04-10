#!/bin/bash
# Script to download ONNX Runtime library for Linux
# This is called automatically by CMake during build

set -e

# Configuration
ONNXRUNTIME_VERSION="1.24.4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu_cuda13-${ONNXRUNTIME_VERSION}.tgz"

# List of expected libraries
EXPECTED_LIBS=(
    "libonnxruntime.so.${ONNXRUNTIME_VERSION}"
    "libonnxruntime_providers_cuda.so"
    "libonnxruntime_providers_shared.so"
    "libonnxruntime_providers_tensorrt.so"
)

# Check if all libraries already exist
all_exist=true
for lib in "${EXPECTED_LIBS[@]}"; do
    if [ ! -f "${SCRIPT_DIR}/${lib}" ]; then
        all_exist=false
        break
    fi
done

if [ "$all_exist" = true ]; then
    echo "✓ ONNX Runtime libraries already present"
    exit 0
fi

echo "📦 ONNX Runtime libraries not found. Downloading from Microsoft releases..."

# Create temp directory
TEMP_DIR="${SCRIPT_DIR}/temp_download"
mkdir -p "${TEMP_DIR}"

# Cleanup function
cleanup() {
    if [ -d "${TEMP_DIR}" ]; then
        rm -rf "${TEMP_DIR}"
    fi
}
trap cleanup EXIT

# Download the archive
ARCHIVE_PATH="${TEMP_DIR}/onnxruntime.tgz"
echo "  Downloading from ${DOWNLOAD_URL}..."

if command -v wget &> /dev/null; then
    wget -q --show-progress -O "${ARCHIVE_PATH}" "${DOWNLOAD_URL}" || {
        echo "❌ Failed to download with wget"
        exit 1
    }
elif command -v curl &> /dev/null; then
    curl -L --progress-bar -o "${ARCHIVE_PATH}" "${DOWNLOAD_URL}" || {
        echo "❌ Failed to download with curl"
        exit 1
    }
else
    echo "❌ Neither wget nor curl found. Please install one of them."
    exit 1
fi

echo "  ✓ Download complete"

# Extract the archive
echo "  Extracting archive..."
tar -xzf "${ARCHIVE_PATH}" -C "${TEMP_DIR}" || {
    echo "❌ Failed to extract archive"
    exit 1
}

# Find and copy the libraries
echo "  Copying libraries..."
for lib in "${EXPECTED_LIBS[@]}"; do
    # Find the library in the extracted directory
    found_lib=$(find "${TEMP_DIR}" -name "${lib}" -type f | head -n 1)
    
    if [ -n "${found_lib}" ]; then
        cp "${found_lib}" "${SCRIPT_DIR}/${lib}"
        echo "    ✓ ${lib}"
    else
        echo "    ⚠ ${lib} not found (may be optional)"
    fi
done

echo "✅ ONNX Runtime ${ONNXRUNTIME_VERSION} downloaded successfully"

