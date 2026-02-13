# Script to download ONNX Runtime library for Windows
# This is called automatically by CMake during build

$ErrorActionPreference = "Stop"

# Configuration
$ONNXRUNTIME_VERSION = "1.22.0"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DOWNLOAD_URL = "https://github.com/microsoft/onnxruntime/releases/download/v$ONNXRUNTIME_VERSION/onnxruntime-win-x64-gpu-$ONNXRUNTIME_VERSION.zip"

# List of expected libraries
$EXPECTED_LIBS = @(
    "onnxruntime.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_tensorrt.dll"
)

# Check if all libraries already exist
$allExist = $true
foreach ($lib in $EXPECTED_LIBS) {
    $libPath = Join-Path $SCRIPT_DIR $lib
    if (-not (Test-Path $libPath)) {
        $allExist = $false
        break
    }
}

if ($allExist) {
    Write-Host "✓ ONNX Runtime libraries already present"
    exit 0
}

Write-Host "📦 ONNX Runtime libraries not found. Downloading from Microsoft releases..."

# Create temp directory
$TEMP_DIR = Join-Path $SCRIPT_DIR "temp_download"
if (Test-Path $TEMP_DIR) {
    Remove-Item -Recurse -Force $TEMP_DIR
}
New-Item -ItemType Directory -Path $TEMP_DIR | Out-Null

try {
    # Download the archive
    $ARCHIVE_PATH = Join-Path $TEMP_DIR "onnxruntime.zip"
    Write-Host "  Downloading from $DOWNLOAD_URL..."
    
    # Use .NET WebClient for progress
    $webClient = New-Object System.Net.WebClient
    $webClient.DownloadFile($DOWNLOAD_URL, $ARCHIVE_PATH)
    
    Write-Host "  ✓ Download complete"

    # Extract the archive
    Write-Host "  Extracting archive..."
    Expand-Archive -Path $ARCHIVE_PATH -DestinationPath $TEMP_DIR -Force
    Write-Host "  ✓ Extraction complete"

    # Find and copy the libraries
    Write-Host "  Copying libraries..."
    foreach ($lib in $EXPECTED_LIBS) {
        # Find the library in the extracted directory
        $foundLib = Get-ChildItem -Path $TEMP_DIR -Filter $lib -Recurse -File | Select-Object -First 1
        
        if ($foundLib) {
            $destPath = Join-Path $SCRIPT_DIR $lib
            Copy-Item -Path $foundLib.FullName -Destination $destPath -Force
            Write-Host "    ✓ $lib"
        } else {
            Write-Host "    ⚠ $lib not found (may be optional)"
        }
    }

    Write-Host "✅ ONNX Runtime $ONNXRUNTIME_VERSION downloaded successfully"
    exit 0

} catch {
    Write-Host "❌ Failed to download ONNX Runtime: $_"
    Write-Host "Please download manually from $DOWNLOAD_URL"
    exit 1

} finally {
    # Clean up temp directory
    if (Test-Path $TEMP_DIR) {
        Remove-Item -Recurse -Force $TEMP_DIR
    }
}

