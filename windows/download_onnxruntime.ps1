# Script to download ONNX Runtime library for Windows
# This is called automatically by CMake during build

$ErrorActionPreference = "Stop"

# Configuration
$ONNXRUNTIME_VERSION = "1.22.0"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
# ONNXRUNTIME_DEPS_URL env var overrides the default base URL
$BASE_URL = if ($env:ONNXRUNTIME_DEPS_URL) { $env:ONNXRUNTIME_DEPS_URL } else { "https://repo.devops.ptech.io/artifactory/tools/onnx-runtime/$ONNXRUNTIME_VERSION" }
$DOWNLOAD_URL = "$BASE_URL/onnx-windows-x64.7z"

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
    Write-Host "[OK] ONNX Runtime libraries already present"
    exit 0
}

Write-Host "ONNX Runtime libraries not found. Downloading from Artifactory..."

# Create temp directory. Outside $SCRIPT_DIR: Flutter reaches the plugin through
# .plugin_symlinks, and tar.exe refuses to extract through a symlink.
$TEMP_DIR = Join-Path ([System.IO.Path]::GetTempPath()) "onnxruntime_v2_download"
if (Test-Path $TEMP_DIR) {
    Remove-Item -Recurse -Force $TEMP_DIR
}
New-Item -ItemType Directory -Path $TEMP_DIR | Out-Null

try {
    # Download the archive
    $ARCHIVE_PATH = Join-Path $TEMP_DIR "onnxruntime.7z"
    Write-Host "  Downloading from $DOWNLOAD_URL..."

    # Use .NET WebClient for progress
    $webClient = New-Object System.Net.WebClient
    $webClient.DownloadFile($DOWNLOAD_URL, $ARCHIVE_PATH)

    Write-Host "  [OK] Download complete"

    # Extract the archive
    Write-Host "  Extracting archive..."
    # Expand-Archive only handles .zip; use 7-Zip if installed, else Windows' own tar.exe (libarchive).
    # Full path: a Git for Windows / MSYS tar earlier on PATH reads "C:\..." as a remote host and fails.
    $sevenZip = (Get-Command 7z -ErrorAction SilentlyContinue).Source
    if (-not $sevenZip -and (Test-Path "$env:ProgramFiles\7-Zip\7z.exe")) { $sevenZip = "$env:ProgramFiles\7-Zip\7z.exe" }
    $windowsTar = Join-Path $env:SystemRoot "System32\tar.exe"
    if ($sevenZip) {
        & $sevenZip x $ARCHIVE_PATH "-o$TEMP_DIR" -y | Out-Null
    } else {
        & $windowsTar -xf $ARCHIVE_PATH -C $TEMP_DIR
    }
    if ($LASTEXITCODE -ne 0) { throw "Failed to extract $ARCHIVE_PATH (install 7-Zip if $windowsTar lacks 7z support)" }
    Write-Host "  [OK] Extraction complete"

    # Find and copy the libraries
    Write-Host "  Copying libraries..."
    foreach ($lib in $EXPECTED_LIBS) {
        # Find the library in the extracted directory
        $foundLib = Get-ChildItem -Path $TEMP_DIR -Filter $lib -Recurse -File | Select-Object -First 1

        if ($foundLib) {
            $destPath = Join-Path $SCRIPT_DIR $lib
            Copy-Item -Path $foundLib.FullName -Destination $destPath -Force
            Write-Host "    [OK] $lib"
        } else {
            Write-Host "    [WARNING] $lib not found (may be optional)"
        }
    }

    Write-Host "[SUCCESS] ONNX Runtime $ONNXRUNTIME_VERSION downloaded successfully"
    exit 0

} catch {
    Write-Host "Failed to download ONNX Runtime: $_"
    Write-Host "Please download manually from: $DOWNLOAD_URL"
    exit 1

} finally {
    # Clean up temp directory
    if (Test-Path $TEMP_DIR) {
        Remove-Item -Recurse -Force $TEMP_DIR
    }
}

