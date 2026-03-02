# The Flutter tooling requires that developers have a version of Visual Studio
# installed that includes CMake 3.14 or later. You should not increase this
# version, as doing so will cause the plugin to fail to compile for some
# customers of the plugin.
cmake_minimum_required(VERSION 3.14)

# Project-level configuration.
set(PROJECT_NAME "onnxruntime_v2")
project(${PROJECT_NAME} LANGUAGES CXX)

# Download ONNX Runtime libraries from official Microsoft release if not present
set(DOWNLOAD_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/download_onnxruntime.ps1")
if(EXISTS "${DOWNLOAD_SCRIPT}")
  execute_process(
    COMMAND powershell -ExecutionPolicy Bypass -File "${DOWNLOAD_SCRIPT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE DOWNLOAD_RESULT
    OUTPUT_VARIABLE DOWNLOAD_OUTPUT
    ERROR_VARIABLE DOWNLOAD_ERROR
  )

  if(DOWNLOAD_OUTPUT)
    message(STATUS "${DOWNLOAD_OUTPUT}")
  endif()

  if(NOT DOWNLOAD_RESULT EQUAL 0)
    message(WARNING "Failed to download ONNX Runtime libraries: ${DOWNLOAD_ERROR}")
    message(WARNING "Please download manually from https://github.com/microsoft/onnxruntime/releases")
  endif()
endif()

# Invoke the build for native code shared with the other target platforms.
# This can be changed to accommodate different builds.
# add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../src" "${CMAKE_CURRENT_BINARY_DIR}/shared")

# List of absolute paths to libraries that should be bundled with the plugin.
# This list could contain prebuilt libraries, or libraries created by an
# external build triggered from this build file.
#
# Includes GPU provider DLLs for CUDA, TensorRT, and DirectML acceleration.
# These DLLs are optional - if the GPU runtime is not available on the target
# system, ONNX Runtime will gracefully fall back to CPU execution.
set(onnxruntime_v2_bundled_libraries
  "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_providers_cuda.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_providers_shared.dll"
  "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_providers_tensorrt.dll"
  PARENT_SCOPE
)
