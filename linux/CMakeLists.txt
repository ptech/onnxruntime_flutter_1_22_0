# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

# Project-level configuration.
set(PROJECT_NAME "onnxruntime_v2")
project(${PROJECT_NAME} LANGUAGES CXX)

# Download ONNX Runtime libraries from official Microsoft release if not present
set(DOWNLOAD_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/download_onnxruntime.sh")
if(EXISTS "${DOWNLOAD_SCRIPT}")
  execute_process(
    COMMAND bash "${DOWNLOAD_SCRIPT}"
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
# Includes GPU-enabled library for CUDA, TensorRT, and ROCm acceleration.
# The GPU library is optional - if CUDA/ROCm runtime is not available,
# ONNX Runtime will gracefully fall back to CPU execution.
set(onnxruntime_v2_bundled_libraries
  "${CMAKE_CURRENT_SOURCE_DIR}/libonnxruntime.so.1.22.0"
  "${CMAKE_CURRENT_SOURCE_DIR}/libonnxruntime_providers_cuda.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/libonnxruntime_providers_shared.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/libonnxruntime_providers_tensorrt.so"
  PARENT_SCOPE
)
