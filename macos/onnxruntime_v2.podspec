#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint onnxruntime_v2.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'onnxruntime_v2'
  s.version          = '0.0.1'
  s.summary          = 'OnnxRuntime plugin for Flutter apps.'
  s.description      = <<-DESC
OnnxRuntime plugin for Flutter apps.
                       DESC
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }

  # Download ONNX Runtime library from official Microsoft release if not present
  require_relative 'download_onnxruntime'
  download_onnxruntime

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }
  # s.source_files = 'Classes/**/*'
  s.dependency 'FlutterMacOS'
  s.vendored_libraries = '*.dylib'
  s.platform = :osx, '10.14'
  s.static_framework = true
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  s.swift_version = '5.0'
end
