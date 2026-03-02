#!/usr/bin/env ruby
# Script to download ONNX Runtime library for macOS
# This is called automatically by the podspec during pod install

require 'fileutils'
require 'open-uri'

# Configuration
ONNXRUNTIME_VERSION = "1.22.0"
SCRIPT_DIR = __dir__
ONNXRUNTIME_LIB = File.join(SCRIPT_DIR, "libonnxruntime.#{ONNXRUNTIME_VERSION}.dylib")

def download_onnxruntime
  # Check if library already exists
  if File.exist?(ONNXRUNTIME_LIB)
    puts "✓ ONNX Runtime library already present"
    return true
  end

  puts "📦 ONNX Runtime library not found. Downloading from Microsoft releases..."

  download_url = "https://github.com/microsoft/onnxruntime/releases/download/v#{ONNXRUNTIME_VERSION}/onnxruntime-osx-universal2-#{ONNXRUNTIME_VERSION}.tgz"
  temp_dir = File.join(SCRIPT_DIR, "temp_download")
  archive_path = File.join(temp_dir, "onnxruntime.tgz")

  begin
    # Create temp directory
    FileUtils.mkdir_p(temp_dir)

    # Download the archive
    puts "  Downloading from #{download_url}..."
    total_size = nil
    URI.open(download_url,
      content_length_proc: ->(size) { total_size = size },
      progress_proc: ->(downloaded) {
        if total_size
          percent = (downloaded.to_f / total_size * 100).round(1)
          print "\r  Progress: #{percent}%"
        end
      }
    ) do |remote_file|
      File.open(archive_path, 'wb') { |f| f.write(remote_file.read) }
    end
    puts "\n  ✓ Download complete"

    # Extract the archive
    puts "  Extracting archive..."
    system("tar -xzf #{archive_path} -C #{temp_dir}") or raise "Failed to extract archive"

    # Find and copy the dylib (exclude .dSYM directories)
    extracted_lib = Dir.glob("#{temp_dir}/**/libonnxruntime.#{ONNXRUNTIME_VERSION}.dylib")
      .reject { |path| path.include?('.dSYM') }
      .first
    if extracted_lib && File.file?(extracted_lib)
      FileUtils.cp(extracted_lib, ONNXRUNTIME_LIB)
      puts "  ✓ Library extracted to #{File.basename(ONNXRUNTIME_LIB)}"
    else
      raise "Could not find libonnxruntime dylib in extracted archive"
    end

    puts "✅ ONNX Runtime #{ONNXRUNTIME_VERSION} downloaded successfully"
    return true

  rescue => e
    puts "❌ Failed to download ONNX Runtime: #{e.message}"
    puts "Please download manually from #{download_url}"
    return false

  ensure
    # Clean up temp directory
    FileUtils.rm_rf(temp_dir) if Dir.exist?(temp_dir)
  end
end

# Run the download if this script is executed directly
if __FILE__ == $0
  success = download_onnxruntime
  exit(success ? 0 : 1)
end

