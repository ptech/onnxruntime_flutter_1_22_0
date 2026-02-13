import 'dart:typed_data';

import 'package:onnxruntime_v2/src/ort_value.dart';

/// Extension on TypedData to get the corresponding ONNX tensor element data type
extension TypedDataOnnxType on TypedData {
  /// Get the ONNXTensorElementDataType for this TypedData
  ///
  /// Throws an exception if the TypedData type is not supported by ONNX.
  ///
  /// Supported types:
  /// - Float32List → ONNXTensorElementDataType.float
  /// - Float64List → ONNXTensorElementDataType.double
  /// - Int8List → ONNXTensorElementDataType.int8
  /// - Uint8List → ONNXTensorElementDataType.uint8
  /// - Int16List → ONNXTensorElementDataType.int16
  /// - Uint16List → ONNXTensorElementDataType.uint16
  /// - Int32List → ONNXTensorElementDataType.int32
  /// - Uint32List → ONNXTensorElementDataType.uint32
  /// - Int64List → ONNXTensorElementDataType.int64
  /// - Uint64List → ONNXTensorElementDataType.uint64
  ONNXTensorElementDataType get onnxType {
    return TypedDataOnnxType.getOnnxTypeFromType(runtimeType);
  }

  /// Static helper to get ONNXTensorElementDataType from a Type
  ///
  /// This is useful when you have a Type object (like from a generic parameter)
  /// and need to get the corresponding ONNX type.
  ///
  /// Example:
  /// ```dart
  /// final type = TypedDataOnnxType.getOnnxTypeFromType(Float32List);
  /// // Returns ONNXTensorElementDataType.float
  /// ```
  static ONNXTensorElementDataType getOnnxTypeFromType(Type type) {
    // Use toString() for reliable Type comparison
    // Type objects from generic parameters don't always match with ==
    final typeName = type.toString();

    if (typeName == 'Float32List') {
      return ONNXTensorElementDataType.float;
    } else if (typeName == 'Float64List') {
      return ONNXTensorElementDataType.double;
    } else if (typeName == 'Int8List') {
      return ONNXTensorElementDataType.int8;
    } else if (typeName == 'Uint8List') {
      return ONNXTensorElementDataType.uint8;
    } else if (typeName == 'Int16List') {
      return ONNXTensorElementDataType.int16;
    } else if (typeName == 'Uint16List') {
      return ONNXTensorElementDataType.uint16;
    } else if (typeName == 'Int32List') {
      return ONNXTensorElementDataType.int32;
    } else if (typeName == 'Uint32List') {
      return ONNXTensorElementDataType.uint32;
    } else if (typeName == 'Int64List') {
      return ONNXTensorElementDataType.int64;
    } else if (typeName == 'Uint64List') {
      return ONNXTensorElementDataType.uint64;
    } else {
      throw Exception('Unsupported TypedData type: $type');
    }
  }
}

