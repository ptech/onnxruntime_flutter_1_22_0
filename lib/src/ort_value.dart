import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:onnxruntime_v2/src/bindings/onnxruntime_bindings_generated.dart'
    as bg;
import 'package:onnxruntime_v2/src/ort_env.dart';
import 'package:onnxruntime_v2/src/ort_status.dart';
import 'package:onnxruntime_v2/src/util/list_shape_extension.dart';
import 'package:onnxruntime_v2/src/util/typed_data_onnx_extension.dart';

/// Internal class to hold reusable buffer information
class _ReusableBuffer {
  final ffi.Pointer<ffi.Void> ptr;
  final int size;

  _ReusableBuffer(this.ptr, this.size);
}

abstract class OrtValue {
  late ffi.Pointer<bg.OrtValue> _ptr;

  ffi.Pointer<bg.OrtValue> get ptr => _ptr;

  int get address => _ptr.address;

  Object? get value;

  Map<OrtTensorTypeAndShapeInfo, OrtTensorTypeAndShapeInfo> _createMapInfo(
      ffi.Pointer<bg.OrtValue> ortValuePtr) {
    final keyPtrPtr = calloc<ffi.Pointer<bg.OrtValue>>();
    final keyPtr = _getOrtValue(ortValuePtr, 0, keyPtrPtr);
    final keyInfo = OrtTensorTypeAndShapeInfo(keyPtr);
    _releaseOrtValue(keyPtr);
    calloc.free(keyPtrPtr);

    final valuePtrPtr = calloc<ffi.Pointer<bg.OrtValue>>();
    final valuePtr = _getOrtValue(ortValuePtr, 1, valuePtrPtr);
    final valueInfo = OrtTensorTypeAndShapeInfo(valuePtr);
    _releaseOrtValue(valuePtr);
    calloc.free(valuePtrPtr);
    return {keyInfo: valueInfo};
  }

  ffi.Pointer<bg.OrtValue> _getOrtValue(ffi.Pointer<bg.OrtValue> ortValuePtr,
      int index, ffi.Pointer<ffi.Pointer<bg.OrtValue>> indexOrtValuePtrPtr) {
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetValue.asFunction<
            bg.OrtStatusPtr Function(
                ffi.Pointer<bg.OrtValue>,
                int,
                ffi.Pointer<bg.OrtAllocator>,
                ffi.Pointer<ffi.Pointer<bg.OrtValue>>)>()(
        ortValuePtr, index, OrtAllocator.instance.ptr, indexOrtValuePtrPtr);
    OrtStatus.checkOrtStatus(statusPtr);
    return indexOrtValuePtrPtr.value;
  }

  ffi.Pointer<T> _getTensorMutableData<T extends ffi.NativeType>(
      ffi.Pointer<bg.OrtValue> ortValuePtr,
      ffi.Pointer<ffi.Pointer<T>> dataPtrPtr) {
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetTensorMutableData
            .asFunction<
                bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
                    ffi.Pointer<ffi.Pointer<ffi.Void>>)>()(
        ortValuePtr, dataPtrPtr.cast());
    OrtStatus.checkOrtStatus(statusPtr);
    return dataPtrPtr.value;
  }

  List<String> _getStringList(ffi.Pointer<bg.OrtValue> ortValuePtr) {
    final info = OrtTensorTypeAndShapeInfo(ortValuePtr);
    final tensorShapeElementCount = info._tensorShapeElementCount;
    final dataLengthPtr = calloc<ffi.Size>();
    var statusPtr = OrtEnv.instance.ortApiPtr.ref.GetStringTensorDataLength
        .asFunction<
            bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
                ffi.Pointer<ffi.Size>)>()(ortValuePtr, dataLengthPtr);
    OrtStatus.checkOrtStatus(statusPtr);
    final dataLength = dataLengthPtr.value;
    calloc.free(dataLengthPtr);
    // last index is '\0'
    final dataPtr = calloc<ffi.Char>(dataLength + 1);
    // last index is dataLength
    final offsetPtr = calloc<ffi.Size>(tensorShapeElementCount + 1);
    statusPtr = OrtEnv.instance.ortApiPtr.ref.GetStringTensorContent.asFunction<
            bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
                ffi.Pointer<ffi.Void>, int, ffi.Pointer<ffi.Size>, int)>()(
        ortValuePtr,
        dataPtr.cast(),
        dataLength,
        offsetPtr,
        tensorShapeElementCount);
    OrtStatus.checkOrtStatus(statusPtr);
    statusPtr = OrtEnv.instance.ortApiPtr.ref.GetStringTensorDataLength
            .asFunction<
                bg.OrtStatusPtr Function(
                    ffi.Pointer<bg.OrtValue>, ffi.Pointer<ffi.Size>)>()(
        ortValuePtr,
        ffi.Pointer.fromAddress(offsetPtr.address +
            tensorShapeElementCount * ffi.sizeOf<ffi.Size>()));
    OrtStatus.checkOrtStatus(statusPtr);
    final list = <String>[];
    for (int i = 0; i < tensorShapeElementCount; ++i) {
      final size = offsetPtr[i + 1] - offsetPtr[i];
      final strPtr = calloc<ffi.Char>(size + 1);
      for (int j = 0; j < size; ++j) {
        strPtr[j] = dataPtr[offsetPtr[i] + j];
      }
      final str = strPtr.cast<Utf8>().toDartString();
      list.add(str);
      calloc.free(strPtr);
    }
    calloc.free(dataPtr);
    calloc.free(offsetPtr);
    return list;
  }

  List<num> _getNumList(ffi.Pointer<bg.OrtValue> ortValuePtr) {
    final info = OrtTensorTypeAndShapeInfo(ortValuePtr);
    final tensorElementType = info._tensorElementType;
    final tensorShapeElementCount = info._tensorShapeElementCount;
    final data = <num>[];
    if (tensorElementType == ONNXTensorElementDataType.uint8) {
      final dataPtrPtr = calloc<ffi.Pointer<ffi.Uint8>>();
      final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
      for (int i = 0; i < tensorShapeElementCount; ++i) {
        data.add(dataPtr[i]);
      }
      calloc.free(dataPtrPtr);
    } else if (tensorElementType == ONNXTensorElementDataType.int8) {
      final dataPtrPtr = calloc<ffi.Pointer<ffi.Int8>>();
      final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
      for (int i = 0; i < tensorShapeElementCount; ++i) {
        data.add(dataPtr[i]);
      }
      calloc.free(dataPtrPtr);
    } else if (tensorElementType == ONNXTensorElementDataType.uint16) {
      final dataPtrPtr = calloc<ffi.Pointer<ffi.Uint16>>();
      final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
      for (int i = 0; i < tensorShapeElementCount; ++i) {
        data.add(dataPtr[i]);
      }
      calloc.free(dataPtrPtr);
    } else if (tensorElementType == ONNXTensorElementDataType.int16) {
      final dataPtrPtr = calloc<ffi.Pointer<ffi.Int16>>();
      final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
      for (int i = 0; i < tensorShapeElementCount; ++i) {
        data.add(dataPtr[i]);
      }
      calloc.free(dataPtrPtr);
    } else if (tensorElementType == ONNXTensorElementDataType.uint32) {
      final dataPtrPtr = calloc<ffi.Pointer<ffi.Uint32>>();
      final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
      for (int i = 0; i < tensorShapeElementCount; ++i) {
        data.add(dataPtr[i]);
      }
      calloc.free(dataPtrPtr);
    } else if (tensorElementType == ONNXTensorElementDataType.int32) {
      final dataPtrPtr = calloc<ffi.Pointer<ffi.Int32>>();
      final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
      for (int i = 0; i < tensorShapeElementCount; ++i) {
        data.add(dataPtr[i]);
      }
      calloc.free(dataPtrPtr);
    } else if (tensorElementType == ONNXTensorElementDataType.uint64) {
      final dataPtrPtr = calloc<ffi.Pointer<ffi.Uint64>>();
      final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
      for (int i = 0; i < tensorShapeElementCount; ++i) {
        data.add(dataPtr[i]);
      }
      calloc.free(dataPtrPtr);
    } else if (tensorElementType == ONNXTensorElementDataType.int64) {
      final dataPtrPtr = calloc<ffi.Pointer<ffi.Int64>>();
      final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
      for (int i = 0; i < tensorShapeElementCount; ++i) {
        data.add(dataPtr[i]);
      }
      calloc.free(dataPtrPtr);
    } else if (tensorElementType == ONNXTensorElementDataType.float) {
      final dataPtrPtr = calloc<ffi.Pointer<ffi.Float>>();
      final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
      for (int i = 0; i < tensorShapeElementCount; ++i) {
        data.add(dataPtr[i]);
      }
      calloc.free(dataPtrPtr);
    } else if (tensorElementType == ONNXTensorElementDataType.double) {
      final dataPtrPtr = calloc<ffi.Pointer<ffi.Double>>();
      final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
      for (int i = 0; i < tensorShapeElementCount; ++i) {
        data.add(dataPtr[i]);
      }
      calloc.free(dataPtrPtr);
    }
    return data;
  }

  List<bool> _getBoolList(ffi.Pointer<bg.OrtValue> ortValuePtr) {
    final info = OrtTensorTypeAndShapeInfo(ortValuePtr);
    final tensorElementType = info._tensorElementType;
    final tensorShapeElementCount = info._tensorShapeElementCount;
    final data = <bool>[];
    if (tensorElementType == ONNXTensorElementDataType.bool) {
      final dataPtrPtr = calloc<ffi.Pointer<ffi.Bool>>();
      final dataPtr = _getTensorMutableData(ortValuePtr, dataPtrPtr);
      for (int i = 0; i < tensorShapeElementCount; ++i) {
        data.add(dataPtr[i]);
      }
      calloc.free(dataPtrPtr);
    }
    return data;
  }

  void _releaseOrtValue(ffi.Pointer<bg.OrtValue> ortValuePtr) {
    OrtEnv.instance.ortApiPtr.ref.ReleaseValue
        .asFunction<void Function(ffi.Pointer<bg.OrtValue>)>()(ortValuePtr);
  }

  void release() {
    _releaseOrtValue(_ptr);
  }
}

class OrtValueTensor extends OrtValue {
  late OrtTensorTypeAndShapeInfo _info;
  ffi.Pointer<ffi.Void> _dataPtr = ffi.nullptr;

  OrtValueTensor(ffi.Pointer<bg.OrtValue> ptr,
      [ffi.Pointer<ffi.Void>? dataPtr]) {
    _ptr = ptr;
    _info = OrtTensorTypeAndShapeInfo(ptr);
    if (dataPtr != null) {
      _dataPtr = dataPtr;
    }
  }

  factory OrtValueTensor.fromAddress(int address) {
    return OrtValueTensor(ffi.Pointer.fromAddress(address));
  }

  static OrtValueTensor _createTensorWithString(String data) {
    return _createTensorWithStringList(<String>[data], []);
  }

  static OrtValueTensor _createTensorWithStringList(List<String> data,
      [List<int>? shape]) {
    final ortValuePtrPtr = calloc<ffi.Pointer<bg.OrtValue>>();
    shape ??= data.shape;
    final shapeSize = shape.length;
    final shapePtr = calloc<ffi.Int64>(shapeSize);
    shapePtr.asTypedList(shapeSize).setRange(0, shapeSize, shape);

    var statusPtr = OrtEnv.instance.ortApiPtr.ref.CreateTensorAsOrtValue
            .asFunction<
                bg.OrtStatusPtr Function(
                    ffi.Pointer<bg.OrtAllocator>,
                    ffi.Pointer<ffi.Int64> shape,
                    int,
                    int,
                    ffi.Pointer<ffi.Pointer<bg.OrtValue>>)>()(
        OrtAllocator.instance.ptr,
        shapePtr,
        shapeSize,
        ONNXTensorElementDataType.string.value,
        ortValuePtrPtr);
    OrtStatus.checkOrtStatus(statusPtr);
    final ortValuePtr = ortValuePtrPtr.value;
    for (int i = 0; i < data.length; ++i) {
      final str = data[i].toNativeUtf8().cast<ffi.Char>();
      statusPtr = OrtEnv.instance.ortApiPtr.ref.FillStringTensorElement
          .asFunction<
              bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
                  ffi.Pointer<ffi.Char>, int)>()(ortValuePtr, str, i);
      OrtStatus.checkOrtStatus(statusPtr);
    }
    calloc.free(ortValuePtrPtr);
    calloc.free(shapePtr);
    return OrtValueTensor(ortValuePtr);
  }

  static OrtValueTensor createTensorWithData(dynamic data) {
    if (data is int) {
      return createTensorWithDataList(<int>[data], []);
    }
    if (data is double) {
      return createTensorWithDataList(<double>[data], []);
    }
    if (data is bool) {
      return createTensorWithDataList(<bool>[data], []);
    }
    if (data is String) {
      return _createTensorWithString(data);
    }
    throw Exception('Invalid element type');
  }

  // Reusable buffer cache for optimized tensor creation
  // Maps data type to (pointer, size)
  static final Map<ONNXTensorElementDataType, _ReusableBuffer> _bufferCache = {};

  /// Helper to get or allocate a reusable buffer
  static ffi.Pointer<T> _getOrAllocateBuffer<T extends ffi.NativeType>(
    ONNXTensorElementDataType dataType,
    int dataSize,
    ffi.Pointer<T> Function(int) allocator,
  ) {
    final cached = _bufferCache[dataType];
    if (cached != null && cached.size >= dataSize) {
      return cached.ptr.cast<T>();
    }

    // Need to allocate new buffer
    if (cached != null) {
      calloc.free(cached.ptr);
    }
    final newPtr = allocator(dataSize);
    _bufferCache[dataType] = _ReusableBuffer(newPtr.cast<ffi.Void>(), dataSize);
    return newPtr;
  }

  /// Get or create a reusable typed buffer backed by native memory
  /// This allows you to write data directly to native memory without an extra copy
  ///
  /// The data type is automatically inferred from the generic type T.
  ///
  /// Supported types: Float32List, Float64List, Int8List, Uint8List, Int16List,
  /// Uint16List, Int32List, Uint32List, Int64List, Uint64List
  ///
  /// Usage:
  /// ```dart
  /// // Get buffer for Float32 - type is inferred!
  /// final buffer = OrtValueTensor.getOrCreateReusableBuffer<Float32List>(1228800);
  ///
  /// // Write directly to buffer
  /// buffer.setRange(0, imageData.length, imageData);
  ///
  /// // Create tensor from the buffer (zero-copy)
  /// final tensor = OrtValueTensor.createTensorFromBuffer(
  ///   buffer,
  ///   [1, 3, 640, 640],
  /// );
  /// ```
  static T getOrCreateReusableBuffer<T extends TypedData>(int size) {
    // Infer the data type from the generic type T using the static helper
    final dataType = TypedDataOnnxType.getOnnxTypeFromType(T);

    if (dataType == ONNXTensorElementDataType.float) {
      final ptr = _getOrAllocateBuffer<ffi.Float>(
        dataType,
        size,
        (size) => calloc<ffi.Float>(size),
      );
      return ptr.asTypedList(size) as T;
    } else if (dataType == ONNXTensorElementDataType.double) {
      final ptr = _getOrAllocateBuffer<ffi.Double>(
        dataType,
        size,
        (size) => calloc<ffi.Double>(size),
      );
      return ptr.asTypedList(size) as T;
    } else if (dataType == ONNXTensorElementDataType.int8) {
      final ptr = _getOrAllocateBuffer<ffi.Int8>(
        dataType,
        size,
        (size) => calloc<ffi.Int8>(size),
      );
      return ptr.asTypedList(size) as T;
    } else if (dataType == ONNXTensorElementDataType.uint8) {
      final ptr = _getOrAllocateBuffer<ffi.Uint8>(
        dataType,
        size,
        (size) => calloc<ffi.Uint8>(size),
      );
      return ptr.asTypedList(size) as T;
    } else if (dataType == ONNXTensorElementDataType.int16) {
      final ptr = _getOrAllocateBuffer<ffi.Int16>(
        dataType,
        size,
        (size) => calloc<ffi.Int16>(size),
      );
      return ptr.asTypedList(size) as T;
    } else if (dataType == ONNXTensorElementDataType.uint16) {
      final ptr = _getOrAllocateBuffer<ffi.Uint16>(
        dataType,
        size,
        (size) => calloc<ffi.Uint16>(size),
      );
      return ptr.asTypedList(size) as T;
    } else if (dataType == ONNXTensorElementDataType.int32) {
      final ptr = _getOrAllocateBuffer<ffi.Int32>(
        dataType,
        size,
        (size) => calloc<ffi.Int32>(size),
      );
      return ptr.asTypedList(size) as T;
    } else if (dataType == ONNXTensorElementDataType.uint32) {
      final ptr = _getOrAllocateBuffer<ffi.Uint32>(
        dataType,
        size,
        (size) => calloc<ffi.Uint32>(size),
      );
      return ptr.asTypedList(size) as T;
    } else if (dataType == ONNXTensorElementDataType.int64) {
      final ptr = _getOrAllocateBuffer<ffi.Int64>(
        dataType,
        size,
        (size) => calloc<ffi.Int64>(size),
      );
      return ptr.asTypedList(size) as T;
    } else if (dataType == ONNXTensorElementDataType.uint64) {
      final ptr = _getOrAllocateBuffer<ffi.Uint64>(
        dataType,
        size,
        (size) => calloc<ffi.Uint64>(size),
      );
      return ptr.asTypedList(size) as T;
    } else {
      throw Exception('Unsupported data type: $dataType');
    }
  }

  /// Create a tensor from a reusable buffer obtained via getOrCreateReusableBuffer
  /// This is a zero-copy operation since the buffer is already in native memory
  ///
  /// The data type is automatically inferred from the buffer type using the extension.
  /// The buffer must have been obtained from getOrCreateReusableBuffer.
  static OrtValueTensor createTensorFromBuffer(
    TypedData buffer,
    List<int> shape,
  ) {
    // Infer the data type from the buffer type using the extension
    final dataType = buffer.onnxType;

    // Get the pointer from the cached buffer
    final cached = _bufferCache[dataType];
    if (cached == null) {
      throw Exception(
        'Buffer not found for type $dataType. Call getOrCreateReusableBuffer first.',
      );
    }

    final dataPtr = cached.ptr;
    final dataSize = buffer.lengthInBytes ~/ buffer.elementSizeInBytes;
    final elementSize = buffer.elementSizeInBytes;
    final dataByteCount = dataSize * elementSize;

    // Create tensor
    final bindings = OrtEnv.instance.ortApiPtr;
    final shapeSize = shape.length;
    final shapePtr = calloc<ffi.Int64>(shapeSize);
    shapePtr.asTypedList(shapeSize).setAll(0, shape);

    final ortMemoryInfoPtrPtr = calloc<ffi.Pointer<bg.OrtMemoryInfo>>();
    var statusPtr = bindings.ref.AllocatorGetInfo.asFunction<
            bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtAllocator>,
                ffi.Pointer<ffi.Pointer<bg.OrtMemoryInfo>>)>()(
        OrtAllocator.instance.ptr, ortMemoryInfoPtrPtr);
    OrtStatus.checkOrtStatus(statusPtr);

    final ortMemoryInfoPtr = ortMemoryInfoPtrPtr.value;
    final ortValuePtrPtr = calloc<ffi.Pointer<bg.OrtValue>>();

    statusPtr = bindings.ref.CreateTensorWithDataAsOrtValue.asFunction<
            bg.OrtStatusPtr Function(
                ffi.Pointer<bg.OrtMemoryInfo>,
                ffi.Pointer<ffi.Void>,
                int,
                ffi.Pointer<ffi.Int64>,
                int,
                int,
                ffi.Pointer<ffi.Pointer<bg.OrtValue>>)>()(
        ortMemoryInfoPtr,
        dataPtr,
        dataByteCount,
        shapePtr,
        shapeSize,
        dataType.value,
        ortValuePtrPtr);
    OrtStatus.checkOrtStatus(statusPtr);

    final ortValuePtr = ortValuePtrPtr.value;
    calloc.free(shapePtr);
    calloc.free(ortValuePtrPtr);
    calloc.free(ortMemoryInfoPtrPtr);

    // Pass null to prevent automatic cleanup of our reusable buffer
    return OrtValueTensor(ortValuePtr, null);
  }

  /// Create a tensor from typed data using a reusable buffer (zero-copy when possible)
  /// This is much faster than createTensorWithDataList for repeated calls with same-sized data
  ///
  /// IMPORTANT: Tensors created with this method do NOT own their memory.
  /// You MUST call disposeReusableBuffers() when done to free the cached buffers.
  ///
  /// Example:
  /// ```dart
  /// // Create tensors (reuses buffers automatically)
  /// final tensor1 = OrtValueTensor.createTensorWithReusableBuffer(data1, shape);
  /// final tensor2 = OrtValueTensor.createTensorWithReusableBuffer(data2, shape);
  ///
  /// // Use tensors...
  /// tensor1.release();
  /// tensor2.release();
  ///
  /// // Clean up buffers when completely done
  /// OrtValueTensor.disposeReusableBuffers();
  /// ```
  static OrtValueTensor createTensorWithReusableBuffer(
    TypedData data,
    List<int> shape,
  ) {
    final int dataSize = data.lengthInBytes ~/ data.elementSizeInBytes;
    final ONNXTensorElementDataType dataType;
    final int elementSize;
    final ffi.Pointer<ffi.Void> dataPtr;

    // Determine type, allocate/reuse buffer, and copy data based on data type
    if (data is Float32List) {
      dataType = ONNXTensorElementDataType.float;
      elementSize = 4;
      final ptr = _getOrAllocateBuffer<ffi.Float>(dataType, dataSize, (size) => calloc<ffi.Float>(size));
      ptr.asTypedList(dataSize).setRange(0, dataSize, data);
      dataPtr = ptr.cast<ffi.Void>();
    } else if (data is Float64List) {
      dataType = ONNXTensorElementDataType.double;
      elementSize = 8;
      final ptr = _getOrAllocateBuffer<ffi.Double>(dataType, dataSize, (size) => calloc<ffi.Double>(size));
      ptr.asTypedList(dataSize).setRange(0, dataSize, data);
      dataPtr = ptr.cast<ffi.Void>();
    } else if (data is Int8List) {
      dataType = ONNXTensorElementDataType.int8;
      elementSize = 1;
      final ptr = _getOrAllocateBuffer<ffi.Int8>(dataType, dataSize, (size) => calloc<ffi.Int8>(size));
      ptr.asTypedList(dataSize).setRange(0, dataSize, data);
      dataPtr = ptr.cast<ffi.Void>();
    } else if (data is Uint8List) {
      dataType = ONNXTensorElementDataType.uint8;
      elementSize = 1;
      final ptr = _getOrAllocateBuffer<ffi.Uint8>(dataType, dataSize, (size) => calloc<ffi.Uint8>(size));
      ptr.asTypedList(dataSize).setRange(0, dataSize, data);
      dataPtr = ptr.cast<ffi.Void>();
    } else if (data is Int16List) {
      dataType = ONNXTensorElementDataType.int16;
      elementSize = 2;
      final ptr = _getOrAllocateBuffer<ffi.Int16>(dataType, dataSize, (size) => calloc<ffi.Int16>(size));
      ptr.asTypedList(dataSize).setRange(0, dataSize, data);
      dataPtr = ptr.cast<ffi.Void>();
    } else if (data is Uint16List) {
      dataType = ONNXTensorElementDataType.uint16;
      elementSize = 2;
      final ptr = _getOrAllocateBuffer<ffi.Uint16>(dataType, dataSize, (size) => calloc<ffi.Uint16>(size));
      ptr.asTypedList(dataSize).setRange(0, dataSize, data);
      dataPtr = ptr.cast<ffi.Void>();
    } else if (data is Int32List) {
      dataType = ONNXTensorElementDataType.int32;
      elementSize = 4;
      final ptr = _getOrAllocateBuffer<ffi.Int32>(dataType, dataSize, (size) => calloc<ffi.Int32>(size));
      ptr.asTypedList(dataSize).setRange(0, dataSize, data);
      dataPtr = ptr.cast<ffi.Void>();
    } else if (data is Uint32List) {
      dataType = ONNXTensorElementDataType.uint32;
      elementSize = 4;
      final ptr = _getOrAllocateBuffer<ffi.Uint32>(dataType, dataSize, (size) => calloc<ffi.Uint32>(size));
      ptr.asTypedList(dataSize).setRange(0, dataSize, data);
      dataPtr = ptr.cast<ffi.Void>();
    } else if (data is Int64List) {
      dataType = ONNXTensorElementDataType.int64;
      elementSize = 8;
      final ptr = _getOrAllocateBuffer<ffi.Int64>(dataType, dataSize, (size) => calloc<ffi.Int64>(size));
      ptr.asTypedList(dataSize).setRange(0, dataSize, data);
      dataPtr = ptr.cast<ffi.Void>();
    } else if (data is Uint64List) {
      dataType = ONNXTensorElementDataType.uint64;
      elementSize = 8;
      final ptr = _getOrAllocateBuffer<ffi.Uint64>(dataType, dataSize, (size) => calloc<ffi.Uint64>(size));
      ptr.asTypedList(dataSize).setRange(0, dataSize, data);
      dataPtr = ptr.cast<ffi.Void>();
    } else {
      throw Exception('Unsupported TypedData type: ${data.runtimeType}');
    }
    final dataByteCount = dataSize * elementSize;

    // Create shape pointer
    final shapeSize = shape.length;
    final shapePtr = calloc<ffi.Int64>(shapeSize);
    shapePtr.asTypedList(shapeSize).setAll(0, shape);

    final ortMemoryInfoPtrPtr = calloc<ffi.Pointer<bg.OrtMemoryInfo>>();
    var statusPtr = OrtEnv.instance.ortApiPtr.ref.AllocatorGetInfo.asFunction<
            bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtAllocator>,
                ffi.Pointer<ffi.Pointer<bg.OrtMemoryInfo>>)>()(
        OrtAllocator.instance.ptr, ortMemoryInfoPtrPtr);
    OrtStatus.checkOrtStatus(statusPtr);

    final ortMemoryInfoPtr = ortMemoryInfoPtrPtr.value;
    final ortValuePtrPtr = calloc<ffi.Pointer<bg.OrtValue>>();

    statusPtr = OrtEnv.instance.ortApiPtr.ref.CreateTensorWithDataAsOrtValue
            .asFunction<
                bg.OrtStatusPtr Function(
                    ffi.Pointer<bg.OrtMemoryInfo>,
                    ffi.Pointer<ffi.Void>,
                    int,
                    ffi.Pointer<ffi.Int64>,
                    int,
                    int,
                    ffi.Pointer<ffi.Pointer<bg.OrtValue>>)>()(
        ortMemoryInfoPtr,
        dataPtr,
        dataByteCount,
        shapePtr,
        shapeSize,
        dataType.value,
        ortValuePtrPtr);
    OrtStatus.checkOrtStatus(statusPtr);

    final ortValuePtr = ortValuePtrPtr.value;
    calloc.free(shapePtr);
    calloc.free(ortValuePtrPtr);
    calloc.free(ortMemoryInfoPtrPtr);

    return OrtValueTensor(ortValuePtr, null);
  }

  /// Dispose all reusable buffers created by createTensorWithReusableBuffer
  /// Call this when you're completely done with tensor creation to free memory
  static void disposeReusableBuffers() {
    for (final buffer in _bufferCache.values) {
      calloc.free(buffer.ptr);
    }
    _bufferCache.clear();
  }

  static OrtValueTensor createTensorWithDataList(List data,
      [List<int>? shape]) {
    shape ??= data.shape;
    final element = data.element();
    var dataType = ONNXTensorElementDataType.undefined;
    ffi.Pointer<ffi.Void> dataPtr = ffi.nullptr;
    int dataSize = 0;
    int dataByteCount = 0;
    if (element is Uint8List) {
      // Optimize: skip flatten if data is already a typed list
      final flattenData = data is Uint8List ? data : data.flatten<int>();
      dataSize = flattenData.length;
      dataType = ONNXTensorElementDataType.uint8;
      dataPtr = (calloc<ffi.Uint8>(dataSize)
            ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
          .cast();
      dataByteCount = dataSize;
    } else if (element is Int8List) {
      final flattenData = data is Int8List ? data : data.flatten<int>();
      dataSize = flattenData.length;
      dataType = ONNXTensorElementDataType.int8;
      dataPtr = (calloc<ffi.Int8>(dataSize)
            ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
          .cast();
      dataByteCount = dataSize;
    } else if (element is Uint16List) {
      final flattenData = data is Uint16List ? data : data.flatten<int>();
      dataSize = flattenData.length;
      dataType = ONNXTensorElementDataType.uint16;
      dataPtr = (calloc<ffi.Uint16>(dataSize)
            ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
          .cast();
      dataByteCount = dataSize * 2;
    } else if (element is Int16List) {
      final flattenData = data is Int16List ? data : data.flatten<int>();
      dataSize = flattenData.length;
      dataType = ONNXTensorElementDataType.int16;
      dataPtr = (calloc<ffi.Int16>(dataSize)
            ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
          .cast();
      dataByteCount = dataSize * 2;
    } else if (element is Uint32List) {
      final flattenData = data is Uint32List ? data : data.flatten<int>();
      dataSize = flattenData.length;
      dataType = ONNXTensorElementDataType.uint32;
      dataPtr = (calloc<ffi.Uint32>(dataSize)
            ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
          .cast();
      dataByteCount = dataSize * 4;
    } else if (element is Int32List) {
      final flattenData = data is Int32List ? data : data.flatten<int>();
      dataSize = flattenData.length;
      dataType = ONNXTensorElementDataType.int32;
      dataPtr = (calloc<ffi.Int32>(dataSize)
            ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
          .cast();
      dataByteCount = dataSize * 4;
    } else if (element is Uint64List) {
      final flattenData = data is Uint64List ? data : data.flatten<int>();
      dataSize = flattenData.length;
      dataType = ONNXTensorElementDataType.uint64;
      dataPtr = (calloc<ffi.Uint64>(dataSize)
            ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
          .cast();
      dataByteCount = dataSize * 8;
    } else if (element is Int64List || element is int) {
      final flattenData = data is Int64List ? data : data.flatten<int>();
      dataSize = flattenData.length;
      dataType = ONNXTensorElementDataType.int64;
      dataPtr = (calloc<ffi.Int64>(dataSize)
            ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
          .cast();
      dataByteCount = dataSize * 8;
    } else if (element is Float32List) {
      final flattenData = data is Float32List ? data : data.flatten<double>();
      dataSize = flattenData.length;
      dataType = ONNXTensorElementDataType.float;
      dataPtr = (calloc<ffi.Float>(dataSize)
            ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
          .cast();
      dataByteCount = dataSize * 4;
    } else if (element is Float64List || element is double) {
      final flattenData = data is Float64List ? data : data.flatten<double>();
      dataSize = flattenData.length;
      dataType = ONNXTensorElementDataType.double;
      dataPtr = (calloc<ffi.Double>(dataSize)
            ..asTypedList(dataSize).setRange(0, dataSize, flattenData))
          .cast();
      dataByteCount = dataSize * 8;
    } else if (element is bool) {
      final flattenData = data.flatten<bool>();
      dataSize = flattenData.length;
      dataType = ONNXTensorElementDataType.bool;
      final ptr = calloc<ffi.Bool>(dataSize);
      for (int i = 0; i < dataSize; ++i) {
        ptr[i] = flattenData[i];
      }
      dataPtr = ptr.cast();
      dataByteCount = dataSize;
    } else if (element is String) {
      return _createTensorWithStringList(data.cast<String>(), shape);
    } else {
      throw Exception('Invalid inputTensor element type.');
    }

    final shapeSize = shape.length;
    final shapePtr = calloc<ffi.Int64>(shapeSize);
    shapePtr.asTypedList(shapeSize).setRange(0, shapeSize, shape);

    final ortMemoryInfoPtrPtr = calloc<ffi.Pointer<bg.OrtMemoryInfo>>();
    var statusPtr = OrtEnv.instance.ortApiPtr.ref.AllocatorGetInfo.asFunction<
            bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtAllocator>,
                ffi.Pointer<ffi.Pointer<bg.OrtMemoryInfo>>)>()(
        OrtAllocator.instance.ptr, ortMemoryInfoPtrPtr);
    OrtStatus.checkOrtStatus(statusPtr);
    // or
    // OrtEnv.instance.ortApiPtr.ref.CreateCpuMemoryInfo.asFunction<
    //         bg.OrtStatusPtr Function(
    //             int, int, ffi.Pointer<ffi.Pointer<bg.OrtMemoryInfo>>)>()(
    //     bg.OrtAllocatorType.OrtDeviceAllocator,
    //     bg.OrtMemType.OrtMemTypeCPU,
    //     ortMemoryInfoPtrPtr);
    final ortMemoryInfoPtr = ortMemoryInfoPtrPtr.value;
    final ortValuePtrPtr = calloc<ffi.Pointer<bg.OrtValue>>();
    statusPtr = OrtEnv.instance.ortApiPtr.ref.CreateTensorWithDataAsOrtValue
            .asFunction<
                bg.OrtStatusPtr Function(
                    ffi.Pointer<bg.OrtMemoryInfo>,
                    ffi.Pointer<ffi.Void>,
                    int,
                    ffi.Pointer<ffi.Int64>,
                    int,
                    int,
                    ffi.Pointer<ffi.Pointer<bg.OrtValue>>)>()(
        ortMemoryInfoPtr,
        dataPtr,
        dataByteCount,
        shapePtr,
        shapeSize,
        dataType.value,
        ortValuePtrPtr);
    OrtStatus.checkOrtStatus(statusPtr);
    final ortValuePtr = ortValuePtrPtr.value;
    calloc.free(shapePtr);
    calloc.free(ortValuePtrPtr);
    calloc.free(ortMemoryInfoPtrPtr);
    return OrtValueTensor(ortValuePtr, dataPtr);
  }

  @override
  dynamic get value {
    if (_info._dimensionsCount == 0) {
      // scalar tensor
      switch (_info._tensorElementType) {
        case ONNXTensorElementDataType.uint8:
        case ONNXTensorElementDataType.int8:
        case ONNXTensorElementDataType.uint16:
        case ONNXTensorElementDataType.int16:
        case ONNXTensorElementDataType.uint32:
        case ONNXTensorElementDataType.int32:
        case ONNXTensorElementDataType.uint64:
        case ONNXTensorElementDataType.int64:
        case ONNXTensorElementDataType.float:
        case ONNXTensorElementDataType.double:
          return _getNumList(_ptr)[0];
        case ONNXTensorElementDataType.bool:
          return _getBoolList(_ptr)[0];
        case ONNXTensorElementDataType.string:
          return _getStringList(_ptr)[0];
        default:
          throw Exception('Extracting the value of an invalid Tensor.');
      }
    } else {
      // vector tensor
      switch (_info._tensorElementType) {
        case ONNXTensorElementDataType.uint8:
        case ONNXTensorElementDataType.int8:
        case ONNXTensorElementDataType.uint16:
        case ONNXTensorElementDataType.int16:
        case ONNXTensorElementDataType.uint32:
        case ONNXTensorElementDataType.int32:
        case ONNXTensorElementDataType.uint64:
        case ONNXTensorElementDataType.int64:
          return _getNumList(_ptr).reshape<int>(_info._tensorShape);
        case ONNXTensorElementDataType.float:
        case ONNXTensorElementDataType.double:
          return _getNumList(_ptr).reshape<double>(_info._tensorShape);
        case ONNXTensorElementDataType.bool:
          return _getBoolList(_ptr).reshape<bool>(_info._tensorShape);
        case ONNXTensorElementDataType.string:
          return _getStringList(_ptr).reshape<String>(_info._tensorShape);
        default:
          throw Exception('Extracting the value of an invalid Tensor.');
      }
    }
  }

  @override
  void release() {
    super.release();
    if (_dataPtr == ffi.nullptr) {
      return;
    }
    calloc.free(_dataPtr);
    _dataPtr = ffi.nullptr;
  }
}

class OrtValueSequence extends OrtValue {
  int _valueCount = 0;
  var _onnxType = ONNXType.unknown;
  OrtTensorTypeAndShapeInfo? _tensorInfo;

  // OrtTensorTypeAndShapeInfo? _firstMapKeyInfo;
  // OrtTensorTypeAndShapeInfo? _firstMapValueInfo;

  OrtValueSequence(ffi.Pointer<bg.OrtValue> ptr) {
    _ptr = ptr;
    final valueCountPtr = calloc<ffi.Size>();
    var statusPtr = OrtEnv.instance.ortApiPtr.ref.GetValueCount.asFunction<
        bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
            ffi.Pointer<ffi.Size>)>()(_ptr, valueCountPtr);
    OrtStatus.checkOrtStatus(statusPtr);
    _valueCount = valueCountPtr.value;
    calloc.free(valueCountPtr);
    if (_valueCount <= 0) {
      return;
    }
    final firstElementPtrPtr = calloc<ffi.Pointer<bg.OrtValue>>();
    final firstElementPtr = _getOrtValue(_ptr, 0, firstElementPtrPtr);
    final onnxTypePtr = calloc<ffi.Int32>();
    statusPtr = OrtEnv.instance.ortApiPtr.ref.GetValueType.asFunction<
        bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
            ffi.Pointer<ffi.Int32>)>()(firstElementPtr, onnxTypePtr);
    OrtStatus.checkOrtStatus(statusPtr);
    _onnxType = ONNXType.valueOf(onnxTypePtr.value);
    if (_onnxType == ONNXType.tensor) {
      _tensorInfo = OrtTensorTypeAndShapeInfo(firstElementPtr);
    } else if (_onnxType == ONNXType.map) {
      // final infoMap = _createMapInfo(firstElementPtr);
      // _firstMapKeyInfo = infoMap.entries.first.key;
      // _firstMapValueInfo = infoMap.entries.first.value;
    }
    _releaseOrtValue(firstElementPtr);
    calloc.free(firstElementPtrPtr);
    calloc.free(onnxTypePtr);
  }

  OrtValueSequence.fromAddress(int address) {
    _ptr = ffi.Pointer.fromAddress(address);
  }

  @override
  List<OrtValue>? get value {
    if (_onnxType == ONNXType.map) {
      final maps = <OrtValueMap>[];
      for (int i = 0; i < _valueCount; ++i) {
        final ortValuePtrPtr = calloc<ffi.Pointer<bg.OrtValue>>();
        final ortValuePtr = _getOrtValue(_ptr, i, ortValuePtrPtr);
        maps.add(OrtValueMap(ortValuePtr));
        calloc.free(ortValuePtrPtr);
      }
      return maps;
    } else if (_onnxType == ONNXType.tensor) {
      switch (_tensorInfo?._tensorElementType) {
        case ONNXTensorElementDataType.string:
        case ONNXTensorElementDataType.int64:
        case ONNXTensorElementDataType.float:
        case ONNXTensorElementDataType.double:
          final tensors = <OrtValueTensor>[];
          for (int i = 0; i < _valueCount; ++i) {
            final ortValuePtrPtr = calloc<ffi.Pointer<bg.OrtValue>>();
            final ortValuePtr = _getOrtValue(_ptr, i, ortValuePtrPtr);
            tensors.add(OrtValueTensor(ortValuePtr));
            calloc.free(ortValuePtrPtr);
          }
          return tensors;
        default:
          throw Exception(
              'Unsupported type in a sequence, found ${_tensorInfo?._tensorElementType}');
      }
    } else {
      throw Exception("Invalid element type found in sequence");
    }
  }
}

class OrtValueMap extends OrtValue {
  late OrtTensorTypeAndShapeInfo _keyInfo;
  late OrtTensorTypeAndShapeInfo _valueInfo;

  OrtValueMap(ffi.Pointer<bg.OrtValue> ptr) {
    _ptr = ptr;
    final infoMap = _createMapInfo(ptr);
    _keyInfo = infoMap.entries.first.key;
    _valueInfo = infoMap.entries.first.value;
  }

  OrtValueMap.fromAddress(int address) {
    _ptr = ffi.Pointer.fromAddress(address);
  }

  @override
  Map get value {
    final keys = _getMapKeys();
    final values = _getMapValues();
    final map = {};
    for (int i = 0; i < keys.length; ++i) {
      map[keys[i]] = values[i];
    }
    return map;
  }

  List<dynamic> _getMapKeys() {
    switch (_keyInfo._tensorElementType) {
      case ONNXTensorElementDataType.string:
        return _getStringListWithIndex(0);
      case ONNXTensorElementDataType.int64:
        return _getNumListWithIndex(0);
      default:
        throw Exception(
            'Invalid or unknown valueType: ${_keyInfo._tensorElementType}');
    }
  }

  List<String> _getStringListWithIndex(int index) {
    final ortValuePtrPtr = calloc<ffi.Pointer<bg.OrtValue>>();
    final ortValuePtr = _getOrtValue(_ptr, index, ortValuePtrPtr);
    final list = _getStringList(ortValuePtr);
    _releaseOrtValue(ortValuePtr);
    calloc.free(ortValuePtrPtr);
    return list;
  }

  List<num> _getNumListWithIndex(int index) {
    final ortValuePtrPtr = calloc<ffi.Pointer<bg.OrtValue>>();
    final ortValuePtr = _getOrtValue(_ptr, index, ortValuePtrPtr);
    final list = _getNumList(ortValuePtr);
    _releaseOrtValue(ortValuePtr);
    calloc.free(ortValuePtrPtr);
    return list;
  }

  List<Object> _getMapValues() {
    switch (_valueInfo._tensorElementType) {
      case ONNXTensorElementDataType.string:
        return _getStringListWithIndex(1);
      case ONNXTensorElementDataType.int64:
      case ONNXTensorElementDataType.float:
      case ONNXTensorElementDataType.double:
        return _getNumListWithIndex(1);
      default:
        throw Exception(
            'Invalid or unknown valueType: ${_keyInfo._tensorElementType}');
    }
  }

  int get size => _keyInfo._tensorShapeElementCount;
}

class OrtValueSparseTensor extends OrtValue {
  // ignore: unused_field
  late OrtTensorTypeAndShapeInfo _info;
  late OrtSparseFormat _ortSparseFormat;

  OrtValueSparseTensor(ffi.Pointer<bg.OrtValue> ptr) {
    _ptr = ptr;
    _info = OrtTensorTypeAndShapeInfo(ptr);
    final ortSparseFormatPtr = calloc<ffi.Int32>();
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetSparseTensorFormat
        .asFunction<
            bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
                ffi.Pointer<ffi.Int32>)>()(ptr, ortSparseFormatPtr);
    OrtStatus.checkOrtStatus(statusPtr);
    _ortSparseFormat = OrtSparseFormat.valueOf(ortSparseFormatPtr.value);
    calloc.free(ortSparseFormatPtr);
  }

  OrtValueSparseTensor.fromAddress(int address) {
    _ptr = ffi.Pointer.fromAddress(address);
  }

  @override
  // ignore: body_might_complete_normally_nullable
  Object? get value {
    switch (_ortSparseFormat) {
      case OrtSparseFormat.coo:
        // TODO: Handle this case.
        break;
      case OrtSparseFormat.csrc:
        // TODO: Handle this case.
        break;
      case OrtSparseFormat.blockSparse:
        // TODO: Handle this case.
        break;
      case OrtSparseFormat.undefined:
        throw Exception('Undefined sparsity type in this sparse tensor.');
    }
  }
}

class OrtTensorTypeAndShapeInfo {
  int _dimensionsCount = 0;
  int _tensorShapeElementCount = 0;
  ONNXTensorElementDataType _tensorElementType =
      ONNXTensorElementDataType.undefined;
  List<int> _tensorShape = [];

  OrtTensorTypeAndShapeInfo(ffi.Pointer<bg.OrtValue> ortValuePtr) {
    final infoPtrPtr = calloc<ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>>();
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetTensorTypeAndShape
            .asFunction<
                bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
                    ffi.Pointer<ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>>)>()(
        ortValuePtr, infoPtrPtr);
    OrtStatus.checkOrtStatus(statusPtr);
    final infoPtr = infoPtrPtr.value;
    _tensorElementType = _getTensorElementType(infoPtr);
    // shape
    _dimensionsCount = _getDimensionsCount(infoPtr);
    _tensorShape = _getDimensions(infoPtr, _dimensionsCount);
    _tensorShapeElementCount = _getTensorShapeElementCount(infoPtr);
    _releaseTensorTypeAndShapeInfo(infoPtr);
    calloc.free(infoPtrPtr);
  }

  static ONNXTensorElementDataType _getTensorElementType(
      ffi.Pointer<bg.OrtTensorTypeAndShapeInfo> infoPtr) {
    final onnxTensorElementDataTypePtr = calloc<ffi.Int32>();
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetTensorElementType
            .asFunction<
                bg.OrtStatusPtr Function(
                    ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>,
                    ffi.Pointer<ffi.Int32>)>()(
        infoPtr, onnxTensorElementDataTypePtr);
    OrtStatus.checkOrtStatus(statusPtr);
    final onnxTensorElementDataType = onnxTensorElementDataTypePtr.value;
    calloc.free(onnxTensorElementDataTypePtr);
    return ONNXTensorElementDataType.valueOf(onnxTensorElementDataType);
  }

  static void _releaseTensorTypeAndShapeInfo(
      ffi.Pointer<bg.OrtTensorTypeAndShapeInfo> infoPtr) {
    OrtEnv.instance.ortApiPtr.ref.ReleaseTensorTypeAndShapeInfo.asFunction<
        void Function(ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>)>()(infoPtr);
  }

  static int _getDimensionsCount(
      ffi.Pointer<bg.OrtTensorTypeAndShapeInfo> infoPtr) {
    final countPtr = calloc<ffi.Size>();
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetDimensionsCount
        .asFunction<
            bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>,
                ffi.Pointer<ffi.Size>)>()(infoPtr, countPtr);
    OrtStatus.checkOrtStatus(statusPtr);
    final count = countPtr.value;
    calloc.free(countPtr);
    return count;
  }

  static List<int> _getDimensions(
      ffi.Pointer<bg.OrtTensorTypeAndShapeInfo> infoPtr, int length) {
    final dimensionsPtr = calloc<ffi.Int64>(length);
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetDimensions.asFunction<
        bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>,
            ffi.Pointer<ffi.Int64>, int)>()(infoPtr, dimensionsPtr, length);
    OrtStatus.checkOrtStatus(statusPtr);
    final dimensions =
        List<int>.generate(length, (index) => dimensionsPtr[index]);
    calloc.free(dimensionsPtr);
    return dimensions;
  }

  static int _getTensorShapeElementCount(
      ffi.Pointer<bg.OrtTensorTypeAndShapeInfo> infoPtr) {
    final countPtr = calloc<ffi.Size>();
    final statusPtr = OrtEnv.instance.ortApiPtr.ref.GetTensorShapeElementCount
        .asFunction<
            bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtTensorTypeAndShapeInfo>,
                ffi.Pointer<ffi.Size>)>()(infoPtr, countPtr);
    OrtStatus.checkOrtStatus(statusPtr);
    final count = countPtr.value;
    calloc.free(countPtr);
    return count;
  }
}

enum ONNXTensorElementDataType {
  undefined(
      bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED),
  float(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
  uint8(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
  int8(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8),
  uint16(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16),
  int16(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
  int32(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
  int64(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
  string(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING),
  bool(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL),
  float16(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
  double(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
  uint32(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
  uint64(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
  complex64(
      bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64),
  complex128(
      bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128),
  bFloat16(bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);

  final int value;

  const ONNXTensorElementDataType(this.value);

  static ONNXTensorElementDataType valueOf(int type) {
    switch (type) {
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return ONNXTensorElementDataType.float;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return ONNXTensorElementDataType.uint8;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return ONNXTensorElementDataType.int8;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return ONNXTensorElementDataType.uint16;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return ONNXTensorElementDataType.int16;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return ONNXTensorElementDataType.int32;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return ONNXTensorElementDataType.int64;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return ONNXTensorElementDataType.string;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return ONNXTensorElementDataType.bool;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return ONNXTensorElementDataType.float16;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return ONNXTensorElementDataType.double;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return ONNXTensorElementDataType.uint32;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return ONNXTensorElementDataType.uint64;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        return ONNXTensorElementDataType.complex64;
      case bg
            .ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        return ONNXTensorElementDataType.complex128;
      case bg.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return ONNXTensorElementDataType.bFloat16;
      default:
        return ONNXTensorElementDataType.undefined;
    }
  }
}

enum ONNXType {
  unknown(bg.ONNXType.ONNX_TYPE_UNKNOWN),
  tensor(bg.ONNXType.ONNX_TYPE_TENSOR),
  sequence(bg.ONNXType.ONNX_TYPE_SEQUENCE),
  map(bg.ONNXType.ONNX_TYPE_MAP),
  opaque(bg.ONNXType.ONNX_TYPE_OPAQUE),
  sparseTensor(bg.ONNXType.ONNX_TYPE_SPARSETENSOR),
  optional(bg.ONNXType.ONNX_TYPE_OPTIONAL);

  final int value;

  const ONNXType(this.value);

  static ONNXType valueOf(int type) {
    switch (type) {
      case bg.ONNXType.ONNX_TYPE_TENSOR:
        return ONNXType.tensor;
      case bg.ONNXType.ONNX_TYPE_SEQUENCE:
        return ONNXType.sequence;
      case bg.ONNXType.ONNX_TYPE_MAP:
        return ONNXType.map;
      case bg.ONNXType.ONNX_TYPE_OPAQUE:
        return ONNXType.opaque;
      case bg.ONNXType.ONNX_TYPE_SPARSETENSOR:
        return ONNXType.sparseTensor;
      case bg.ONNXType.ONNX_TYPE_OPTIONAL:
        return ONNXType.optional;
      default:
        return ONNXType.unknown;
    }
  }
}

enum OrtSparseFormat {
  undefined(bg.OrtSparseFormat.ORT_SPARSE_UNDEFINED),
  coo(bg.OrtSparseFormat.ORT_SPARSE_COO),
  csrc(bg.OrtSparseFormat.ORT_SPARSE_CSRC),
  blockSparse(bg.OrtSparseFormat.ORT_SPARSE_BLOCK_SPARSE);

  final int value;

  const OrtSparseFormat(this.value);

  static OrtSparseFormat valueOf(int type) {
    switch (type) {
      case bg.OrtSparseFormat.ORT_SPARSE_COO:
        return OrtSparseFormat.coo;
      case bg.OrtSparseFormat.ORT_SPARSE_CSRC:
        return OrtSparseFormat.csrc;
      case bg.OrtSparseFormat.ORT_SPARSE_BLOCK_SPARSE:
        return OrtSparseFormat.blockSparse;
      default:
        return OrtSparseFormat.undefined;
    }
  }
}
