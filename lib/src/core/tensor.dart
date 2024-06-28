import 'dart:math' as math;

class Tensor {
  List<double> data;
  List<int> shape;
  late List<int> strides;
  late int size;

  Tensor(this.data, this.shape) {
    _computeStrides();
    size = data.length;
  }

  void _computeStrides() {
    strides = List<int>.filled(shape.length, 1);
    for (int i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }

  factory Tensor.zeros(List<int> shape) {
    int size = shape.reduce((a, b) => a * b);
    return Tensor(List<double>.filled(size, 0.0), shape);
  }

  factory Tensor.zeroslike(Tensor tensor){
    return Tensor.zeros(tensor.shape);
  }
  
  factory Tensor.oneslike(Tensor tensor){
    return Tensor.ones(tensor.shape);
  }

  factory Tensor.ones(List<int> shape) {
    int size = shape.reduce((a, b) => a * b);
    return Tensor(List<double>.filled(size, 1.0), shape);
  }

  factory Tensor.random(List<int> shape) {
    int size = shape.reduce((a, b) => a * b);
    var rng = math.Random();
    return Tensor(List<double>.generate(size, (_) => rng.nextDouble()), shape);
  }

  double operator [](List<int> indices) {
    int flatIndex = _flattenIndices(indices);
    return data[flatIndex];
  }

  void operator []=(List<int> indices, double value) {
    int flatIndex = _flattenIndices(indices);
    data[flatIndex] = value;
  }

  int _flattenIndices(List<int> indices) {
    if (indices.length != shape.length) {
      throw ArgumentError('Indices must match tensor dimensions');
    }
    int flatIndex = 0;
    for (int i = 0; i < indices.length; i++) {
      if (indices[i] < 0 || indices[i] >= shape[i]) {
        throw ArgumentError('Index out of range');
      }
      flatIndex += indices[i] * strides[i];
    }
    return flatIndex;
  }

  bool _areShapesEqual(List<int> shape1, List<int> shape2) {
    if (shape1.length != shape2.length) return false;
    for (int i = 0; i < shape1.length; i++) {
      if (shape1[i] != shape2[i]) return false;
    }
    return true;
  }

  Tensor add(Tensor other) {
    if (!_areShapesEqual(shape, other.shape)) {
      throw ArgumentError('Tensors must have the same shape for addition');
    }
    var resultData =
        List<double>.generate(size, (i) => data[i] + other.data[i]);
    return Tensor(resultData, shape);
  }

  Tensor multiply(Tensor other) {
    if (!_areShapesEqual(shape, other.shape)) {
      throw ArgumentError(
          'Tensors must have the same shape for element-wise multiplication');
    }
    var resultData =
        List<double>.generate(size, (i) => data[i] * other.data[i]);
    return Tensor(resultData, shape);
  }

  Tensor subtract(Tensor other) {
    if (!_areShapesEqual(shape, other.shape)) {
      throw ArgumentError('Tensors must have the same shape for subtraction');
    }
    var resultData =
        List<double>.generate(size, (i) => data[i] - other.data[i]);
    return Tensor(resultData, shape);
  }


  Tensor matmul(Tensor other) {
    if (shape.length != 2 || other.shape.length != 2) {
      throw ArgumentError(
          'Matrix multiplication is only defined for 2D tensors');
    }
    if (shape[1] != other.shape[0]) {
      throw ArgumentError(
          'Inner tensor dimensions must match for matrix multiplication');
    }
    var resultShape = [shape[0], other.shape[1]];
    var resultData = List<double>.filled(resultShape[0] * resultShape[1], 0.0);
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < other.shape[1]; j++) {
        double sum = 0.0;
        for (int k = 0; k < shape[1]; k++) {
          sum += this[[i, k]] * other[[k, j]];
        }
        resultData[i * resultShape[1] + j] = sum;
      }
    }
    return Tensor(resultData, resultShape);
  }

@override
  String toString() {
    var buffer = StringBuffer();
    buffer.write('Tensor(array([');

    for (int i = 0; i < shape[0]; i++) {
      buffer.write('[');
      for (int j = 0; j < shape[1]; j++) {
        buffer.write(data[_flattenIndices([i, j])].toStringAsFixed(5));
        if (j < shape[1] - 1) {
          buffer.write(', ');
        }
      }
      buffer.write(']');
      if (i < shape[0] - 1) {
        buffer.write(',\n       ');
      }
    }

    buffer.write('], dtype=double)');
    return buffer.toString();
  }
}
