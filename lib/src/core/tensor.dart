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

  /// Compute the strides of the tensor.
  void _computeStrides() {
    strides = List<int>.filled(shape.length, 1);
    for (int i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }

  int _flattenIndicesWithStrides(List<int> indices, List<int> strides) {
    int flatIndex = 0;
    for (int i = 0; i < indices.length; i++) {
      flatIndex += indices[i] * strides[i];
    }
    return flatIndex;
  }

  /// Creates a tensor of zeros with the given shape.
  factory Tensor.zeros(List<int> shape) {
    int size = shape.reduce((a, b) => a * b);
    return Tensor(List<double>.filled(size, 0.0), shape);
  }

  /// Creates a tensor of zeros with the same shape as the given tensor.
  factory Tensor.zeroslike(Tensor tensor) {
    return Tensor.zeros(tensor.shape);
  }

  /// Creates a tensor of ones with the same shape as the given tensor.
  factory Tensor.oneslike(Tensor tensor) {
    return Tensor.ones(tensor.shape);
  }

  /// Creates a tensor of ones with the given shape.
  factory Tensor.ones(List<int> shape) {
    int size = shape.reduce((a, b) => a * b);
    return Tensor(List<double>.filled(size, 1.0), shape);
  }

  /// Creates a tensor with random values between 0 and 1 with the given shape.
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

  /// UnFlattens the indices of the tensor.
  List<int> _unflattenIndex(int flatIndex) {
    List<int> indices = List<int>.filled(shape.length, 0);

    for (int i = 0; i < shape.length; i++) {
      indices[i] = (flatIndex ~/ strides[i]) % shape[i];
    }

    return indices;
  }

  bool _areShapesEqual(List<int> shape1, List<int> shape2) {
    if (shape1.length != shape2.length) return false;
    for (int i = 0; i < shape1.length; i++) {
      if (shape1[i] != shape2[i]) return false;
    }
    return true;
  }

  /// Adds two tensors element-wise.
  Tensor add(Tensor other) {
    if (!_areShapesEqual(shape, other.shape)) {
      throw ArgumentError('Tensors must have the same shape for addition');
    }
    var resultData =
        List<double>.generate(size, (i) => data[i] + other.data[i]);
    return Tensor(resultData, shape);
  }

  /// Multiplies two tensors element-wise.
  Tensor multiply(Tensor other) {
    if (!_areShapesEqual(shape, other.shape)) {
      throw ArgumentError(
          'Tensors must have the same shape for element-wise multiplication');
    }
    var resultData =
        List<double>.generate(size, (i) => data[i] * other.data[i]);
    return Tensor(resultData, shape);
  }

  /// Subtracts two tensors element-wise.
  Tensor subtract(Tensor other) {
    if (!_areShapesEqual(shape, other.shape)) {
      throw ArgumentError('Tensors must have the same shape for subtraction');
    }
    var resultData =
        List<double>.generate(size, (i) => data[i] - other.data[i]);
    return Tensor(resultData, shape);
  }

  /// Matrix multiplication of a tensor with a given tensor.
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

  /// Add a scalar to a tensor element-wise.
  Tensor addScalar(double scalar) {
    var resultData = List<double>.generate(size, (i) => data[i] + scalar);
    return Tensor(resultData, shape);
  }

  /// Multiply a tensor by a scalar element-wise.
  Tensor multiplyScalar(double scalar) {
    var resultData = List<double>.generate(size, (i) => data[i] * scalar);
    return Tensor(resultData, shape);
  }

  /// Subtract a scalar from a tensor element-wise.
  Tensor subtractScalar(double scalar) {
    var resultData = List<double>.generate(size, (i) => data[i] - scalar);
    return Tensor(resultData, shape);
  }

  /// Divide a tensor by a scalar element-wise.
  Tensor divideScalar(double scalar) {
    var resultData = List<double>.generate(size, (i) => data[i] / scalar);
    return Tensor(resultData, shape);
  }

  /// Dot product by another tensor.
  // Tensor dotProduct(Tensor other){

  // }
  //! Linear Algebra operations

  /// Transpose of a Tensor
  Tensor transpose() {
    if (shape.length != 2) {
      throw ArgumentError('Transpose is only implemented for 2D tensors');
    }

    int rows = shape[0];
    int cols = shape[1];
    List<double> transposedData = List<double>.filled(rows * cols, 0.0);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        transposedData[j * rows + i] = data[i * cols + j];
      }
    }
    return Tensor(transposedData, [cols, rows]);
  }

  /// Computes the inverse of a 2D tensor using the Gauss-Jordan elimination method.
  Tensor inverse() {
    if (shape.length != 2 || shape[0] != shape[1]) {
      throw ArgumentError('Inverse is only implemented for square 2D tensors');
    }

    int n = shape[0];
    var augmented = List<List<double>>.generate(n, (i) {
      return List<double>.generate(2 * n, (j) {
        if (j < n) {
          return data[i * n + j];
        } else if (j - n == i) {
          return 1.0;
        } else {
          return 0.0;
        }
      });
    });

    for (int i = 0; i < n; i++) {
      double pivot = augmented[i][i];
      if (pivot == 0.0) {
        throw ArgumentError('Matrix is singular and cannot be inverted');
      }

      for (int j = 0; j < 2 * n; j++) {
        augmented[i][j] /= pivot;
      }

      for (int k = 0; k < n; k++) {
        if (k != i) {
          double factor = augmented[k][i];
          for (int j = 0; j < 2 * n; j++) {
            augmented[k][j] -= factor * augmented[i][j];
          }
        }
      }
    }

    var resultData = List<double>.filled(n * n, 0.0);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        resultData[i * n + j] = augmented[i][j + n];
      }
    }

    return Tensor(resultData, shape);
  }

  /// Computes the Determinant of a Tensor
  double determinant() {
    if (shape.length != 2 || shape[0] != shape[1]) {
      throw ArgumentError(
          'Determinant is only implemented for square 2D tensors');
    }

    int n = shape[0];
    var matrix = List<List<double>>.generate(n, (i) {
      return List<double>.generate(n, (j) {
        return data[i * n + j];
      });
    });

    double det = 1.0;
    for (int i = 0; i < n; i++) {
      if (matrix[i][i] == 0.0) {
        // Find a row to swap with
        bool found = false;
        for (int k = i + 1; k < n; k++) {
          if (matrix[k][i] != 0.0) {
            var temp = matrix[i];
            matrix[i] = matrix[k];
            matrix[k] = temp;
            det = -det;
            found = true;
            break;
          }
        }
        if (!found) {
          return 0.0;
        }
      }

      det *= matrix[i][i];

      for (int j = i + 1; j < n; j++) {
        matrix[i][j] /= matrix[i][i];
      }

      for (int k = i + 1; k < n; k++) {
        for (int j = i + 1; j < n; j++) {
          matrix[k][j] -= matrix[k][i] * matrix[i][j];
        }
      }
    }

    return det;
  }

  /// Computes the rank of a given tensor
  int rank() {
    if (shape.length != 2) {
      throw ArgumentError('Rank is only implemented for 2D tensors');
    }

    int numRows = shape[0];
    int numCols = shape[1];
    int minDim = math.min(numRows, numCols);

    var matrix = List<List<double>>.generate(numRows, (i) {
      return List<double>.generate(numCols, (j) {
        return data[i * numCols + j];
      });
    });

    int rank = 0;
    for (int i = 0; i < minDim; i++) {
      // Find pivot
      int pivotRow = i;
      while (pivotRow < numRows && matrix[pivotRow][i] == 0.0) {
        pivotRow++;
      }
      if (pivotRow == numRows) {
        continue; // No nonzero pivot in this column
      }

      // Swap rows if necessary
      if (pivotRow != i) {
        var temp = matrix[i];
        matrix[i] = matrix[pivotRow];
        matrix[pivotRow] = temp;
      }

      // Perform row operations to zero out the column below the pivot
      for (int j = i + 1; j < numRows; j++) {
        if (matrix[j][i] != 0.0) {
          double ratio = matrix[j][i] / matrix[i][i];
          for (int k = i; k < numCols; k++) {
            matrix[j][k] -= ratio * matrix[i][k];
          }
        }
      }

      rank++;
    }

    return rank;
  }

//! Statistical Operations

  /// Sum of the elments in the tensor
  double sum() {
    return data.reduce((a, b) => a + b);
  }

  /// Mean of the elements in the tensor
  double mean() {
    return sum() / size;
  }

  /// Variance of the elements in the tensor
  double variance({bool population = false}) {
    double meanValue = mean();
    num sumSquaredDiff =
        data.map((e) => math.pow(e - meanValue, 2)).reduce((a, b) => a + b);
    return population ? sumSquaredDiff / size : sumSquaredDiff / (size - 1);
  }

  /// Standard Deviation of the elements in the tensor
  double std({bool population = false}) {
    return math.sqrt(variance(population: population));
  }

  /// Power of the tensor element-wise to the given exponent
  Tensor pow(double exponent) {
    var resultData = List<double>.generate(
        size, (i) => math.pow(data[i], exponent).toDouble());
    return Tensor(resultData, shape);
  }

  /// Square root of the tensor element-wise
  Tensor sqrt() {
    var resultData = List<double>.generate(size, (i) => math.sqrt(data[i]));
    return Tensor(resultData, shape);
  }

  /// Exponential of the tensor element-wise
  Tensor exp() {
    var resultData = List<double>.generate(size, (i) => math.exp(data[i]));
    return Tensor(resultData, shape);
  }

  /// Logarithm of the tensor element-wise
  Tensor log() {
    var resultData = List<double>.generate(size, (i) => math.log(data[i]));
    return Tensor(resultData, shape);
  }

  /// Absolute value of the tensor element-wise
  Tensor abs() {
    var result = data.map((x) => x.abs()).toList();
    return Tensor(result, shape);
  }

  /// Sign of the tensor element-wise
  Tensor sign() {
    var result = data.map((x) => x.sign).toList();
    return Tensor(result, shape);
  }

//! Rounding Operations
  /// Returns the largest integer less than or equal to the tensor element-wise
  Tensor floor() {
    var resultData =
        List<double>.generate(size, (i) => data[i].floorToDouble());
    return Tensor(resultData, shape);
  }
  /// Returns the smallest integer greater than or equal to the tensor element-wise
  Tensor ceil() {
    var resultData = List<double>.generate(size, (i) => data[i].ceilToDouble());
    return Tensor(resultData, shape);
  }
  /// Round the tensor element-wise to the nearest integer
  Tensor round() {
    var resultData =
        List<double>.generate(size, (i) => data[i].roundToDouble());
    return Tensor(resultData, shape);
  }

//! Trignonometric Operations
  /// Sine of the tensor element-wise
  Tensor sin() {
    var resultData = List<double>.generate(size, (i) => math.sin(data[i]));
    return Tensor(resultData, shape);
  }

  /// Cosine of the tensor element-wise
  Tensor cos() {
    var resultData = List<double>.generate(size, (i) => math.cos(data[i]));
    return Tensor(resultData, shape);
  }

  /// Tangent of the tensor element-wise
  Tensor tan() {
    var resultData = List<double>.generate(size, (i) => math.tan(data[i]));
    return Tensor(resultData, shape);
  }

  /// Arcsine of the tensor element-wise
  Tensor asin() {
    var resultData = List<double>.generate(size, (i) => math.asin(data[i]));
    return Tensor(resultData, shape);
  }

  /// Arccosine of the tensor element-wise
  Tensor acos() {
    var resultData = List<double>.generate(size, (i) => math.acos(data[i]));
    return Tensor(resultData, shape);
  }

  /// Arctangent of the tensor element-wise
  Tensor atan() {
    var resultData = List<double>.generate(size, (i) => math.atan(data[i]));
    return Tensor(resultData, shape);
  }

  /// Hyperbolic Sine of the tensor element-wise
  Tensor sinh() {
    var result = data.map((x) => (math.exp(x) - math.exp(-x)) / 2).toList();
    return Tensor(result, shape);
  }

  /// Hyperbolic Cosine of the tensor element-wise
  Tensor cosh() {
    var result = data.map((x) => (math.exp(x) + math.exp(-x)) / 2).toList();
    return Tensor(result, shape);
  }

  /// Hyperbolic Tangent of the tensor element-wise
  Tensor tanh() {
    var result = data.map((x) {
      double exp2x = math.exp(2 * x);
      return (exp2x - 1) / (exp2x + 1);
    }).toList();
    return Tensor(result, shape);
  }


//! Aggregation Operations

  /// Returns the maximum value in the tensor
  double max() {
    return data.reduce((a, b) => math.max(a, b));
  }

  /// Returns the minimumvalue in the tensor
  double min() {
    return data.reduce((a, b) => math.min(a, b));
  }

  /// Returns the index of the maximum value in the tensor
  List<int> argmax() {
    double maxValue = double.negativeInfinity;
    List<int> maxIndex = [];
    for (int i = 0; i < size; i++) {
      if (data[i] > maxValue) {
        maxValue = data[i];
        maxIndex = _unflattenIndex(i);
      }
    }
    return maxIndex;
  }

  /// Returns the index of the minimum value in the tensor
  List<int> argmin() {
    double minValue = double.infinity;
    List<int> minIndex = [];
    for (int i = 0; i < size; i++) {
      if (data[i] < minValue) {
        minValue = data[i];
        minIndex = _unflattenIndex(i);
      }
    }
    return minIndex;
  }

  //! Reshaping Operations
  /// Reshape the tensort into the desired shape
  Tensor reshape(List<int> newShape) {
    if (size != newShape.reduce((a, b) => a * b)) {
      throw ArgumentError('New shape must have the same number of elements');
    }
    return Tensor(data, newShape);
  }

  /// Flattens the tensor into a 1D tensor
  Tensor flatten() {
    return Tensor(data, [size]);
  }

  //! Tensor Utilities

  /// Checks if two tensors are equal (element-wise comparison)
  bool equals(Tensor other) {
    if (!_areShapesEqual(shape, other.shape)) {
      return false;
    }
    for (int i = 0; i < size; i++) {
      if (data[i] != other.data[i]) {
        return false;
      }
    }
    return true;
  }

//! Advanced Operations

  /// Tensor concatenation along a given axis
  Tensor concatenate(Tensor other, {int axis = 0}) {
    // Check if the axis is within vlaid range
    if (axis < 0 || axis >= shape.length) {
      throw ArgumentError('Axis out of range');
    }

    // Check if bot htensors have the same number of dimensions
    if (shape.length != other.shape.length) {
      throw ArgumentError('Tensors must have the same number of dimensions');
    }

    // Check ifhte shapes match along all dimensions execpt the concatenation axis
    for (int i = 0; i < shape.length; i++) {
      if (i != axis && shape[i] != other.shape[i]) {
        throw ArgumentError(
            'Tensors must have the same shape except in the concatenation axis');
      }
    }
    // Calculate the new shape after concatenation
    var newShape = List<int>.from(shape);
    newShape[axis] += other.shape[axis];

    var newData = List<double>.filled(newShape.reduce((a, b) => a * b), 0);

    int stride = shape.sublist(axis + 1).fold(1, (a, b) => a * b);
    int newStride = newShape.sublist(axis + 1).fold(1, (a, b) => a * b);

    for (int i = 0; i < newData.length; i++) {
      int oldIndex = i ~/ newStride * stride + i % newStride;
      if (oldIndex < data.length) {
        newData[i] = data[oldIndex];
      } else {
        newData[i] = other.data[oldIndex - data.length];
      }
    }

    return Tensor(newData, newShape);
  }

  ///  Tensor stacking along a given axis
  Tensor stack(Tensor other, {int axis = 0}) {
    if (!_areShapesEqual(shape, other.shape)) {
      throw ArgumentError('Tensors must have the same shape for stacking');
    }

    // Add a new axis to the shape
    List<int> newShape = List<int>.from(shape);
    newShape.insert(axis, 2);

    // Compute the new strides
    List<int> newStrides = List<int>.filled(newShape.length, 1);
    for (int i = newShape.length - 2; i >= 0; i--) {
      newStrides[i] = newStrides[i + 1] * newShape[i + 1];
    }

    // Prepare the new data array
    List<double> newData = List<double>.filled(size * 2, 0.0);

    // Copy data from the first tensor
    for (int i = 0; i < size; i++) {
      List<int> oldIndices = _unflattenIndex(i);
      List<int> newIndices = List<int>.from(oldIndices);
      newIndices.insert(axis, 0);
      int newIndex = _flattenIndicesWithStrides(newIndices, newStrides);
      newData[newIndex] = data[i];
    }

    // Copy data from the second tensor
    for (int i = 0; i < size; i++) {
      List<int> oldIndices = _unflattenIndex(i);
      List<int> newIndices = List<int>.from(oldIndices);
      newIndices.insert(axis, 1);
      int newIndex = _flattenIndicesWithStrides(newIndices, newStrides);
      newData[newIndex] = other.data[i];
    }

    return Tensor(newData, newShape);
  }

  @override
  String toString() {
    return 'Tensor(${_formatData()}, dtype=double)';
  }

  String _formatData() {
    if (shape.length == 1) {
      return '[${_formatRow(data)}]';
    } else {
      List<String> rows = [];
      int rowSize = shape.last;
      for (int i = 0; i < data.length; i += rowSize) {
        rows.add(_formatRow(data.sublist(i, i + rowSize)));
      }
      String indent = ' ' * 7; // 7 spaces to align with 'Tensor('
      return '[\n$indent${rows.join(',\n$indent')}\n]';
    }
  }

  String _formatRow(List<double> row) {
    return '[${row.map((e) => e.toStringAsFixed(5)).join(', ')}]';
  }
}
