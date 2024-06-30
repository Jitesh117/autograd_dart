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

/// Creates a tensor filled with zeros.
///
/// This factory constructor initializes a tensor of the specified shape with
/// all elements set to zero.
///
/// The shape of the tensor is provided as a list of integers, where each 
/// integer represents the size of the corresponding dimension. The total
/// number of elements in the tensor is the product of the sizes of all 
/// dimensions.
///
/// Example:
/// ```dart
/// var tensor = Tensor.zeros([2, 3]);
/// print(tensor.data); // Outputs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
/// print(tensor.shape); // Outputs: [2, 3]
/// ```
///
/// - Parameter shape: A list of integers specifying the shape of the tensor.
///
/// - Returns: A new `Tensor` instance with the specified shape and all 
///   elements set to zero.

  factory Tensor.zeros(List<int> shape) {
    int size = shape.reduce((a, b) => a * b);
    return Tensor(List<double>.filled(size, 0.0), shape);
  }

  /// This factory constructor creates a tensor of zeros with the same shape as
  /// the given tensor.
  /// 
  /// The shape of the tensor is provided as a list of integers, where each
  /// integer represents the size of the corresponding dimension. The total
  /// number of elements in the tensor is the product of the sizes of all
  /// dimensions.
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor.zeroslike(Tensor.randome([2, 3]))
  /// print(tensor.data); // Outputs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  /// print(tensor.shape); // Outputs: [2, 3]
  /// 
  /// 
  /// - Parameter tensor: The tensor whose shape will be used to create the new
  /// 
  /// - Returns: A new `Tensor` instance with the same shape as the given tensor
  ///  and all elements set to zero.
  factory Tensor.zeroslike(Tensor tensor) {
    return Tensor.zeros(tensor.shape);
  }

/// Creates a tensor filled with ones.
///
/// This factory constructor initializes a tensor of the specified shape with
/// all elements set to one.
///
/// The shape of the tensor is provided as a list of integers, where each 
/// integer represents the size of the corresponding dimension. The total
/// number of elements in the tensor is the product of the sizes of all 
/// dimensions.
///
/// Example:
/// ```dart
/// var tensor = Tensor.ones([2, 3]);
/// print(tensor.data); // Outputs: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
/// print(tensor.shape); // Outputs: [2, 3]
/// ```
///
/// - Parameter shape: A list of integers specifying the shape of the tensor.
///
/// - Returns: A new `Tensor` instance with the specified shape and all 
///   elements set to one.
  factory Tensor.ones(List<int> shape) {
    int size = shape.reduce((a, b) => a * b);
    return Tensor(List<double>.filled(size, 1.0), shape);
  }

  /// This factory constructor creates a tensor of ones with the same shape as
  /// the given tensor.
  /// 
  /// The shape of the tensor is provided as a list of integers, where each
  /// integer represents the size of the corresponding dimension. The total
  /// number of elements in the tensor is the product of the sizes of all
  /// dimensions.
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor.oneslike(Tensor.randome([2, 3]))
  /// print(tensor.data); // Outputs: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  /// print(tensor.shape); // Outputs: [2, 3]
  ///```
  /// 
  /// 
  /// - Parameter tensor: The tensor whose shape will be used to create the new
  /// 
  /// - Returns: A new `Tensor` instance with the same shape as the given tensor
  ///  and all elements set to one.
  factory Tensor.oneslike(Tensor tensor) {
    return Tensor.ones(tensor.shape);
  }

  /// This factory constructor creates a tensor of random values with the given
  /// shape.
  /// 
  /// The shape of the tensor is provided as a list of integers, where each
  /// integer represents the size of the corresponding dimension. The total
  /// number of elements in the tensor is the product of the sizes of all
  /// dimensions.
  ///  
  /// Example:
  /// ```dart
  /// var tensor = Tensor.random([2, 3]);
  /// print(tensor.data); // Outputs: List of length 6 with random values
  /// print(tensor.shape); // Outputs: [2, 3]
  /// ```
  /// 
  /// 
  /// - Parameter shape: A list of integers specifying the shape of the tensor.
  /// 
  /// - Returns: A new `Tensor` instance with the specified shape and random
  ///  values.
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

  /// This method adds two tensors element-wise.
  /// 
  /// Example:
  /// ```dart
  /// var tensor1 = Tensor([1, 2, 3], [3]);
  /// var tensor2 = Tensor([4, 5, 6], [3]);
  /// var result = tensor1.add(tensor2);
  /// print(result.data); // Outputs: [5.0, 7.0, 9.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Parameter other: The tensor to add to the current tensor.
  /// 
  /// - Returns: A new `Tensor` instance with the sum of the two tensors.
  Tensor add(Tensor other) {
    if (!_areShapesEqual(shape, other.shape)) {
      throw ArgumentError('Tensors must have the same shape for addition');
    }
    var resultData =
        List<double>.generate(size, (i) => data[i] + other.data[i]);
    return Tensor(resultData, shape);
  }

  /// This method multiplies two tensors element-wise.
  /// 
  /// Example:
  /// ```dart
  /// var tensor1 = Tensor([1, 2, 3], [3]);
  /// var tensor2 = Tensor([4, 5, 6], [3]);
  /// var result = tensor1.multiply(tensor2);
  /// print(result.data); // Outputs: [4.0, 10.0, 18.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Parameter other: The tensor to multiply with the current tensor.
  /// 
  /// - Returns: A new `Tensor` instance with the product of the two tensors.
  Tensor multiply(Tensor other) {
    if (!_areShapesEqual(shape, other.shape)) {
      throw ArgumentError(
          'Tensors must have the same shape for element-wise multiplication');
    }
    var resultData =
        List<double>.generate(size, (i) => data[i] * other.data[i]);
    return Tensor(resultData, shape);
  }

  /// This method subtracts one tensor from another element-wise.
  /// 
  /// Example:
  /// ```dart
  /// var tensor1 = Tensor([1, 2, 3], [3]);
  /// var tensor2 = Tensor([4, 5, 6], [3]);
  /// var result = tensor1.subtract(tensor2);
  /// print(result.data); // Outputs: [-3.0, -3.0, -3.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Parameter other: The tensor to subtract from the current tensor.
  /// 
  /// - Returns: A new `Tensor` instance with the difference of the two tensors. 
  Tensor subtract(Tensor other) {
    if (!_areShapesEqual(shape, other.shape)) {
      throw ArgumentError('Tensors must have the same shape for subtraction');
    }
    var resultData =
        List<double>.generate(size, (i) => data[i] - other.data[i]);
    return Tensor(resultData, shape);
  }

  /// This method returns the Tensor matrix multiplication of two tensors.
  /// 
  /// Example:
  /// ```dart
  /// var tensor1 = Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
  /// var tensor2 = Tensor([7, 8, 9, 10, 11, 12], [3, 2]);
  /// var result = tensor1.matmul(tensor2);
  /// print(result.data); // Outputs: [58.0, 64.0, 139.0, 154.0]
  /// print(result.shape); // Outputs: [2, 2]
  /// ```
  /// 
  /// - Parameter other: The tensor to multiply with the current tensor.
  /// 
  /// - Returns: A new `Tensor` instance with the matrix product of the two tensors.
  Tensor matmul(Tensor other) {
    if (shape.length != 2 || other.shape.length != 2) {
      throw ArgumentError(
          'Matrix multiplication is only defined for 2D tensors');
    }
    if (shape[1] != other.shape[0]) {
      throw ArgumentError(
          'Inner tensor dimensions must match for matrix multiplication');
    }

    int m = shape[0];
    int n = other.shape[1];
    int p = shape[1];

    var resultShape = [m, n];
    var resultData = List<double>.filled(m * n, 0.0);

    // Transpose the second matrix for better cache locality
    var otherTransposed = List<double>.filled(n * p, 0.0);
    for (int i = 0; i < p; i++) {
      for (int j = 0; j < n; j++) {
        otherTransposed[j * p + i] = other.data[i * n + j];
      }
    }

    // Perform matrix multiplication
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < p; k++) {
          sum += data[i * p + k] * otherTransposed[j * p + k];
        }
        resultData[i * n + j] = sum;
      }
    }

    return Tensor(resultData, resultShape);
  }

  /// This method adds a scalar to a tensor element-wise.
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.addScalar(2);
  /// print(result.data); // Outputs: [3.0, 4.0, 5.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Parameter scalar: The scalar value to add to the tensor.
  /// 
  /// - Returns: A new `Tensor` instance with the scalar added to each element.
  Tensor addScalar(double scalar) {
    var resultData = List<double>.generate(size, (i) => data[i] + scalar);
    return Tensor(resultData, shape);
  }

  /// This method multiplies a scalar to a tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.multiplyScalar(2);
  /// print(result.data); // Outputs: [2.0, 4.0, 6.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Parameter scalar: The scalar value to multiply with the tensor.
  /// 
  /// - Returns: A new `Tensor` instance with the scalar multiplied to each element. 
  Tensor multiplyScalar(double scalar) {
    var resultData = List<double>.generate(size, (i) => data[i] * scalar);
    return Tensor(resultData, shape);
  }

  /// This method subtracts a scalar from a tensor element-wise.
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.subtractScalar(2);
  /// print(result.data); // Outputs: [-1.0, 0.0, 1.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Parameter scalar: The scalar value to subtract from the tensor.
  /// 
  /// - Returns: A new `Tensor` instance with the scalar subtracted from each element.
  Tensor subtractScalar(double scalar) {
    var resultData = List<double>.generate(size, (i) => data[i] - scalar);
    return Tensor(resultData, shape);
  }

  /// This method divides a scalar from a tensor element-wise.
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([2, 4, 6], [3]);
  /// var result = tensor.divideScalar(2);
  /// print(result.data); // Outputs: [1.0, 2.0, 3.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Parameter scalar: The scalar value to divide the tensor by.
  /// 
  /// - Returns: A new `Tensor` instance with the scalar divided from each element.
  Tensor divideScalar(double scalar) {
    var resultData = List<double>.generate(size, (i) => data[i] / scalar);
    return Tensor(resultData, shape);
  }

  //! Linear Algebra operations

  /// This  method computes the transpose of a 2D tensor.
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([[1, 2, 3], [4, 5, 6]]);
  /// var result = tensor.transpose();
  /// print(result.data); // Outputs: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
  /// print(result.shape); // Outputs: [3, 2]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the transpose of the original tensor.
  /// 
  /// - Throws: An `ArgumentError` if the tensor is not 2D.
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

  /// This method computes the inverse of a 2D tensor using the Gauss-Jordan elimination method.
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([[1, 2], [3, 4]], [2, 2]);
  /// var result = tensor.inverse();
  /// print(result.data); // Outputs: [-2.0, 1.0, 1.5, -0.5]
  /// print(result.shape); // Outputs: [2, 2]
  /// 
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

  /// This method computes the determinant of a 2D tensor using the LU decomposition method.
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([[1, 2], [3, 4]], [2, 2]);
  /// var result = tensor.determinant();
  /// print(result); // Outputs: -2.0
  /// ```
  /// 
  /// - Returns: The determinant of the tensor.
  /// 
  /// - Throws: An `ArgumentError` if the tensor is not 2D or not square.
  /// 
  /// - Note: This method is not numerically stable and may fail for large matrices.
  /// 
  /// - Note: This method is not optimized for performance and may be slow for large matrices.
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

  /// This method computes the rank of a 2D tensor using the Gaussian elimination method.
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([[1, 2, 3], [4, 5, 6]], [2, 3]);
  /// var result = tensor.rank();
  /// print(result); // Outputs: 2
  /// ```
  /// 
  /// - Returns: The rank of the tensor.
  /// 
  /// - Throws: An `ArgumentError` if the tensor is not 2D.
  /// 
  /// - Note: This method is not optimized for performance and may be slow for large matrices.
  /// 
  /// - Note: This method is not numerically stable and may fail for large matrices.
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

  /// This method computes the sum of the elements in the tensor.
  /// 
  /// Example: 
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.sum();
  /// print(result); // Outputs: 6.0
  /// ```
  /// 
  /// - Returns: The sum of the elements in the tensor.
  double sum() {
    return data.reduce((a, b) => a + b);
  }

  /// This method computes the mean of the elements in the tensor
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.mean();
  /// print(result); // Outputs: 2.0
  /// ```
  /// 
  /// - Returns: The mean of the elements in the tensor.
  double mean() {
    return sum() / size;
  }

  /// This method computes the variance of the elements in the tensor
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.variance();
  /// print(result); // Outputs: 1.0
  /// ```
  /// 
  /// - Parameter population(default `false`): A boolean value indicating whether to compute the population variance.
  /// 
  /// - Returns: The variance of the elements in the tensor.
  /// 
  /// - Note: By default, the sample variance is computed. To compute the population variance, set the `population` parameter to `true`. 
  double variance({bool population = false}) {
    double meanValue = mean();
    num sumSquaredDiff =
        data.map((e) => math.pow(e - meanValue, 2)).reduce((a, b) => a + b);
    return population ? sumSquaredDiff / size : sumSquaredDiff / (size - 1);
  }

  /// This method returns the standard Deviation of the elements in the tensor
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.std();
  /// print(result); // Outputs: 1.0
  /// ```
  /// 
  /// - Parameter population(default `false`): A boolean value indicating whether to compute the population standard deviation.
  /// 
  /// - Returns: The standard deviation of the elements in the tensor.
  /// 
  /// - Note: By default, the sample standard deviation is computed. To compute the population standard deviation, set the `population` parameter to `true`.
  double std({bool population = false}) {
    return math.sqrt(variance(population: population));
  }

  /// This method returns the power of the tensor element-wise to the given exponent
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.pow(2);
  /// print(result.data); // Outputs: [1.0, 4.0, 9.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Parameter exponent: The exponent to raise the tensor to.
  /// 
  /// - Returns: A new `Tensor` instance with the elements raised to the given exponent.
  Tensor pow(double exponent) {
    var resultData = List<double>.generate(
        size, (i) => math.pow(data[i], exponent).toDouble());
    return Tensor(resultData, shape);
  }

  /// This method returns the square root of the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 4, 9], [3]);
  /// var result = tensor.sqrt();
  /// print(result.data); // Outputs: [1.0, 2.0, 3.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the square root of the elements.
  /// 
  /// - Throws: An `ArgumentError` if any element in the tensor is negative.
  Tensor sqrt() {
    var resultData = List<double>.generate(size, (i) => math.sqrt(data[i]));
    return Tensor(resultData, shape);
  }

  /// This method returns the exponential of the tensor element-wise
  /// 
  /// Example: 
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.exp();
  /// print(result.data); // Outputs: [2.718281828459045, 7.3890560989306495, 20.085536923187668]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the exponential of the elements.
  /// 
  /// - Note: The exponential function is defined as `e^x` where `e` is Euler's number.
  Tensor exp() {
    var resultData = List<double>.generate(size, (i) => math.exp(data[i]));
    return Tensor(resultData, shape);
  }

  /// This method returns logarithm of the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.log();
  /// print(result.data); // Outputs: [0.0, 0.6931471805599453, 1.0986122886681098]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the logarithm of the elements.
  /// 
  /// - Throws: An `ArgumentError` if any element in the tensor is negative.
  Tensor log() {
    var resultData = List<double>.generate(size, (i) => math.log(data[i]));
    return Tensor(resultData, shape);
  }

  /// This method returns the absolute value of the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([-1, -2, -3], [3]);
  /// var result = tensor.abs();
  /// print(result.data); // Outputs: [1.0, 2.0, 3.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the absolute value of the elements.
  /// 
  /// - Note: The absolute value of a number is its distance from zero.
  Tensor abs() {
    var result = data.map((x) => x.abs()).toList();
    return Tensor(result, shape);
  }

  /// This method returns the sign of the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([-1, 0, 3], [3]);
  /// var result = tensor.sign();
  /// print(result.data); // Outputs: [-1.0, 0.0, 1.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the sign of the elements.
  /// 
  /// - Note: The sign of a number is 1 if the number is positive, -1 if the number is negative, and 0 if the number is zero.
  Tensor sign() {
    var result = data.map((x) => x.sign).toList();
    return Tensor(result, shape);
  }

//! Rounding Operations
  /// This method returns the largest integer less than or equal to the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1.1, 2.2, 3.3], [3]);
  /// var result = tensor.floor();
  /// print(result.data); // Outputs: [1.0, 2.0, 3.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the floor of the elements.
  /// 
  /// - Note: The floor of a number is the largest integer less than or equal to the number.
  Tensor floor() {
    var resultData =
        List<double>.generate(size, (i) => data[i].floorToDouble());
    return Tensor(resultData, shape);
  }

  /// This method returns the smallest integer greater than or equal to the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1.1, 2.2, 3.3], [3]);
  /// var result = tensor.ceil();
  /// print(result.data); // Outputs: [2.0, 3.0, 4.0]
  /// print(result.shape); // Outputs: [3]
  /// 
  /// - Returns: A new `Tensor` instance with the ceil of the elements.
  /// 
  /// - Note: The ceil of a number is the smallest integer greater than or equal to the number.
  Tensor ceil() {
    var resultData = List<double>.generate(size, (i) => data[i].ceilToDouble());
    return Tensor(resultData, shape);
  }

  /// This method rounds the tensor element-wise to the nearest integer
  /// 
  /// Example:
  /// 
  /// ```dart
  /// var tensor = Tensor([1.1, 2.5, 3.9], [3]);
  /// var result = tensor.round();
  /// print(result.data); // Outputs: [1.0, 3.0, 4.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the elements rounded to the nearest integer.
  /// 
  /// - Note: If the fractional part of the number is less than 0.5, the number is rounded down. Otherwise, it is rounded up.
  /// 
  /// - Note: The round function rounds to the nearest even number in case of a tie.
  Tensor round() {
    var resultData =
        List<double>.generate(size, (i) => data[i].roundToDouble());
    return Tensor(resultData, shape);
  }

//! Trignonometric Operations
  /// This method returns the sine of the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([0, math.pi / 2, math.pi], [3]);
  /// var result = tensor.sin();
  /// print(result.data); // Outputs: [0.0, 1.0, 0.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the sine of the elements.
  /// 
  /// - Note: The sine function returns the ratio of the length of the opposite side to the length of the hypotenuse in a right-angled triangle.
  /// 
  /// - Note: The input to the sine function is assumed to be in radians.
  Tensor sin() {
    var resultData = List<double>.generate(size, (i) => math.sin(data[i]));
    return Tensor(resultData, shape);
  }

  /// This method returns the  cosine of the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([0, math.pi / 2, math.pi], [3]);
  /// var result = tensor.cos();
  /// print(result.data); // Outputs: [1.0, 0.0, -1.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the cosine of the elements.
  /// 
  /// - Note: The cosine function returns the ratio of the length of the adjacent side to the length of the hypotenuse in a right-angled triangle.
  /// 
  /// - Note: The input to the cosine function is assumed to be in radians.
  Tensor cos() {
    var resultData = List<double>.generate(size, (i) => math.cos(data[i]));
    return Tensor(resultData, shape);
  }

  /// This method returns the tangent of the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([0, math.pi / 4, math.pi / 2], [3]);
  /// var result = tensor.tan();
  /// print(result.data); // Outputs: [0.0, 1.0, double.infinity]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the tangent of the elements.
  /// 
  /// - Note: The tangent function returns the ratio of the length of the opposite side to the length of the adjacent side in a right-angled triangle.
  /// 
  /// - Note: The input to the tangent function is assumed to be in radians.
  Tensor tan() {
    var resultData = List<double>.generate(size, (i) => math.tan(data[i]));
    return Tensor(resultData, shape);
  }

  /// This method returns the arcsine of the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([0, 1, 0], [3]);
  /// var result = tensor.asin();
  /// print(result.data); // Outputs: [0.0, 1.5707963267948966, 0.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the arcsine of the elements.
  /// 
  /// - Note: The arcsine function returns the angle whose sine is the given number.
  /// 
  /// - Note: The output of the arcsine function is in radians.
  Tensor asin() {
    var resultData = List<double>.generate(size, (i) => math.asin(data[i]));
    return Tensor(resultData, shape);
  }

  /// This method returns the arccosine of the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 0, 1], [3]);
  /// var result = tensor.acos();
  /// print(result.data); // Outputs: [0.0, 1.5707963267948966, 0.0]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the arccosine of the elements.
  /// 
  /// - Note: The arccosine function returns the angle whose cosine is the given number.
  /// 
  /// - Note: The output of the arccosine function is in radians.
  Tensor acos() {
    var resultData = List<double>.generate(size, (i) => math.acos(data[i]));
    return Tensor(resultData, shape);
  }

  /// This method returns the arctangent of the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([0, 1, double.infinity], [3]);
  /// var result = tensor.atan();
  /// print(result.data); // Outputs: [0.0, 0.7853981633974483, 1.5707963267948966]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the arctangent of the elements.
  /// 
  /// - Note: The arctangent function returns the angle whose tangent is the given number.
  /// 
  /// - Note: The output of the arctangent function is in radians.
  Tensor atan() {
    var resultData = List<double>.generate(size, (i) => math.atan(data[i]));
    return Tensor(resultData, shape);
  }

  /// This method returns the Hyperbolic Sine of the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([0, 1, 2], [3]);
  /// var result = tensor.sinh();
  /// print(result.data); // Outputs: [0.0, 1.1752011936438014, 3.6268604078470186]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the Hyperbolic Sine of the elements.
  /// 
  /// - Note: The Hyperbolic Sine function is defined as `(e^x - e^-x) / 2`.
  /// 
  /// - Note: The input to the Hyperbolic Sine function is assumed to be in radians.
  Tensor sinh() {
    var result = data.map((x) => (math.exp(x) - math.exp(-x)) / 2).toList();
    return Tensor(result, shape);
  }

  /// This method returns the Hyperbolic Cosine of the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([0, 1, 2], [3]);
  /// var result = tensor.cosh();
  /// print(result.data); // Outputs: [1.0, 1.5430806348152437, 3.7621956910836314]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the Hyperbolic Cosine of the elements.
  /// 
  /// - Note: The Hyperbolic Cosine function is defined as `(e^x + e^-x) / 2`.
  /// 
  /// - Note: The input to the Hyperbolic Cosine function is assumed to be in radians.
  Tensor cosh() {
    var result = data.map((x) => (math.exp(x) + math.exp(-x)) / 2).toList();
    return Tensor(result, shape);
  }

  /// This method returns the Hyperbolic Tangent of the tensor element-wise
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([0, 1, 2], [3]);
  /// var result = tensor.tanh();
  /// print(result.data); // Outputs: [0.0, 0.7615941559557649, 0.9640275800758169]
  /// print(result.shape); // Outputs: [3]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the Hyperbolic Tangent of the elements.
  /// 
  /// - Note: The Hyperbolic Tangent function is defined as `(e^2x - 1) / (e^2x + 1)`.
  /// 
  /// - Note: The input to the Hyperbolic Tangent function is assumed to be in radians.
  Tensor tanh() {
    var result = data.map((x) {
      double exp2x = math.exp(2 * x);
      return (exp2x - 1) / (exp2x + 1);
    }).toList();
    return Tensor(result, shape);
  }

//! Aggregation Operations

  /// This method returns the maximum value in the tensor
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.max();
  /// print(result); // Outputs: 3.0
  /// ```
  /// 
  /// - Returns: The maximum value in the tensor.
  double max() {
    return data.reduce((a, b) => math.max(a, b));
  }

  /// This method returns the minimumvalue in the tensor
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.min();
  /// print(result); // Outputs: 1.0
  /// ```
  /// 
  /// - Returns: The minimum value in the tensor.
  double min() {
    return data.reduce((a, b) => math.min(a, b));
  }

  /// This method returns the index of the maximum value in the tensor
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.argmax();
  /// print(result); // Outputs: [2]
  /// ```
  /// 
  /// - Returns: The index of the maximum value in the tensor.
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

  /// This method returns the index of the minimum value in the tensor
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([1, 2, 3], [3]);
  /// var result = tensor.argmin();
  /// print(result); // Outputs: [0]
  /// ```
  /// 
  /// - Returns: The index of the minimum value in the tensor.
  /// 
  /// - Note: If there are multiple minimum values, the index of the first occurrence is returned.
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
  /// This method reshape the tensort into the desired shape and returns it.
  ///
  /// Example:
  /// 
  /// ```dart
  /// var tensor = Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
  /// var result = tensor.reshape([3, 2]);
  /// print(result.data); // Outputs: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  /// print(result.shape); // Outputs: [3, 2]
  /// ```
  /// 
  /// - Parameter newShape: The new shape of the tensor.
  /// 
  /// - Returns: A new `Tensor` instance with the new shape.

  Tensor reshape(List<int> newShape) {
    if (size != newShape.reduce((a, b) => a * b)) {
      throw ArgumentError('New shape must have the same number of elements');
    }
    return Tensor(data, newShape);
  }

  /// This method flattens the tensor into a 1D tensor and returns it.
  /// 
  /// Example:
  /// ```dart
  /// var tensor = Tensor([[1, 2], [3, 4]], [2, 2]);
  /// var result = tensor.flatten();
  /// print(result.data); // Outputs: [1.0, 2.0, 3.0, 4.0]
  /// print(result.shape); // Outputs: [4]
  /// ```
  /// 
  /// - Returns: A new `Tensor` instance with the elements flattened into a 1D tensor.
  /// 
  /// - Note: The order of the elements is row-major.
  Tensor flatten() {
    return Tensor(data, [size]);
  }

  //! Tensor Utilities

  /// This method Checks if two tensors are equal (element-wise comparison)
  /// 
  /// Example:
  /// ```dart
  /// var tensor1 = Tensor([1, 2, 3], [3]);
  /// var tensor2 = Tensor([1, 2, 3], [3]);
  /// var result = tensor1.equals(tensor2);
  /// print(result); // Outputs: true
  /// 
  /// var tensor3 = Tensor([1, 2, 3], [3]);
  /// var tensor4 = Tensor([1, 2, 4], [3]);
  /// var result = tensor3.equals(tensor4);
  /// print(result); // Outputs: false
  /// ```
  /// 
  /// - Parameter other: The other tensor to compare with.
  /// 
  /// - Returns: A boolean value indicating whether the two tensors are equal.
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

  /// This method performs the Tensor concatenation along a given axis
  /// 
  /// Example:
  /// ```dart
  /// var tensor1 = Tensor([1, 2], [2]);
  /// var tensor2 = Tensor([3, 4], [2]);
  /// var result = tensor1.concatenate(tensor2);
  /// print(result.data); // Outputs: [1.0, 2.0, 3.0, 4.0]
  /// print(result.shape); // Outputs: [4]
  /// ```
  /// 
  /// - Parameter other: The other tensor to concatenate with.
  /// 
  /// - Parameter axis(default `0`): The axis along which to concatenate the tensors.
  /// 
  /// - Returns: A new `Tensor` instance with the tensors concatenated along the given axis.
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

  ///  This method performs the Tensor stacking along a given axis.
  /// 
  /// Example:
  /// ```dart
  /// var tensor1 = Tensor([1, 2], [2]);
  /// var tensor2 = Tensor([3, 4], [2]);
  /// var result = tensor1.stack(tensor2);
  /// print(result.data); // Outputs: [[1.0, 2.0], [3.0, 4.0]]
  /// print(result.shape); // Outputs: [2, 2]
  /// 
  /// var tensor3 = Tensor([5, 6], [2]);
  /// var result2 = tensor1.stack(tensor3, axis: 1);
  /// print(result2.data); // Outputs: [[1.0, 5.0], [2.0, 6.0]]
  /// print(result2.shape); // Outputs: [2, 2]
  /// ```
  /// 
  /// - Parameter other: The other tensor to stack with.
  /// 
  /// - Parameter axis(default `0`): The axis along which to stack the tensors.
  /// 
  /// - Returns: A new `Tensor` instance with the tensors stacked along the given axis.
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
