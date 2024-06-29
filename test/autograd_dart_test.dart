import 'package:autograd_dart/src/core/tensor.dart';
import 'dart:math' as math;
import 'package:test/test.dart';

void main() {
  group('Tensor', () {
    test('creates tensor with given shape and data', () {
      var data = [1.0, 2.0, 3.0, 4.0];
      var shape = [2, 2];
      var tensor = Tensor(data, shape);
      expect(tensor.data, equals(data));
      expect(tensor.shape, equals(shape));
    });

    test('computes strides correctly', () {
      var data = [1.0, 2.0, 3.0, 4.0];
      var shape = [2, 2];
      var tensor = Tensor(data, shape);
      expect(tensor.strides, equals([2, 1]));
    });

    test('creates tensor of zeros with given shape', () {
      var shape = [2, 2];
      var tensor = Tensor.zeros(shape);
      expect(tensor.data, equals([0.0, 0.0, 0.0, 0.0]));
      expect(tensor.shape, equals(shape));
    });

    test('creates tensor of ones with given shape', () {
      var shape = [2, 2];
      var tensor = Tensor.ones(shape);
      expect(tensor.data, equals([1.0, 1.0, 1.0, 1.0]));
      expect(tensor.shape, equals(shape));
    });

    test('creates random tensor with given shape', () {
      var shape = [2, 2];
      var tensor = Tensor.random(shape);
      expect(tensor.data.length, equals(4));
      expect(tensor.shape, equals(shape));
    });

    test('adds two tensors element-wise', () {
      var tensor1 = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);
      var tensor2 = Tensor([5.0, 6.0, 7.0, 8.0], [2, 2]);
      var result = tensor1.add(tensor2);
      expect(result.data, equals([6.0, 8.0, 10.0, 12.0]));
    });

    test('multiplies two tensors element-wise', () {
      var tensor1 = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);
      var tensor2 = Tensor([5.0, 6.0, 7.0, 8.0], [2, 2]);
      var result = tensor1.multiply(tensor2);
      expect(result.data, equals([5.0, 12.0, 21.0, 32.0]));
    });

    test('subtracts two tensors element-wise', () {
      var tensor1 = Tensor([5.0, 6.0, 7.0, 8.0], [2, 2]);
      var tensor2 = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);
      var result = tensor1.subtract(tensor2);
      expect(result.data, equals([4.0, 4.0, 4.0, 4.0]));
    });

    test('transposes a 2D tensor', () {
      var tensor = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);
      var result = tensor.transpose();
      expect(result.data, equals([1.0, 3.0, 2.0, 4.0]));
      expect(result.shape, equals([2, 2]));
    });

    test('calculates the determinant of a 2x2 tensor', () {
      var tensor = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);
      var det = tensor.determinant();
      expect(det, equals(-2.0));
    });

    test('computes the sum of elements in the tensor', () {
      var tensor = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);
      var sum = tensor.sum();
      expect(sum, equals(10.0));
    });

    test('computes the mean of elements in the tensor', () {
      var tensor = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);
      var mean = tensor.mean();
      expect(mean, equals(2.5));
    });

    test('computes the variance of elements in the tensor', () {
      var tensor = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);
      var variance = tensor.variance();
      expect(variance, closeTo(1.6667, 0.0001));
    });

    test('computes the standard deviation of elements in the tensor', () {
      var tensor = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);
      var std = tensor.std();
      expect(std, closeTo(1.29099, 0.0001));
    });

    test('flattens a tensor', () {
      var tensor = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);
      var flattened = tensor.flatten();
      expect(flattened.data, equals([1.0, 2.0, 3.0, 4.0]));
      expect(flattened.shape, equals([4]));
    });

    test('reshapes a tensor', () {
      var tensor = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);
      var reshaped = tensor.reshape([4]);
      expect(reshaped.data, equals([1.0, 2.0, 3.0, 4.0]));
      expect(reshaped.shape, equals([4]));
    });

    test('concatenates two tensors along a given axis', () {
      var tensor1 = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);
      var tensor2 = Tensor([5.0, 6.0, 7.0, 8.0], [2, 2]);
      var concatenated = tensor1.concatenate(tensor2, axis: 0);
      expect(
          concatenated.data, equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
      expect(concatenated.shape, equals([4, 2]));
    });

    test('stacks two tensors along a given axis', () {
      var tensor1 = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);
      var tensor2 = Tensor([5.0, 6.0, 7.0, 8.0], [2, 2]);
      var stacked = tensor1.stack(tensor2, axis: 0);
      expect(stacked.data, equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
      expect(stacked.shape, equals([2, 2, 2]));
    });
    test('Sine', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.sin();

      for (int i = 0; i < tensor.size; i++) {
        expect(result.data[i], equals(math.sin(tensor.data[i])));
      }
    });

    test('Cosine', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.cos();

      for (int i = 0; i < tensor.size; i++) {
        expect(result.data[i], equals(math.cos(tensor.data[i])));
      }
    });

    test('Tangent', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.tan();

      for (int i = 0; i < tensor.size; i++) {
        expect(result.data[i], equals(math.tan(tensor.data[i])));
      }
    });

    test('Arcsine', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.asin();

      for (int i = 0; i < tensor.size; i++) {
        expect(result.data[i], equals(math.asin(tensor.data[i])));
      }
    });

    test('Arccosine', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.acos();

      for (int i = 0; i < tensor.size; i++) {
        expect(result.data[i], equals(math.acos(tensor.data[i])));
      }
    });

    test('Arctangent', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.atan();

      for (int i = 0; i < tensor.size; i++) {
        expect(result.data[i], equals(math.atan(tensor.data[i])));
      }
    });

    test('Hyperbolic Sine', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.sinh();

      for (int i = 0; i < tensor.size; i++) {
        expect(result.data[i], equals((math.exp(tensor.data[i]) - math.exp(-tensor.data[i])) / 2));
      }
    });

    test('Hyperbolic Cosine', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.cosh();

      for (int i = 0; i < tensor.size; i++) {
        expect(result.data[i], equals((math.exp(tensor.data[i]) + math.exp(-tensor.data[i])) / 2));
      }
    });

    test('Hyperbolic Tangent', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.tanh();

      for (int i = 0; i < tensor.size; i++) {
        expect(result.data[i], equals((math.exp(2 * tensor.data[i]) - 1) / (math.exp(2 * tensor.data[i]) + 1)));
      }
    });


  });

  group('Tensor Statistical Operations', () {
    test('Sum', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.sum();
      var expected = tensor.data.reduce((a, b) => a + b);
      expect(result, equals(expected));
    });

    test('Mean', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.mean();
      var expected = tensor.data.reduce((a, b) => a + b) / tensor.size;
      expect(result, equals(expected));
    });

    test('Variance (Sample)', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.variance(population: false);
      var mean = tensor.mean();
      var expected = tensor.data.map((e) => math.pow(e - mean, 2)).reduce((a, b) => a + b) / (tensor.size - 1);
      expect(result, equals(expected));
    });

    test('Variance (Population)', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.variance(population: true);
      var mean = tensor.mean();
      var expected = tensor.data.map((e) => math.pow(e - mean, 2)).reduce((a, b) => a + b) / tensor.size;
      expect(result, equals(expected));
    });

    test('Standard Deviation (Sample)', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.std(population: false);
      var expected = math.sqrt(tensor.variance(population: false));
      expect(result, equals(expected));
    });

    test('Standard Deviation (Population)', () {
      var tensor = Tensor.random([3, 3]);
      var result = tensor.std(population: true);
      var expected = math.sqrt(tensor.variance(population: true));
      expect(result, equals(expected));
    });
  });
}
