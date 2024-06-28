import 'package:autograd_dart/src/core/tensor.dart';

void main() {
  // Create a 2x3 tensor filled with zeros
  var t1 = Tensor.zeros([2, 3]);
  print('t1: $t1');

  // Create a 2x3 tensor filled with ones
  var t2 = Tensor.ones([2, 3]);
  print('t2: $t2');

  // Create a 2x3 tensor with random values
  var t3 = Tensor.random([2, 3]);
  print('t3: $t3');

  // Access and modify elements
  print('t3[0, 1] before: ${t3[[0, 1]]}');
  t3[[0, 1]] = 5.0;
  print('t3[0, 1] after: ${t3[[0, 1]]}');

  // Perform operations
  var t4 = t2.add(t3);
  print('t2 + t3: $t4');

  var t5 = t2.multiply(t3);
  print('t2 * t3: $t5');

  // Try to add tensors with incompatible shapes
}