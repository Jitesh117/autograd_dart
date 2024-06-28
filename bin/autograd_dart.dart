import 'package:autograd_dart/src/core/tensor.dart';

void main() {
  // Creating a tensor of zeros with shape [2, 3]
  Tensor ones = Tensor.random([4, 4]);
  print(ones);
  print(ones.sum());
  print(ones.mean());
  print(ones.variance());
  print(ones.argmax());
  print(ones.argmin());
}
