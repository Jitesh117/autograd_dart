import 'package:autograd_dart/src/core/tensor.dart';

void main() {
  // Creating a tensor of zeros with shape [2, 3]
  Tensor first = Tensor.random([2, 2, 2]);
  print(first);
  Tensor second = Tensor.random([2, 2, 2]);
  print(second);
  print(first.concatenate(second));
}
