import 'package:autograd_dart/src/core/tensor.dart';

void main() {
  // Creating a tensor of zeros with shape [2, 3]
  Tensor first = Tensor.random([2, 2, 2]);
  print(first);
  Tensor second = Tensor.random([2, 2, 2]);
  print(second);
  print("Concatenation:");
  Tensor concatenatedTensor = first.concatenate(second);
  print(concatenatedTensor);
  print("Shape after concatenating: ${concatenatedTensor.shape}");
  Tensor stackedTensor = first.stack(second, axis:1);
  print(stackedTensor);
  print("Shape after stacking: ${stackedTensor.shape}");
  print(first.stack(second, axis: 1).equals(first.concatenate(second)));
  print(first.concatenate(second).equals(first.concatenate(second)));
}
