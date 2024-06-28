import 'package:autograd_dart/src/core/tensor.dart';


void main() {
  // Creating a tensor of zeros with shape [2, 3]
  Tensor randomTensor = Tensor.random([2, 3]);
  Tensor randTensor = Tensor.random([3, 2]);
  print("Tensor one: $randomTensor");
  print("Tensor one: $randTensor");
  print(randomTensor.matmul(randTensor));
}
