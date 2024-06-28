import 'package:autograd_dart/src/core/tensor.dart';

void main() {
  // Creating a tensor of zeros with shape [2, 3]
  Tensor ones = Tensor.random([4, 4]);
  print("Tensor: ");
  print(ones);
  print("Determinant: ");
  print(ones.determinant());
  print("Inverse: ");
  print(ones.inverse());
  print("Transpose: ");
  print(ones.transpose());
  print("Rank: ");
  print(ones.rank());
}
