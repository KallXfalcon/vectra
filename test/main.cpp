#include <ScalarType/ScalarType.h>
#include <cpu/TensorOps.h>
#include <iostream>

int main(){

    Tensor<vectra::float32> aten = randn<vectra::float32>({3, 3});
    Tensor<vectra::float32> bten = randn<vectra::float32>({3, 3});
    Tensor<vectra::float32> result = add(aten, bten);

    std::cout << aten;
    std::cout << bten;
    std::cout << result;

    return 0;
}