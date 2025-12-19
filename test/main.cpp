#include <ScalarType/ScalarType.h>
#include <cpu/TensorOps.h>
#include <iostream>

int main(){

    Tensor<vectra::float32> ten = full({3, 3}, 2.0f);
    Tensor<vectra::float32> result = exp(ten);

    std::cout << result;

    return 0;
}