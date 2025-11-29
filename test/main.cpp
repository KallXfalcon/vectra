#include <ScalarType/ScalarType.h>
#include <cpu/TensorOps.h>
#include <iostream>

int main(){
    Tensor<vectra::float32> aten = full({28, 28}, 1.0f);

    std::cout << aten;

    return 0;
}