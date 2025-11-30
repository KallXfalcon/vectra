#include <ScalarType/ScalarType.h>
#include <cpu/TensorOps.h>
#include <iostream>

int main(){
    Tensor<vectra::float32> aten = full({5, 5}, 1.0f);
    Tensor<vectra::float32> bten = flatten(aten);

    std::cout << aten;
    std::cout << bten;

    return 0;
}