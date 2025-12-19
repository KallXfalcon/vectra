// binding.cpp
#include <sstream>
#include <vector>
#include <iterator>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cpu/TensorOps.h>
#include <ScalarType/ScalarType.h>

namespace py = pybind11;

template <typename T>
void bind_tensor(py::module_& m, const char* name)
{
    using TensorT = Tensor<T>;

    py::class_<TensorT>(m, name)
        .def(py::init<const std::vector<size_t>&>(), py::arg("shape"))

        .def_property_readonly("shape", [](const TensorT& t) {
            return t.shape;
        })

        .def("to_list", [](const TensorT& t) {
            std::vector<T> out;
            out.reserve(t.data.size());
            for (size_t i = 0; i < t.data.size(); ++i)
                out.push_back(t.data.data[i].get_scalar());
            return out;
        })

        .def("__add__", [](const TensorT& a, const TensorT& b) {
            return add<T>(a, b);
        })
        .def("__sub__", [](const TensorT& a, const TensorT& b) {
            return sub<T>(a, b);
        })
        .def("__mul__", [](const TensorT& a, const TensorT& b) {
            return mul<T>(a, b);
        })
        .def("__divide__", [](const TensorT& a, const TensorT& b) {
            return div<T>(a, b);
        })

        .def("__repr__", [](const TensorT& t) {
            std::ostringstream oss;
            oss << t;
            return oss.str();
        });
}


PYBIND11_MODULE(tensor_ops, m)
{
    m.doc() = "Tensor operations";

    bind_tensor<vectra::int16>(m, "TensorInt16");
    bind_tensor<vectra::int32>(m, "TensorInt32");
    bind_tensor<vectra::float32>(m, "TensorFloat32");
    bind_tensor<vectra::float64>(m, "TensorFloat64");

    m.def("vectra_full_int16",   &full<vectra::int16>);
    m.def("vectra_full_int32",   &full<vectra::int32>);
    m.def("vectra_full_float32", &full<vectra::float32>);
    m.def("vectra_full_float64", &full<vectra::float64>);

    m.def("vectra_zeros_int16",   &zeros<vectra::int16>);
    m.def("vectra_zeros_int32",   &zeros<vectra::int32>);
    m.def("vectra_zeros_float32", &zeros<vectra::float32>);
    m.def("vectra_zeros_float64", &zeros<vectra::float64>);

    m.def("vectra_ones_int16",   &ones<vectra::int16>);
    m.def("vectra_ones_int32",   &ones<vectra::int32>);
    m.def("vectra_ones_float32", &ones<vectra::float32>);
    m.def("vectra_ones_float64", &ones<vectra::float64>);

    m.def("vectra_rand_float32", &rand<vectra::float32>);
    m.def("vectra_rand_float64", &rand<vectra::float64>);

    m.def("vectra_randn_float32", &randn<vectra::float32>);
    m.def("vectra_randn_float64", &randn<vectra::float64>);

    m.def("vectra_add_int16",   &add<vectra::int16>);
    m.def("vectra_add_int32",   &add<vectra::int32>);
    m.def("vectra_add_float32", &add<vectra::float32>);
    m.def("vectra_add_float64", &add<vectra::float64>);

    m.def("vectra_sub_int16",   &sub<vectra::int16>);
    m.def("vectra_sub_int32",   &sub<vectra::int32>);
    m.def("vectra_sub_float32", &sub<vectra::float32>);
    m.def("vectra_sub_float64", &sub<vectra::float64>);

    m.def("vectra_mul_int16",   &mul<vectra::int16>);
    m.def("vectra_mul_int32",   &mul<vectra::int32>);
    m.def("vectra_mul_float32", &mul<vectra::float32>);
    m.def("vectra_mul_float64", &mul<vectra::float64>);

    m.def("vectra_div_float32", &div<vectra::float32>);
    m.def("vectra_div_float64", &div<vectra::float64>);

    m.def("vectra_dot_int16",   &dot<vectra::int16>);
    m.def("vectra_dot_int32",   &dot<vectra::int32>);
    m.def("vectra_dot_float32", &dot<vectra::float32>);
    m.def("vectra_dot_float64", &dot<vectra::float64>);
}
