#include <pybind11/pybind11.h>
#include <torch/extension.h>
int sampling(torch::Tensor& input_sols, torch::Tensor& prob, torch::Tensor& flip_idx);
int naive_local_search(torch::Tensor& input_sols, torch::Tensor& Q, 
                          torch::Tensor& lin, torch::Tensor& perm);
int batch_local_search(torch::Tensor& input_sols, torch::Tensor& Q, 
                            torch::Tensor& lin, torch::Tensor& perm, size_t batch_size);
PYBIND11_MODULE(mcpg_kernel, m) {
    m.doc() = "MCPG kernel functions";
    m.def("sampling", &sampling, "Perform sampling on CUDA",
          py::arg("input_sols"), py::arg("prob"), py::arg("flip_idx"));
    m.def("naive_local_search", &naive_local_search, "Perform naive local search on CUDA",
            py::arg("input_sols"), py::arg("Q"), py::arg("lin"), py::arg("perm"));
    m.def("batch_local_search", &batch_local_search, "Perform batch local search on CUDA",
            py::arg("input_sols"), py::arg("Q"), py::arg("lin"), py::arg("perm"), 
            py::arg("batch_size"), "Batch size for local search");
}