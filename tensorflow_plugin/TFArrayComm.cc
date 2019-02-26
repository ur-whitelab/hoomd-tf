#include "TFArrayComm.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>
#include <hoomd/extern/pybind/include/pybind11/stl_bind.h>

namespace hoomd_tf{

#ifdef ENABLE_CUDA

void tf_check_cuda_error(cudaError_t err, const char* file,
                          unsigned int line) {
  // if there was an error
  if (err != cudaSuccess) {
    // print an error message
    std::cerr << "error type: " << cudaGetErrorName(err) << ": "
              << std::string(cudaGetErrorString(err)) << " before " << file
              << ":" << line << std::endl;

    // throw an error exception
    throw(std::runtime_error("CUDA Error"));
  }
#endif

  void* int2ptr(int64_t address) { return reinterpret_cast<void*>(address); }

void export_TFArrayComm(pybind11::module& m) {
  pybind11::class_<TFArrayComm<TFCommMode::CPU, double>,
                   std::shared_ptr<TFArrayComm<TFCommMode::CPU, double> > >(
      m, "TFArrayCommCPU")
      .def(pybind11::init())
      .def("getArray", &TFArrayComm<TFCommMode::CPU, double>::getArray,
           pybind11::return_value_policy::take_ownership)
      ;

    m.def("int2ptr", &int2ptr);
  }
}
