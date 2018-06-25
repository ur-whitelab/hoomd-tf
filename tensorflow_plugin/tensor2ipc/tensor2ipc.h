#ifndef TENSOR2IPC
#define TENSOR2IPC

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "../TensorflowUpdater.h"

using namespace tensorflow;

REGISTER_OP("TensorToIpc")
    .Attr("T: {float}")
    .Attr("size: int")
    .Attr("address: int") 
    .Input("input: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return Status::OK();
    });

template <typename Device, typename T>
struct TF2IPCFunctor {
  void operator()(const Device& d, int size, int64 address, T* in);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct TF2IPCFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, int64 address, T* in);
};
#endif

#endif //TENSOR2IPC