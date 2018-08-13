#ifndef TENSOR2IPC
#define TENSOR2IPC

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("TensorToIpc")
    .Input("input: T")
    .Attr("T: {float, double}")
    .Attr("maxsize: int")
    .Attr("address: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int32 size;
      c->GetAttr("maxsize", &size);
      //check shape of input is not larger than buffer
      auto result = c->Value(c->NumElements(c->input(0)));
      if(!result || size > result)
        return errors::InvalidArgument("Input tensor may exceed buffer size, which is set by maxsize");
      return Status::OK();
    });

template <typename Device, typename T>
struct TF2IPCFunctor {
  void operator()(const Device& d, int size, int64 address, const T* in);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct TF2IPCFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, int64 address, T* in);
};
#endif

#endif //TENSOR2IPC