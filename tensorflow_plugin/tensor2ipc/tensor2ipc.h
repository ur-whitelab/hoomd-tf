#ifndef TENSOR2IPC
#define TENSOR2IPC

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "../TensorflowCompute.h"

using namespace tensorflow;

REGISTER_OP("TensorToIpc")
    .Attr("T: {float}")
    .Attr("size: int")
    .Attr("address: int")
    .Input("input: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int32 size;
      tensorflow::shape_inference::DimensionHandle v1; //unused
      tensorflow::shape_inference::ShapeHandle v2; //unused
      c->GetAttr("size", &size);
      //check shape of input is size x 4
      //just looking for errors, not using result
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), 0), size, &v1));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), 1), 4, &v1));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &v2));
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