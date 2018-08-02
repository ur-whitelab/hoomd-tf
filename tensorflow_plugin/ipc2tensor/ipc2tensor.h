#ifndef IPC2TENSOR
#define IPC2TENSOR

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "../TensorflowCompute.h"

using namespace tensorflow;

REGISTER_OP("IpcToTensor")
    .Attr("T: {float, double}")
    .Attr("Tshape: {int32, int64}")
    .Input("shape: Tshape")
    .Attr("address: int") //memory address. Should be scalar. TODO: learn to check rank. Not sure about type to use here!
    .Output("output: T")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      //Taken from common_shape_functions and following
      //example for random_ops.cc in TF source
      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      c->set_output(0, out);
      return Status::OK();
    });

template <typename Device, typename T>
struct IPC2TFunctor {
  void operator()(const Device& d, int size, int64 address, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct IPC2TFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, int64 address, T* out);
};
#endif

#endif //IPC2TENSOR