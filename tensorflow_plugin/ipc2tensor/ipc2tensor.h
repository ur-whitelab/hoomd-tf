#ifndef KERNEL_HOOMD_H_
#define KERNEL_HOOMD_H_

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("IPC2Tensor")
    .Input("shape: float")
    .Input("address: float") //memory address. Should be scalar. TODO: learn to check rank
    .Output("output: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle out;
      //this should make the size be the size of shape. Should be N x 3
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      c->set_output(0, out);
      return Status::OK();
    });

template <typename Device, typename T>
struct HoomdFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct HoomdFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif //KERNEL_HOOMD_H_