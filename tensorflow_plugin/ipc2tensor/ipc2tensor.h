#ifndef IPC2TENSOR
#define IPC2TENSOR

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "../TensorflowUpdater.h"

using namespace tensorflow;

REGISTER_OP("IpcToTensor")
    .Attr("T: {float}")
    .Attr("shape: int")
    .Attr("address: int") //memory address. Should be scalar. TODO: learn to check rank. Not sure about type to use here!
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int32 shape;
      c->GetAttr("shape", &shape);

      //this should make the size be the size of shape. Should be N x 4
      shape_inference::DimensionHandle particle_dimension = c->MakeDim(shape);
      shape_inference::DimensionHandle spatial_dimension = c->MakeDim(4);
      shape_inference::ShapeHandle out = c->MakeShape({particle_dimension, spatial_dimension});
      c->set_output(0, out);
      return Status::OK();
    });

template <typename Device, typename T>
struct IPC2TFunctor {
  void operator()(const Device& d, int size, int64 address, T* out);
};

template <typename Device>
struct IPC2TInitialize {
  bool operator()(int32 size, int64 address);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct IPC2TFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, int64 address, T* out);
};
#endif

#endif //IPC2TENSOR