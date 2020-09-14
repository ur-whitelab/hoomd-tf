// Copyright (c) 2020 HOOMD-TF Developers

#include "hoomd2tf.h"
#include <string.h>
#include <sys/mman.h>
#include <typeinfo>
#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

REGISTER_OP("HoomdToTf")
    .Attr("T: {float, double}")
    .Attr("Tshape: {int32, int64}")
    .Input("shape: Tshape")
    .Attr("address: int") // memory address. Should be scalar. TODO: learn to
                          // check rank. Not sure about type to use here!
    .Output("output: T")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      // Taken from common_shape_functions and following
      // example for random_ops.cc in TF source
      shape_inference::ShapeHandle shape_input;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &shape_input));

      shape_inference::ShapeHandle shape_unknown = c->Vector(c->UnknownDim());
      shape_inference::ShapeHandle shape_output;
      TF_RETURN_IF_ERROR(c->Concatenate(shape_unknown, shape_input, &shape_output));
      c->set_output(0, shape_output);
      return Status::OK();
    });

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct HOOMD2TFFunctor<CPUDevice, T>
{
  void operator()(const CPUDevice &d, int size, CommStruct *in_memory,
                  T *out)
  {
    in_memory->readCPUMemory(out, size * sizeof(T));
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T, typename Tshape>
class HoomdToTfOp : public OpKernel
{
public:
  explicit HoomdToTfOp(OpKernelConstruction *context) : OpKernel(context)
  {
    // get memory address
    int64 tmp;
    context->GetAttr("address", &tmp);
    m_input_memory = reinterpret_cast<CommStruct *>(tmp);
  }

  void Compute(OpKernelContext *context) override
  {
    const Tensor &shape = context->input(0);

    // Create an output tensor
    Tensor *output_tensor = nullptr;
    TensorShape tmp_shape;

    OP_REQUIRES(context, TensorShapeUtils::IsVector(shape.shape()),
                errors::InvalidArgument(
                    "Shape specification to HoomdToTf should be vector"));

    // TODO: Is there a performance hit for this?
    TensorShapeUtils::MakeShape(m_input_memory->num_elements, m_input_memory->num_dims, &tmp_shape);

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, tmp_shape, &output_tensor));

    // Do the computation
    OP_REQUIRES(context, output_tensor->NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    auto output = output_tensor->flat<T>();
    HOOMD2TFFunctor<Device, T>()(context->eigen_device<Device>(),
                                 output.size(), m_input_memory,
                                 output.data());
  }

private:
  CommStruct *m_input_memory;
};

// Register the CPU kernels.
#define REGISTER_CPU(T, Tshape)                                 \
  REGISTER_KERNEL_BUILDER(Name("HoomdToTf")                     \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<Tshape>("Tshape") \
                              .TypeConstraint<T>("T"),          \
                          HoomdToTfOp<CPUDevice, T, Tshape>);
REGISTER_CPU(float, int32);
REGISTER_CPU(float, int64);
REGISTER_CPU(double, int32);
REGISTER_CPU(double, int64);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T, Tshape)                                 \
  REGISTER_KERNEL_BUILDER(Name("HoomdToTf")                     \
                              .Device(DEVICE_GPU)               \
                              .HostMemory("shape")              \
                              .TypeConstraint<Tshape>("Tshape") \
                              .TypeConstraint<T>("T"),          \
                          HoomdToTfOp<GPUDevice, T, Tshape>);
REGISTER_GPU(float, int32);
REGISTER_GPU(float, int64);
REGISTER_GPU(double, int32);
REGISTER_GPU(double, int64);
#endif // GOOGLE_CUDA
