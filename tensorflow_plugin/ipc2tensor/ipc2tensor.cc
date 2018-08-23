#include "ipc2tensor.h"
#include <string.h>
#include <sys/mman.h>
#include <typeinfo>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

REGISTER_OP("IpcToTensor")
    .Attr("T: {float, double}")
    .Attr("Tshape: {int32, int64}")
    .Input("shape: Tshape")
    .Attr("address: int")  // memory address. Should be scalar. TODO: learn to
                           // check rank. Not sure about type to use here!
    .Output("output: T")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Taken from common_shape_functions and following
      // example for random_ops.cc in TF source
      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      c->set_output(0, out);
      return Status::OK();
    });

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct IPC2TFunctor<CPUDevice, T, T> {
  void operator()(const CPUDevice& d, int size, void* address, T& ipc_memory,
                  T* out) {
    T* input_buffer = reinterpret_cast<T*>(address);
    std::memcpy(out, input_buffer, size * sizeof(T));
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T, typename Tshape, typename IPCM>
class IpcToTensorOp : public OpKernel {
 public:
  explicit IpcToTensorOp(OpKernelConstruction* context) : OpKernel(context) {
    // get memory address
    int64 tmp;
    context->GetAttr("address", &tmp);
    _input_memory = reinterpret_cast<void*>(tmp);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& shape = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    TensorShape output_shape;

    OP_REQUIRES(context, TensorShapeUtils::IsVector(shape.shape()),
                errors::InvalidArgument(
                    "Shape specification to IpcToTensor should be vector"));

    // TODO: why is this necessary?!
    TensorShapeUtils::MakeShape(shape.vec<Tshape>(), &output_shape);

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, output_tensor->NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    auto output = output_tensor->flat<T>();
    IPC2TFunctor<Device, T, IPCM>()(context->eigen_device<Device>(),
                                    output.size(), _input_memory, _ipc_memory,
                                    output.data());
  }

 private:
  void* _input_memory;
  IPCM _ipc_memory;
};

// Register the CPU kernels.
#define REGISTER_CPU(T, Tshape)                                 \
  REGISTER_KERNEL_BUILDER(Name("IpcToTensor")                   \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<Tshape>("Tshape") \
                              .TypeConstraint<T>("T"),          \
                          IpcToTensorOp<CPUDevice, T, Tshape, T>);
REGISTER_CPU(float, int32);
REGISTER_CPU(float, int64);
REGISTER_CPU(double, int32);
REGISTER_CPU(double, int64);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T, Tshape)                                 \
  REGISTER_KERNEL_BUILDER(Name("IpcToTensor")                   \
                              .Device(DEVICE_GPU)               \
                              .HostMemory("shape")              \
                              .TypeConstraint<Tshape>("Tshape") \
                              .TypeConstraint<T>("T"),          \
                          IpcToTensorOp<GPUDevice, T, Tshape, cudaIPC_t<T> >);
REGISTER_GPU(float, int32);
REGISTER_GPU(float, int64);
REGISTER_GPU(double, int32);
REGISTER_GPU(double, int64);
#endif  // GOOGLE_CUDA
