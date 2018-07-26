#include "ipc2tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include <sys/mman.h>
#include <typeinfo>
#include <string.h>

using namespace tensorflow;


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct IPC2TFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, int64 address, T* out) {
    //TODO: access address
    T* input_buffer = reinterpret_cast<T*> (address);
    std::memcpy(out, input_buffer, size * sizeof(T));
  }
};


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T, typename Tshape>
class IpcToTensorOp : public OpKernel {
 public:
  explicit IpcToTensorOp(OpKernelConstruction* context) : OpKernel(context) {

    //get memory address
    context->GetAttr("address", &_input_address);

  }

  void Compute(OpKernelContext* context) override {


    const Tensor& shape = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape output_shape;


    OP_REQUIRES(context, TensorShapeUtils::IsVector(shape.shape()),
                errors::InvalidArgument("Shape specification to IpcToTensor should be vector"));

    //TODO: why is this necessary?!
    TensorShapeUtils::MakeShape(shape.vec<Tshape>(), &output_shape);

    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, output_tensor->NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    auto output = output_tensor->flat<T>();
    IPC2TFunctor<Device, T>()(
        context->eigen_device<Device>(),
        output.size(),
        _input_address,
        output.data());
  }

private:
  int64 _input_address;
};

// Register the CPU kernels.
#define REGISTER_CPU(T, Tshape)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("IpcToTensor") \
      .Device(DEVICE_CPU) \
      .TypeConstraint<Tshape>("Tshape") \
      .TypeConstraint<T>("T"), \
      IpcToTensorOp<CPUDevice, T, Tshape>);
REGISTER_CPU(float, int32);
REGISTER_CPU(float, int64);
REGISTER_CPU(double, int32);
REGISTER_CPU(double, int64);


// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_IPC2T.cu.cc. */ \
  extern template IPC2TFunctor<GPUDevice, float>;              \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("IpcToTensor")
      .Device(DEVICE_GPU).TypeConstraint<T>("T")
      .HostMemory("shape")
      .HostMemory("address"), \
      IpcToTensorOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
