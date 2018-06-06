#include "ipc2tensor.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct IPC2TFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, long address, T* out) {
    //TODO: access address
    for(int i = 0; i < size; ++i)
      T[i] = 0;
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class IPC2Tensor : public OpKernel {
 public:
  explicit IPC2Tensor(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Get input shape
    const Tensor& input_shape_tensor = context->input(0);
    int input_shape = input_shape_tensor.tensor<int, 1>()(0);

    //get memory address
    //TODO: I have no idea what I'm doing
    const Tensor& input_memory_tensor = context->input(1);
    long input_address = input_memory_tensor.tensor<long, 1>()(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape,
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    IPC2TFunctor<Device, T>()(
        context->eigen_device<Device>(),
        input_shape,
        input_address,
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("IPC2Tensor").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      IPC2Tensor<CPUDevice, T>);
REGISTER_CPU(float);


// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_IPC2T.cu.cc. */ \
  extern template IPC2TFunctor<GPUDevice, float>;              \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("IPC2Tensor")
      .Device(DEVICE_GPU).TypeConstraint<T>("T")
      .HostMemory("shape")
      .HostMemory("address"), \
      IPC2Tensor<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
