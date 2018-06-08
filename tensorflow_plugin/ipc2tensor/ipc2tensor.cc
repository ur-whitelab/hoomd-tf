#include "ipc2tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include <stdio.h>
#include <sys/mman.h>


using namespace tensorflow;


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct IPC2TFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, int64 address, T* out) {
    //TODO: access address

    for(int i = 0; i < size; ++i)
      out[i] = 0;
  }
};

// CPU Initializer
template<>
struct IPC2TInitialize<CPUDevice> {
  bool operator()(int32 size, int64 address) {
    // check shared memory
    // Scalar4* input_buffer = reinterpret_cast<Scalar4*> (address);
    LOG(ERROR) << "about to try reading from " << address;
    // check for magic byte sequence
    //return input_buffer[0].x == MMAP_MAGIC_FLOAT;
    return true;
  }
};


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class IpcToTensorOp : public OpKernel {
 public:
  explicit IpcToTensorOp(OpKernelConstruction* c) : OpKernel(c) {

    //get shape
    c->GetAttr("shape", &_input_shape);

    //get memory address
    int64 input_address;
    c->GetAttr("address", &_input_address);

    int temp_dims [2] = {_input_shape, 3};
    //TODO: why is this necessary?!
    TensorShapeUtils::MakeShape(temp_dims, 1, &_output_shape);

    //call device initializer
    OP_REQUIRES(c, IPC2TInitialize<Device>()(_input_shape,
                              _input_address),
                errors::FailedPrecondition("Memory mapped buffer not accessible or invalid."));
    LOG(INFO) << "OP constructed and mmap connection validated";

  }

  void Compute(OpKernelContext* context) override {

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, _output_shape,
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, output_tensor->NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    IPC2TFunctor<Device, T>()(
        context->eigen_device<Device>(),
        _input_shape,
        _input_address,
        output_tensor->flat<T>().data());
  }

private:
  int32 _input_shape;
  int64 _input_address;
  TensorShape _output_shape;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("IpcToTensor").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      IpcToTensorOp<CPUDevice, T>);
REGISTER_CPU(float);


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
