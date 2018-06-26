#include "tensor2ipc.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include <sys/mman.h>
#include <typeinfo>

using namespace tensorflow;


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct TF2IPCFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, int64 address, const T* in) {
    //TODO: access address
    Scalar4* output_buffer = reinterpret_cast<Scalar4*> (address);
    for(int i = 0; i < size; ++i) {
      output_buffer[i].x = in[4 * i + 0];
      output_buffer[i].y = in[4 * i + 1];
      output_buffer[i].z = in[4 * i + 2];
      output_buffer[i].w = in[4 * i + 3];
    }
  }
};


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class TensorToIpcOp : public OpKernel {
 public:
  explicit TensorToIpcOp(OpKernelConstruction* c) : OpKernel(c) {

    LOG(INFO) << "TensorToIpcOp construction starting";
    //get shape
    c->GetAttr("size", &_input_size);

    //get memory address
    c->GetAttr("address", &_output_address);

  }

  void Compute(OpKernelContext* context) override {

    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>().data();


    // Do the computation.
    TF2IPCFunctor<Device, T>()(
        context->eigen_device<Device>(),
        _input_size,
        _output_address,
        input);
  }

private:
  int _input_size;
  int64 _output_address;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TensorToIpc").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      TensorToIpcOp<CPUDevice, T>);
REGISTER_CPU(float);


// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_IPC2T.cu.cc. */ \
  extern template TF2IPCFunctor<GPUDevice, float>;              \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TensorToIpc")
      .Device(DEVICE_GPU).TypeConstraint<T>("T")
      .HostMemory("shape")
      .HostMemory("address"), \
      TensorToIpcOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
