#include "tensor2ipc.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include <sys/mman.h>
#include <typeinfo>

using namespace tensorflow;


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("TensorToIpc")
    .Input("input: T")
    .Attr("T: {float, double}")
    .Attr("maxsize: int")
    .Attr("address: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int32 size;
      c->GetAttr("maxsize", &size);
      //check shape of input is not larger than buffer
      auto result = c->Value(c->NumElements(c->input(0)));
      if(!result || size > result)
        return errors::InvalidArgument("Input tensor may exceed buffer size, which is set by maxsize");
      return Status::OK();
    });

// CPU specialization of actual computation.
template <typename T>
struct TF2IPCFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, int64 address, const T* in, T** ipc_memory) {
    //TODO: access address
    T* output_buffer = reinterpret_cast<T*> (address);
    std::memcpy(output_buffer, in, sizeof(T) * size);
  }
};


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class TensorToIpcOp : public OpKernel {
 public:
  explicit TensorToIpcOp(OpKernelConstruction* c) : OpKernel(c), _ipc_memory(NULL) {

    //get shape
    c->GetAttr("maxsize", &_input_size);

    //get memory address
    c->GetAttr("address", &_output_address);

  }

  void Compute(OpKernelContext* context) override {

    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    if(input.size() > _input_size) {
      errors::InvalidArgument("Tensor input size is too large for output buffer!");
    }
    // Do the computation.
    TF2IPCFunctor<Device, T>()(
        context->eigen_device<Device>(),
        input.size(),
        _output_address,
        input.data(),
        _ipc_memory);
  }

private:
  int _input_size;
  int64 _output_address;
  T** _ipc_memory;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TensorToIpc").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      TensorToIpcOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);


// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_IPC2T.cu.cc. */ \
  extern template TF2IPCFunctor<GPUDevice, T>;              \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TensorToIpc")
      .Device(DEVICE_GPU).TypeConstraint<T>("T")
      TensorToIpcOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA
