#include "tensor2ipc.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include <sys/mman.h>
#include <typeinfo>

using namespace tensorflow;

//TODO: This class is not threadsafe.
//We need to use a resource manager to achive that


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("TensorToIpc")
    .Input("input: T")
    .Attr("T: {float, double}")
    .Attr("maxsize: int")
    .Attr("address: int")
    .SetIsStateful()
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
  void operator()(const CPUDevice& d, int size, void* out, const T* in, T** ipc_memory) {
    T* output_buffer = reinterpret_cast<T*> (out);
    std::memcpy(output_buffer, in, sizeof(T) * size);
  }
};


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class TensorToIpcOp : public OpKernel {
 public:
  explicit TensorToIpcOp(OpKernelConstruction* c) : OpKernel(c), _ipc_memory(nullptr) {

    //get shape
    c->GetAttr("maxsize", &_input_size);

    //get memory address
    int64 tmp;
    c->GetAttr("address", &tmp);
    _output_memory = reinterpret_cast<void*> (tmp);

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
        _output_memory,
        input.data(),
        &_ipc_memory);
  }

private:
  int _input_size;
  void* _output_memory;
  T* _ipc_memory;
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
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TensorToIpc") \
      .Device(DEVICE_GPU).TypeConstraint<T>("T"),	\
      TensorToIpcOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA
