#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "ipc2tensor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// GPU specialization of actual computation.
template<typename T>
struct IPC2TFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, int64 address, T** ipc_memory, T* out) {
    if(!ipc_memory) {
      cudaIpcMemHandle_t* ipc_handle = reinterpret_cast<cudaIpcMemHandle_t*> (address);
      cudaIpcOpenMemHandle((void**) (ipc_memory), *ipc_handle, cudaIpcMemLazyEnablePeerAccess);
    }
    cudaMemcpy(out, *ipc_memory, size * sizeof(T), cudaMemcpyDeviceToDevice);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct IPC2TFunctor<GPUDevice, float>;
template struct IPC2TFunctor<GPUDevice, double>;

#endif  // GOOGLE_CUDA
