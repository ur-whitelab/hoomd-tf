#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensor2ipc.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// GPU specialization of actual computation.
template<typename T>
struct TF2IPCFunctor<GPUDevice, T> {
    void operator()(const GPUDevice& d, int size, int64 address, const T* in, T** ipc_memory) {
    if(!(*ipc_memory)) {
      cudaIpcMemHandle_t* ipc_handle = reinterpret_cast<cudaIpcMemHandle_t*> (address);
      cudaIpcOpenMemHandle((void**) (ipc_memory), *ipc_handle, cudaIpcMemLazyEnablePeerAccess);
    }
    cudaMemcpy(*ipc_memory, in, size * sizeof(T), cudaMemcpyDeviceToDevice);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct TF2IPCFunctor<GPUDevice, float>;
template struct TF2IPCFunctor<GPUDevice, double>;

#endif  // GOOGLE_CUDA
