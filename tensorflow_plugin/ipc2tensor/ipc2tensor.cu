#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <iostream>
#include "ipc2tensor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// GPU specialization of actual computation.
template <typename T>
void IPC2TFunctor<GPUDevice, T >::operator()(
    const GPUDevice& d, int size, IPCStruct_t* ipc_memory,T* out) {

  //cudaEventSynchronize(ipc_memory.event);
  cudaMemcpy((void*)(out), (const void*)(ipc_memory->array), ipc_memory->num_elements * sizeof(T),
             cudaMemcpyDeviceToDevice);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct IPC2TFunctor<GPUDevice, float> >;
template struct IPC2TFunctor<GPUDevice, double> >;

#endif  // GOOGLE_CUDA
