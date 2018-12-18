#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <iostream>
#include "tf2hoomd.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// GPU specialization of actual computation.
// TODO Add cuda error checks here
template <typename T>
void TF2IPCFunctor<GPUDevice, T>::operator()(const GPUDevice& d, int size,
    CommStruct_t* handle, const T* in) {
  cudaMemcpy(handle->mem_handle, (const void*)(in), size * sizeof(T),
             cudaMemcpyDeviceToDevice);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct TF2IPCFunctor<GPUDevice, float>;
template struct TF2IPCFunctor<GPUDevice, double>;

#endif  // GOOGLE_CUDA
