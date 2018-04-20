#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "hoomd.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void HoomdCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * ldg(in + i);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void HoomdFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int size, const T* in, T* out) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for Hoomd of computing
  // block count and thread_per_block count.
  int block_count = 1024;
  int thread_per_block = 20;
  HoomdCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct HoomdFunctor<GPUDevice, float>;
template struct HoomdFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
