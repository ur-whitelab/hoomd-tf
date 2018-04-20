#ifndef KERNEL_HOOMD_H_
#define KERNEL_HOOMD_H_

template <typename Device, typename T>
struct HoomdFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct HoomdFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif //KERNEL_HOOMD_H_