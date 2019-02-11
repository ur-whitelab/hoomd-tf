
#ifndef CommStruct_H_H
#define CommStruct_H_H

#include <vector>
#include <iostream>
#include <cstddef>

#if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace hoomd_tf {

  struct CommStruct {

    CommStruct(int num_dims, size_t element_size,
                const char* name) :
      num_dims(num_dims),
      element_size(element_size),
      name(name) {
    }

    void set_num_elements(int* num_elements_t) {
      size_t size = 1;
      num_elements = new int[num_dims];
      for(unsigned int i = 0; i < num_dims; i++) {
        num_elements[i] = num_elements_t[i];
        size *= num_elements[i];
      }

      mem_size = size * element_size;
    }

    CommStruct() {

    }

    CommStruct& operator=(const CommStruct& other) {
      num_elements = other.num_elements;
      num_dims = other.num_dims;
      element_size = other.element_size;
      name = other.name;
      #if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
      event_handle = other.event_handle;
      stream = other.stream;
      #endif

      return *this;
    }

  std::ostream& print(std::ostream& os) const {
    os << name <<  ":\n  " << "Dims: [";
    for(unsigned int i = 0; i < num_dims; i++) {
      os << num_elements[i] << " ";
    }
    os << "]\n  " << "Element Size: " << element_size << "\n";
    return os;
  }
    virtual void read_gpu_memory(void *dest, size_t n) = 0;
    virtual void read_cpu_memory(void *dest, size_t n) = 0;
    virtual void write_gpu_memory(const void *src, size_t n) = 0;
    virtual void write_cpu_memory(const void *src, size_t n) = 0;

    int* num_elements; //would be better as size_t, but need this for TF
    int num_dims;
    size_t element_size;
    size_t mem_size;
    const char* name;
    //TODO Why is ENABLE_CUDA set for compilng tf code? We don't have any hoomd headers...
    #if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
    cudaEvent_t event_handle;
    cudaStream_t stream = 0;
    #endif
  };

}
#ifndef GOOGLE_CUDA
#include <hoomd/GPUArray.h>
namespace hoomd_tf {
  template <typename T>
  struct CommStructDerived : CommStruct {
    GPUArray<T>* _array;

    CommStructDerived(GPUArray<T>& array, const char* name) {
      T::unimplemented_function;
    }

    CommStructDerived() {

    }

    CommStructDerived& operator=(const CommStructDerived<T>& other) {
      _array = other._array;
      CommStruct::operator=(other);
      return *this;
    }

    #ifdef ENABLE_CUDA
    void read_gpu_memory(void *dest, size_t n) override {
      ArrayHandle<T> handle(*_array, access_location::device, access_mode::read);
      cudaMemcpy(dest, handle.data, n, cudaMemcpyDeviceToDevice);
    }
    void write_gpu_memory(const void* src, size_t n) override {
      ArrayHandle<T> handle(*_array, access_location::device, access_mode::overwrite);
      cudaMemcpy(handle.data, src, n, cudaMemcpyDeviceToDevice);
    }
    #else 
    void read_gpu_memory(void *dest, size_t n) override {
      throw "Should not call read_gpu_memory without CUDA";
    }
    void write_gpu_memory(const void* src, size_t n) override {
      throw "Should not call read_gpu_memory without CUDA";
    }
    #endif //ENABLE_CUDA
    void read_cpu_memory(void* dest, size_t n) override {
      ArrayHandle<T> handle(*_array, access_location::host, access_mode::read);
      memcpy(dest, handle.data, n);
    }
    void write_cpu_memory(const void* src, size_t n) override {
      ArrayHandle<T> handle(*_array, access_location::host, access_mode::overwrite);
      memcpy(handle.data, src, n);
    }
  };

  template<> CommStructDerived<Scalar4>::CommStructDerived(GPUArray<Scalar4>&, const char*);
  template<> CommStructDerived<Scalar>::CommStructDerived(GPUArray<Scalar>&, const char*);
}
#endif //GOOGLE_CUDA
#endif //guard