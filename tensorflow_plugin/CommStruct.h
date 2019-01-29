
#ifndef CommStruct_H_H
#define CommStruct_H_H

#if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace hoomd_tf {

  struct CommStruct {

    CommStruct(const std::vector<int> num_elements,
                int num_dims, size_t element_size,
                const char* name) :
      num_elements(num_elements.data()),
      num_dims(num_dims),
      element_size(element_size),
      name(name) {
      size_t size = 1;
      for(unsigned int i = 0; i < num_dims; i++)
        size *= num_elements[i];
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

    virtual void read_gpu_memory(void *dest, size_t n);
    virtual void read_cpu_memory(void *dest, size_t n);
    virtual void write_gpu_memory(const void *src, size_t n);
    virtual void write_cpu_memory(const void *src, size_t n);

    const int* num_elements; //would be better as size_t, but need this for TF
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

  #ifndef GOOGLE_CUDA
  #include <hoomd/GPUArray.h>
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

    #if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
    void read_gpu_memory(void *dest, size_t n) {
      ArrayHandle<T> handle(*_array, access_location::device, access_mode::read);
      cudaMemcpy(dest, handle.data, n, cudaMemcpyDeviceToDevice);
    }
    void write_gpu_memory(void* dest, size_t n) {
      ArrayHandle<T> handle(*_array, access_location::device, access_mode::overwrite);

    }
    #endif
    void read_cpu_memory(const void* src, size_t n) {
      ArrayHandle<T> handle(*_array, access_location::host, access_mode::read);
      memcpy(handle.data, src, n);
    }
    void write_cpu_memory(void* dest, size_t n) {
      ArrayHandle<T> handle(*_array, access_location::host, access_mode::overwrite);
      memcpy(dest, handle.data, n);
    }
  };

  template<> CommStructDerived<Scalar4>::CommStructDerived(GPUArray<Scalar4>&, const char*);
  template<> CommStructDerived<Scalar>::CommStructDerived(GPUArray<Scalar>&, const char*);
  #endif //GOOGLE_CUDA
}

#endif //guard