
#ifndef CommStruct_H_H
#define CommStruct_H_H

#if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
#include <cuda_runtime.h>
#include <hoomd/GPUArray.h>
#endif

namespace hoomd_tf {

  struct CommStruct {

    CommStruct(const size_t* num_elements,
                size_t num_dims, size_t element_size,
                const char* name) :
      num_elements(num_elements),
      num_dims(num_dims),
      element_size(element_size),
      name(name) {
      size_t size = 1;
      for(unsigned int i; i < num_dims; i++)
        size *= num_elements[i];
      mem_size = size * element_size;
    }

    CommStruct() :
      num_elements(0),
      num_dims(0),
      element_size(0),
      name("null") {}

    CommStruct& operator=(CommStruct& other) {
      num_elements = other.num_elements;
      num_dims = other.num_dims;
      element_size = other.element_size;
      name = other.name;
      #if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
      event_handle = other.event_handle;
      stream = other.stream;
      #endif

      other.mem_size = 0;

      return *this;
    }

    virtual void* read_gpu_memory();
    virtual void* read_cpu_memory();
    virtual void* write_gpu_memory();
    virtual void* write_cpu_memory();

    const size_t* num_elements;
    size_t num_dims;
    size_t element_size;
    size_t mem_size;
    const char* name;
    //TODO Why is ENABLE_CUDA set for compilng tf code? We don't have any hoomd headers...
    #if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
    cudaEvent_t event_handle;
    cudaStream_t stream = 0;
    #endif
  };

  template <typename T>
  struct CommStructDerived : CommStruct {
    GPUArray<T>& array;

    CommStructDerived() : array(GPUArray<Scalar>(1))
    {

    }

    void read_gpu_memory(void *dest, size_t n) {
      ArrayHandle<T> handle(array, access_location::device, access_mode::read);
      cudaMemcpy(dest, handle.data, n, cudaMemcpyDeviceToDevice);
    }
    void read_cpu_memory(const void* src, size_t n) {
      ArrayHandle<T> handle(array, access_location::host, access_mode::read);
      memcpy(handle.data, src, n);
    }
    void write_gpu_memory(void* dest, size_t n) {
      ArrayHandle<T> handle(array, access_location::device, access_mode::overwrite);

    }
    void write_cpu_memory(void* dest, size_t n) {
      ArrayHandle<T> handle(array, access_location::host, access_mode::overwrite);
      memcpy(dest, handle.data, n);
    }
  };
}
#endif