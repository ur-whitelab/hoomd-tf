w#ifndef _IPC_ARRAY_COMM_
#define _IPC_ARRAY_COMM_

#include <hoomd/ExecutionConfiguration.h>
#include <hoomd/GPUArray.h>
#include <string.h>
#include <sys/mman.h>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include "CommStruct.h"


namespace hoomd_tf {

  // I do not use specialization
  // to treat the CommMode because you cannot do partial specialization of a
  // method The overhead of the ifs is nothing, since the compiler will see them
  // as if (1 == 0) and if(1 == 1) so they will be optimized.

  enum class TFCommMode { GPU, CPU };

  // need this without access to context
  #ifdef ENABLE_CUDA
  #ifndef z
  void tf_check_cuda_error(cudaError_t err, const char* file, unsigned int line);
  #define TF_CHECK_CUDA_ERROR()                           \
    {                                                      \
      cudaError_t err_sync = cudaGetLastError();           \
      tf_check_cuda_error(err_sync, __FILE__, __LINE__);  \
      cudaError_t err_async = cudaDeviceSynchronize();     \
      tf_check_cuda_error(err_async, __FILE__, __LINE__); \
    }
  #else
  #define TF_CHECK_CUDA_ERROR()
  #endif  // NDDEBUG
  #else   // ENABLE_CUDA
  #define TF_CHECK_CUDA_ERROR()
  #endif  // ENABLE_CUDA

  // M: Communication mode (GPU or CPU)
  // two-way communication where one owns array and other does not
  // own_array: if it owns the array, it does not own the underlying data. Thus
  // can use array as mapping
  //! own_array: it has reference to underlying data.
  template <TFCommMode M, class A>
  class TFArrayComm {
    typedef typename A::value_type T;
  public:
    TFArrayComm()
        : comm_struct(nullptr),
          _array(nullptr),
          _own_array(false)
          {
      checkDevice();
    }

    TFArrayComm(void* array, size_t array_size, const char* name,
                std::shared_ptr<const ExecutionConfiguration> exec_conf)
        : _comm_struct(array, {array_size}, 1, sizeof(T), name),
          _array(nullptr),
          _own_array(true)
          {
      checkDevice();
      _array = new A(array_size, exec_conf);
      allocate();
    }

<<<<<<< HEAD
    TFArrayComm(GPUArray<T>& gpu_array,
                size_t* num_elements, size_t, num_dims const char* name)
        : _comm_struct(nullptr, num_elements, num_dims, sizeof(T)),
          _comm_struct(nullptr),
=======
    TFArrayComm(A& gpu_array,
                const size_t* num_elements, size_t num_dims,
                const char* name)
        : _comm_struct(gpu_array, num_elements, num_dims, sizeof(T), name),
>>>>>>> b771e30bda6ab74a3a0369d9304f592b4d0e7c2d
        _array(&gpu_array),
          _own_array(false) {
      checkDevice();
      allocate();
    }

    TFArrayComm(TFArrayComm&& other) {
      // use the assignment operator
      *this = std::move(other);
    }
    TFArrayComm& operator=(TFArrayComm&& other) {
      checkDevice();
      // copy over variables
      _array = other._array;
<<<<<<< HEAD
      _comm_struct = other._comm_struct;
      _own_array = other._own_array;
      // prevent other from deleting array
      other._own_array = false;
      other._comm_struct = nullptr;

=======
      _comm_struct = other._comm_struct.;
      other._comm_struct = nullptr;
      _own_array = other._own_array;
      // prevent other from deleting array
      other._own_array = false;
>>>>>>> b771e30bda6ab74a3a0369d9304f592b4d0e7c2d
      return *this;
    }

    ~TFArrayComm() {
      if (_own_array) delete _array;
      this->deallocate();
    }

    void receiveArray(const A& array) {
      if (M == TFCommMode::CPU) {
        ArrayHandle<T> handle(*_array, access_location::host,
                              access_mode::overwrite);
        ArrayHandle<T> ohandle(array, access_location::host,
                        access_mode::read);
        memcpy(handle.data, ohandle.data, _comm_struct.mem_size);
      } else {
        #ifdef ENABLE_CUDA
        ArrayHandle<T> handle(*_array, access_location::device,
                              access_mode::overwrite);
        ArrayHandle<T> ohandle(array, access_location::device,
                  access_mode::read);
        cudaMemcpy(handle.data, ohandle.data, _comm_struct.mem_size,
                  cudaMemcpyDeviceToDevice);
        TF_CHECK_CUDA_ERROR();
        #endif
      }
    }

      void memsetArray(int v) {
      if (M == TFCommMode::CPU) {
        ArrayHandle<T> handle(*_array, access_location::host,
                              access_mode::overwrite);
        memset( static_cast<void*> (handle.data), v, _comm_struct.mem_size);
      } else {
        #ifdef ENABLE_CUDA
        ArrayHandle<T> handle(*_array, access_location::device,
                              access_mode::overwrite);
        cudaMemset(static_cast<void*> (handle.data), v, _comm_struct.mem_size);
        TF_CHECK_CUDA_ERROR();
        #endif
      }
    }

    void receive() {
      if (M == TFCommMode::CPU) {
        ArrayHandle<T> handle(*_array, access_location::host,
                              access_mode::overwrite);
        memcpy(handle.data, _comm_struct.mem_handle,  _comm_struct.mem_size);
      } else {
        #ifdef ENABLE_CUDA
        ArrayHandle<T> handle(*_array, access_location::device,
                              access_mode::overwrite);
        cudaMemcpy(handle.data, _comm_struct.mem_handle, _comm_struct.mem_size,
                  cudaMemcpyDeviceToDevice);
        TF_CHECK_CUDA_ERROR();
        #endif
      }
    }

    void receiveAsync() {
      if (M == TFCommMode::CPU) {
        receive();
      } else {
        #ifdef ENABLE_CUDA
        ArrayHandle<T> handle(*_array, access_location::device,
                              access_mode::overwrite);
        cudaMemcpyAsync(handle.data, _comm_struct.mem_handle,
                        _comm_struct.mem_size,
                        cudaMemcpyDeviceToDevice, _comm_struct.stream);
        TF_CHECK_CUDA_ERROR();
        #endif
      }
    }

    template <typename Func>
    void receiveOp(Func fun) {
      if (M == TFCommMode::CPU) {
        ArrayHandle<T> handle(*_array, access_location::host,
                              access_mode::readwrite);
        // have to use funny fun.template because compiler thinks
        // that the '<' means less than instead of start of template arguments
        fun.template call<M>(handle.data, static_cast<T*>(_comm_struct.mem_handle));
      } else {
        #ifdef ENABLE_CUDA
        ArrayHandle<T> handle(*_array, access_location::device,
                              access_mode::readwrite);
        fun._stream = &_comm_struct.stream;  // set stream for functor
        fun.template call<M>(handle.data, static_cast<T*>(_comm_struct.mem_handle));
        #endif
      }
    }

    void send() {
      if (M == TFCommMode::CPU) {
        ArrayHandle<T> handle(*_array, access_location::host, access_mode::read);
        memcpy(_comm_struct.mem_handle, handle.data, _comm_struct.mem_size);
      } else {
        #ifdef ENABLE_CUDA
        ArrayHandle<T> handle(*_array, access_location::device,
                              access_mode::read);
        cudaMemcpy(_comm_struct.mem_handle, handle.data, _comm_struct.mem_size,
                  cudaMemcpyDeviceToDevice);
        TF_CHECK_CUDA_ERROR();
        #endif
      }
    }

    void sendAsync() {
      if (M == TFCommMode::CPU) {
        send();
      } else {
        #ifdef ENABLE_CUDA
        ArrayHandle<T> handle(*_array, access_location::device,
                              access_mode::read);
        cudaMemcpyAsync(_comm_struct.mem_handle, handle.data, _comm_struct.mem_size,
                        cudaMemcpyDeviceToDevice, _comm_struct.stream);
        TF_CHECK_CUDA_ERROR();
        #endif
      }
    }

    std::vector<T> getArray() const {
      ArrayHandle<T> handle(*_array, access_location::host, access_mode::read);
      return std::vector<T>(handle.data, handle.data + _array->getNumElements());
    }

    int64_t getAddress() const {
        return reinterpret_cast<int64_t>(_comm_struct);
    }

    #ifdef ENABLE_CUDA
    void setCudaStream(cudaStream_t s) { _comm_struct.stream = s;}
    cudaStream_t getCudaStream() const {
      return _comm_struct.stream; }
    #endif

  protected:
    void checkDevice() {
  #ifndef ENABLE_CUDA
      if (M == TFCommMode::GPU)
        throw std::runtime_error(
            "CUDA compilation not enabled so cannot use GPU CommMode");
  #endif
    }

    void allocate() {
      _comm_struct = static_cast<CommStruct_t*> (malloc(sizeof(CommStruct_t)));
      if (_comm_struct.mem_size == 0) _comm_struct.mem_size = sizeof(T) * _array->getNumElements();
      if (M == TFCommMode::CPU) {
        if(!_comm_struct.mem_handle)
          _comm_struct.mem_handle = calloc(_comm_struct.mem_size / sizeof(T), sizeof(T));
      }
      #ifdef ENABLE_CUDA
      cudaEvent_t ipc_event;
      if (M == TFCommMode::GPU) {
        // flush errors
        TF_CHECK_CUDA_ERROR();
        if (_comm_struct.mem_handle) {
          // we will open the existing mapped memory in cuda
        } else {
          // We will create a shared block
          cudaMalloc((void**)&_comm_struct.mem_handle, _comm_struct.mem_size);
          _comm_struct.mem_handle = _comm_struct.mem_handle;
          cudaEventCreateWithFlags(
              &ipc_event, cudaEventInterprocess | cudaEventDisableTiming);
          _comm_struct.event_handle = ipc_event;
        }
        TF_CHECK_CUDA_ERROR();
      }
      _comm_struct.stream = 0;
      #endif
      _comm_struct.mem_handle = _comm_struct.mem_handle;
      _comm_struct.num_elements = _comm_struct.mem_size / sizeof(T);
      _comm_struct.element_size = sizeof(T);
    }

    void deallocate() {
      if (M == TFCommMode::CPU && _comm_struct.mem_handle) {
        free(_comm_struct.mem_handle);
      }
      if (M == TFCommMode::GPU) {
      #ifdef ENABLE_CUDA
        if (_comm_struct) {
          cudaFree(_comm_struct.mem_handle);
          cudaEventDestroy(_comm_struct.event_handle);
        }
      #endif
      }
    }

    using value_type = T;
    using array_type = A;

  private:
    CommStruct_t& _comm_struct;
    A* _array;
    bool _own_array;
  };

  void export_TFArrayComm(pybind11::module& m);

}
#endif  //_IPC_ARRAY_COMM_
