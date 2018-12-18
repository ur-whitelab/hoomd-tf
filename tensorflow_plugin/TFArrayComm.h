#ifndef _IPC_ARRAY_COMM_
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
  #ifndef NDEBUG
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
  template <TFCommMode M, typename T>
  class TFArrayComm {
  public:
    TFArrayComm()
        : _shared_array(nullptr),
          _ipc_handle(nullptr),
          _array(nullptr),
          _array_size(0),
          _own_array(false)
          {
      checkDevice();
    }

    TFArrayComm(void* _shared_array, size_t array_size,
                std::shared_ptr<const ExecutionConfiguration> exec_conf)
        : _shared_array(_shared_array),
          _ipc_handle(nullptr),
          _array(nullptr),
          _array_size(array_size),
          _own_array(true)
          {
      checkDevice();
      _array = new GPUArray<T>(array_size, exec_conf);
      allocate();
    }

    TFArrayComm(GPUArray<T>& gpu_array,
                size_t num_elements)
        : _shared_array(nullptr),
          _ipc_handle(nullptr),
        _array(&gpu_array),
          _array_size(0),
          _own_array(false) {
      _array_size = num_elements * sizeof(T);
      checkDevice();
      allocate();
    }

    TFArrayComm(GPUArray<T>& gpu_array)
        : _shared_array(nullptr),
          _ipc_handle(nullptr),
        _array(&gpu_array),
          _array_size(0),
          _own_array(false)
          {
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
      _array_size = other._array_size;
      other._array_size = 0;
      _own_array = other._own_array;
      // prevent other from deleting array
      other._own_array = false;
      _shared_array = other._shared_array;
      other._shared_array = nullptr;
      _ipc_handle = other._ipc_handle;
      other._ipc_handle = nullptr;

      return *this;
    }

    ~TFArrayComm() {
      if (_own_array) delete _array;
      this->deallocate();
    }

    void receiveArray(const GPUArray<T>& array) {
      if (M == TFCommMode::CPU) {
        ArrayHandle<T> handle(*_array, access_location::host,
                              access_mode::overwrite);
        ArrayHandle<T> ohandle(array, access_location::host,
                        access_mode::read);
        memcpy(handle.data, ohandle.data, _array->getNumElements() * sizeof(T));
      } else {
        #ifdef ENABLE_CUDA
        ArrayHandle<T> handle(*_array, access_location::device,
                              access_mode::overwrite);
        ArrayHandle<T> ohandle(array, access_location::device,
                  access_mode::read);
        cudaMemcpy(handle.data, ohandle.data, _array->getNumElements() * sizeof(T),
                  cudaMemcpyDeviceToDevice);
        TF_CHECK_CUDA_ERROR();
        #endif
      }
    }

      void memsetArray(int v) {
      if (M == TFCommMode::CPU) {
        ArrayHandle<T> handle(*_array, access_location::host,
                              access_mode::overwrite);
        memset( static_cast<void*> (handle.data), v, _array->getNumElements() * sizeof(T));
      } else {
        #ifdef ENABLE_CUDA
        ArrayHandle<T> handle(*_array, access_location::device,
                              access_mode::overwrite);
        cudaMemset(static_cast<void*> (handle.data), v, _array->getNumElements() * sizeof(T));
        TF_CHECK_CUDA_ERROR();
        #endif
      }
    }

    void receive() {
      if (M == TFCommMode::CPU) {
        ArrayHandle<T> handle(*_array, access_location::host,
                              access_mode::overwrite);
        memcpy(handle.data, _shared_array,  _array_size);
      } else {
        #ifdef ENABLE_CUDA
        ArrayHandle<T> handle(*_array, access_location::device,
                              access_mode::overwrite);
        cudaMemcpy(handle.data, _shared_array, _array->getNumElements() * sizeof(T),
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
        cudaMemcpyAsync(handle.data, _shared_array,
                        _array->getNumElements() * sizeof(T),
                        cudaMemcpyDeviceToDevice, _ipc_handle->stream);
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
        fun.template call<M>(handle.data, static_cast<T*>(_shared_array));
      } else {
        #ifdef ENABLE_CUDA
        ArrayHandle<T> handle(*_array, access_location::device,
                              access_mode::readwrite);
        fun._stream = &_ipc_handle->stream;  // set stream for functor
        fun.template call<M>(handle.data, static_cast<T*>(_shared_array));
        #endif
      }
    }

    void send() {
      if (M == TFCommMode::CPU) {
        ArrayHandle<T> handle(*_array, access_location::host, access_mode::read);
        memcpy(_shared_array, handle.data, _array_size);
      } else {
        #ifdef ENABLE_CUDA
        ArrayHandle<T> handle(*_array, access_location::device,
                              access_mode::read);
        cudaMemcpy(_shared_array, handle.data, _array_size,
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
        cudaMemcpyAsync(_shared_array, handle.data, _array_size,
                        cudaMemcpyDeviceToDevice, _ipc_handle->stream);
        TF_CHECK_CUDA_ERROR();
        #endif
      }
    }

    std::vector<T> getArray() const {
      ArrayHandle<T> handle(*_array, access_location::host, access_mode::read);
      return std::vector<T>(handle.data, handle.data + _array->getNumElements());
    }

    int64_t getAddress() const {
        return reinterpret_cast<int64_t>(_ipc_handle);
    }

    #ifdef ENABLE_CUDA
    void setCudaStream(cudaStream_t s) { _ipc_handle->stream = s;}
    cudaStream_t getCudaStream() const {
      return _ipc_handle->stream; }
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
      _ipc_handle = static_cast<CommStruct_t*> (malloc(sizeof(CommStruct_t)));
      if (_array_size == 0) _array_size = sizeof(T) * _array->getNumElements();
      if (M == TFCommMode::CPU) {
        if(!_shared_array)
          _shared_array = calloc(_array_size / sizeof(T), sizeof(T));
      }
      #ifdef ENABLE_CUDA
      cudaEvent_t ipc_event;
      if (M == TFCommMode::GPU) {
        // flush errors
        TF_CHECK_CUDA_ERROR();
        if (_shared_array) {
          // we will open the existing mapped memory in cuda
        } else {
          // We will create a shared block
          cudaMalloc((void**)&_shared_array, _array_size);
          _ipc_handle->mem_handle = _shared_array;
          cudaEventCreateWithFlags(
              &ipc_event, cudaEventInterprocess | cudaEventDisableTiming);
          _ipc_handle->event_handle = ipc_event;
        }
        TF_CHECK_CUDA_ERROR();
      }
      _ipc_handle->stream = 0;
      #endif
      _ipc_handle->mem_handle = _shared_array;
      _ipc_handle->num_elements = _array_size / sizeof(T);
      _ipc_handle->element_size = sizeof(T);
    }

    void deallocate() {
      if (M == TFCommMode::CPU && _shared_array) {
        free(_shared_array);
      }
      if (M == TFCommMode::GPU) {
      #ifdef ENABLE_CUDA
        if (_ipc_handle) {
          cudaFree(_shared_array);
          cudaEventDestroy(_ipc_handle->event_handle);
        }
      #endif
      }
      if(_ipc_handle)
        free(_ipc_handle);
    }

    void* _shared_array;

  private:
    CommStruct_t* _ipc_handle;
    GPUArray<T>* _array;
    size_t _array_size;
    bool _own_array;
  };

  void export_TFArrayComm(pybind11::module& m);

}
#endif  //_IPC_ARRAY_COMM_
