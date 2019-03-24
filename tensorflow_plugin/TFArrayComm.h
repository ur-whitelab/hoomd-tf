// Copyright (c) 2018 Andrew White at the University of Rochester
//  This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

#ifndef _IPC_ARRAY_COMM_
#define _IPC_ARRAY_COMM_

#include <hoomd/ExecutionConfiguration.h>
#include <hoomd/GlobalArray.h>
#include <string.h>
#include <sys/mman.h>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include "CommStruct.h"


/*! \file TFArrayComm.h
    \brief Declaration of TFArrayComm class
*/

namespace hoomd_tf {

  // I do not use specialization
  // to treat the CommMode because you cannot do partial specialization of a
  // method. The overhead of the ifs is nothing, since the compiler will see them
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

  /*! Template class for TFCompute
  *  \tfparam M If TF is on CPU or GPU.
  */
  template <TFCommMode M, typename T>
  class TFArrayComm {
  public:

  /*! Default constructor. Just checks GPU exists if used
  */
    TFArrayComm() {
      checkDevice();
    }

  /*! Normal constructor.
  * \param gpu_array The local array which will store communicated data
  * \param name The name of the array
  */
    TFArrayComm(GlobalArray<T>& gpu_array, const char* name)
        : _comm_struct(gpu_array, name),
          _array(&gpu_array) {
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
      _comm_struct = std::move(other._comm_struct);
      return *this;
    }

    ~TFArrayComm() {
      this->deallocate();
    }

    /*! Copy contents of given array to our array
    */
    void receiveArray(const GlobalArray<T>& array) {
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

    /*! Returns our underlying array as a vector
    */
    std::vector<T> getArray() const {
      ArrayHandle<T> handle(*_array, access_location::host, access_mode::read);
      return std::vector<T>(handle.data, handle.data + _array->getNumElements());
    }

    /*! Return memory address of the CommStruct wrapper around our array
    */
    int64_t getAddress() const {
        //_comm_struct.print(std::cout) << std::endl;
        //this insanity is because I need to cast to base class
        //then get pointer to that.
        return reinterpret_cast<int64_t>(static_cast<const CommStruct*>(&_comm_struct));
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
      #ifdef ENABLE_CUDA
      if (M == TFCommMode::GPU) {
        cudaEvent_t ipc_event;
        // flush errors
        TF_CHECK_CUDA_ERROR();
        cudaEventCreateWithFlags(
            &ipc_event, cudaEventInterprocess | cudaEventDisableTiming);
        _comm_struct.event_handle = ipc_event;
        TF_CHECK_CUDA_ERROR();
      }
      #endif
    }

    void deallocate() {
      if (M == TFCommMode::GPU) {
        #ifdef ENABLE_CUDA
        cudaEventDestroy(_comm_struct.event_handle);
        #endif
      }
    }

    using value_type = T;

  private:
    CommStructDerived<T> _comm_struct;
    GlobalArray<T>* _array;
  };

  void export_TFArrayComm(pybind11::module& m);

}
#endif  //_IPC_ARRAY_COMM_
