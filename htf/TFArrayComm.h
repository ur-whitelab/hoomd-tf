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

namespace hoomd_tf
    {

    // This uses an enum instead of specialization
    // to treat the CommMode because you cannot do partial specialization of a
    // method. The overhead of the ifs is nothing, since the compiler will see them
    // as if (1 == 0) and if(1 == 1) and optimize them.

    enum class TFCommMode { GPU, CPU };

    /*! TFCommMode class
     *  Template class for TFArrayComm
     *  \tfparam M Whether TF is on CPU or GPU.
     */
    template <TFCommMode M, typename T>
        class TFArrayComm
        {
        public:

        /*! Default constructor. Just checks GPU exists if used
         */
        TFArrayComm()
            {
            checkDevice();
            }

        /*! Normal constructor.
         * \param gpu_array The local array which will store communicated data
         * \param name The name of the array
         */
        TFArrayComm(GlobalArray<T>& gpu_array,
            const char* name,
            std::shared_ptr<const ExecutionConfiguration> exec_conf)
            : _comm_struct(gpu_array, name),
              _array(&gpu_array),
              m_exec_conf(exec_conf)
                {
                checkDevice();
                allocate();
                }

        //! Copy constructor
        TFArrayComm(TFArrayComm&& other)
            {
            // use the assignment overloaded operator
            *this = std::move(other);
            }

        //! overloaded assignment operator
        TFArrayComm& operator=(TFArrayComm&& other)
            {
            checkDevice();
            // copy over variables
            _array = other._array;
            _comm_struct = std::move(other._comm_struct);
            m_exec_conf = other.m_exec_conf;
            return *this;
            }

        //! destructor
        ~TFArrayComm()
            {
            this->deallocate();
            }

        /*! Copy contents of given array to this array
         *  \param array the array whose contents to copy into this one
         */
        void receiveArray(const GlobalArray<T>& array, int offset = 0, unsigned int size = 0)
            {
            // convert size into mem size
            if(!size)
                {
                size = _comm_struct.mem_size;
                }
            else
                {
                size = size * sizeof(T);
                }
            assert(offset + size / sizeof(T) <= array.getNumElements());
            if (M == TFCommMode::CPU)
                {
                ArrayHandle<T> handle(*_array,
                    access_location::host,
                    access_mode::overwrite);
                ArrayHandle<T> ohandle(array,
                    access_location::host,
                    access_mode::read);
                memcpy(handle.data, ohandle.data + offset, size);
                }
            else
                {
                #ifdef ENABLE_CUDA
                    ArrayHandle<T> handle(*_array,
                        access_location::device,
                        access_mode::overwrite);
                    ArrayHandle<T> ohandle(array,
                        access_location::device,
                        access_mode::read);
                    cudaMemcpy(handle.data,
                        ohandle.data + offset,
                        size,
                        cudaMemcpyDeviceToDevice);
                    CHECK_CUDA_ERROR();
                #endif
                }
            }

        //! Set all values in an array to target value
        //! \param v target int value to fill array with
        void memsetArray(int v)
            {
            if (M == TFCommMode::CPU)
                {
                ArrayHandle<T> handle(*_array,
                    access_location::host,
                    access_mode::overwrite);
                memset( static_cast<void*> (handle.data), v, _comm_struct.mem_size);
                }
            else
                {
                #ifdef ENABLE_CUDA
                    ArrayHandle<T> handle(*_array,
                        access_location::device,
                        access_mode::overwrite);
                    cudaMemset(static_cast<void*> (handle.data), v, _comm_struct.mem_size);
                    CHECK_CUDA_ERROR();
                #endif
                }
            }

        //! Set offset value for batching
        void setOffset(size_t offset)
        {
        _comm_struct.offset = offset;
        }

        //! Set batch size
        void setBatchSize(size_t N)
            {
            _comm_struct.num_elements[0] = N;
            }

        /*! Returns our underlying array as a vector
         */
        std::vector<T> getArray() const
            {
            ArrayHandle<T> handle(*_array, access_location::host, access_mode::read);
            return std::vector<T>(handle.data, handle.data + _array->getNumElements());
            }

        /*! Return memory address of the CommStruct wrapper around our array
         */
        int64_t getAddress() const
            {
            //need to cast to base class then get pointer to that.
            return reinterpret_cast<int64_t>(static_cast<const CommStruct*>(&_comm_struct));
            }

        #ifdef ENABLE_CUDA
            //! Set the CUDA stream that will be used
            void setCudaStream(cudaStream_t s) { _comm_struct.stream = s;}
            cudaStream_t getCudaStream() const
                {
                return _comm_struct.stream;
                }
        #endif

        protected:
        //! Make sure that CUDA is enabled before running in GPU mode
        void checkDevice()
            {
            #ifndef ENABLE_CUDA
                if (M == TFCommMode::GPU)
                    throw std::runtime_error(
                        "CUDA compilation not enabled so cannot use GPU CommMode");
            #endif
            }

        //! Create a CUDA event for our CommStruct, if CUDA exists
        void allocate()
            {
            #ifdef ENABLE_CUDA
                if (M == TFCommMode::GPU)
                    {
                    cudaEvent_t ipc_event;
                    // flush errors
                    CHECK_CUDA_ERROR();
                    cudaEventCreateWithFlags(
                        &ipc_event, cudaEventInterprocess | cudaEventDisableTiming);
                    _comm_struct.event_handle = ipc_event;
                    CHECK_CUDA_ERROR();
                    }
            #endif
            }

        //! Deallocate the CUDA event for our CommStruct, if CUDA exists
        void deallocate()
            {
            if (M == TFCommMode::GPU)
                {
                #ifdef ENABLE_CUDA
                    cudaEventDestroy(_comm_struct.event_handle);
                #endif
                }
            }

        using value_type = T;

        private:
        CommStructDerived<T> _comm_struct;
        GlobalArray<T>* _array;
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
        };

    //! Export python binding
    void export_TFArrayComm(pybind11::module& m);

    }

#endif  //_IPC_ARRAY_COMM_
