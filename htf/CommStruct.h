// Copyright (c) 2020 HOOMD-TF Developers

#ifndef CommStruct_H_H
#define CommStruct_H_H

#include <vector>
#include <iostream>
#include <cstddef>

#if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace hoomd_tf
    {

    /*! \file CommStruct.h
        \brief CommStruct and CommStructDerived declaration
    */


    /*!  CommStruct class
     *   This is a wrapper around a hoomd array that has
     *   convienence functions for getting dimensions, types,
     *   and copying
     *
     */

    struct CommStruct
        {
        CommStruct(int num_dims, size_t element_size, const char* name) :
        num_dims(num_dims),
        element_size(element_size),
        offset(0),
	mem_size(0),
        name(name)
        {}

        //! Set number of elements and calculate needed memory
        void setNumElements(int* num_elements_t)
            {
            size_t size = 1;
            num_elements = new int[num_dims];
            for(int i = 0; i < num_dims; i++)
                {
                num_elements[i] = num_elements_t[i];
                size *= num_elements[i];
                }
            mem_size = size * element_size;
            }

        CommStruct() {}

        //! Assignment operator overload
        CommStruct& operator=(const CommStruct& other)
            {
            num_elements = other.num_elements;
            num_dims = other.num_dims;
            element_size = other.element_size;
            mem_size = other.mem_size;
            name = other.name;
            offset = other.offset;
#if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
            stream = other.stream;
#endif

            return *this;
            }

        // I cannot figure out how to get operator
        // overloading printing to work for derived classes.
        std::ostream& print(std::ostream& os) const
            {
            os << name <<  ":\n  " << "Dims: [";
            for(int i = 0; i < num_dims; i++)
                {
                os << num_elements[i] << " ";
                }
            os << "]\n  "
               << "Element Size: "
               << element_size
               << "\n"
               << "Total Size: "
               << mem_size
               << "\n";
            return os;
            }
        virtual void readGPUMemory(void *dest, size_t n) = 0;        //! Read GPU memory
        virtual void readCPUMemory(void *dest, size_t n) = 0;        //! Read CPU memory
        virtual void writeGPUMemory(const void *src, size_t n) = 0;  //! Write to GPU
        virtual void writeCPUMemory(const void *src, size_t n) = 0;  //! Write to CPU

        int* num_elements;   //! Number of elements. would be better as size_t
                             //! but need this for TF
        int num_dims;        //! Dimensionality
        size_t element_size; //! Bit size of each element
        size_t offset;       //! Offset for doing batches
        size_t mem_size;     //! Total memory of all elements together
        const char* name;    //! Name of this communication object
        //TODO Why is ENABLE_CUDA set for compilng tf code? We don't have any hoomd headers...
#if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
        cudaStream_t stream = 0;  //! This CommStruct's CUDA stream
#endif
        };

    }

    /*!  CommStructDerived class
     *   This is for a derived child of CommStruct and has
     *   all its functionality
     *
     */

#ifndef GOOGLE_CUDA
#include <hoomd/GlobalArray.h>
namespace hoomd_tf
    {
    template <typename T>
        struct CommStructDerived : CommStruct
        {
        GlobalArray<T>* m_array;

        CommStructDerived(GlobalArray<T>& array, const char* name)
            {
            //! Disallow unspecialized construction with explicit array
            T::unimplemented_function;
            }

        CommStructDerived()
            {
            //! Only here for class variables that have CommStrucDerived
            }

        CommStructDerived& operator=(const CommStructDerived<T>& other)
            {
            m_array = other.m_array;
            CommStruct::operator=(other);
            return *this;
            }

#ifdef ENABLE_CUDA
        void readGPUMemory(void *dest, size_t n) override
            {
            assert(offset * sizeof(T) + n <= mem_size);
            ArrayHandle<T> handle(*m_array, access_location::device, access_mode::read);
            cudaMemcpy(dest, handle.data + offset, n, cudaMemcpyDeviceToDevice);
            }
        void writeGPUMemory(const void* src, size_t n) override
            {
            assert(offset * sizeof(T) + n <= mem_size);
            ArrayHandle<T> handle(*m_array, access_location::device, access_mode::overwrite);
            cudaMemcpy(handle.data + offset, src, n, cudaMemcpyDeviceToDevice);
            }
#else
        void readGPUMemory(void *dest, size_t n) override
            {
            throw "Should not call readGPUMemory without CUDA";
            }
        void writeGPUMemory(const void* src, size_t n) override
            {
            throw "Should not call readGPUMemory without CUDA";
            }
#endif //ENABLE_CUDA
        void readCPUMemory(void* dest, size_t n) override
            {
            assert(offset * sizeof(T) + n <= mem_size);
            ArrayHandle<T> handle(*m_array, access_location::host, access_mode::read);
            memcpy(dest, handle.data + offset, n);
            }
        void writeCPUMemory(const void* src, size_t n) override
            {
            assert(offset * sizeof(T) + n <= mem_size);
            ArrayHandle<T> handle(*m_array, access_location::host, access_mode::overwrite);
            memcpy(handle.data + offset, src, n);
            }
        };

    //! Forward declare specialized templates
    template<> CommStructDerived<Scalar4>::CommStructDerived(GlobalArray<Scalar4>&, const char*);
    template<> CommStructDerived<Scalar3>::CommStructDerived(GlobalArray<Scalar3>&, const char*);
    template<> CommStructDerived<Scalar>::CommStructDerived(GlobalArray<Scalar>&, const char*);
    }
#endif //GOOGLE_CUDA
#endif //guard
