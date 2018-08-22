// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// **********************
// This is a simple example code written for no function purpose other then to demonstrate the steps needed to write a
// c++ source code plugin for HOOMD-Blue. This example includes an example Compute class, but it can just as easily be
// replaced with a ForceCompute, Integrator, or any other C++ code at all.

// inclusion guard
#ifndef _TENSORFLOW_COMPUTE_H_
#define _TENSORFLOW_COMPUTE_H_

/*! \file TensorflowCompute.h
    \brief Declaration of TensorflowCompute
*/

#include "TensorflowCompute.h"
#include "IPCArrayComm.h"
#include "IPCTaskLock.h"
#include <hoomd/ForceCompute.h>
#include <hoomd/HOOMDMath.h>
#include <hoomd/ParticleData.h>
#include <hoomd/SystemDefinition.h>
#include <hoomd/md/NeighborList.h>
#include <hoomd/Autotuner.h>


// pybind11 is used to create the python bindings to the C++ object,
// but not if we are compiling GPU kernels
#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>
#include <hoomd/extern/pybind/include/pybind11/stl_bind.h>
#endif

// (if you really don't want to include the whole hoomd.h, you can include individual files IF AND ONLY IF
// hoomd_config.h is included first)
// For example:
// #include <hoomd/Compute.h>

// Second, we need to declare the class. One could just as easily use any class in HOOMD as a template here, there are
// no restrictions on what a template can do

//! A nonsense particle Compute written to demonstrate how to write a plugin
/*! This Compute simply sets all of the particle's velocities to 0 when update() is called.
*/

enum class FORCE_MODE{
    overwrite, add, ignore, output
};

IPCReservation* reserve_memory(unsigned int natoms, unsigned int nneighs);

//these functors use 'call' instead of 'operator()' to avoid
//writing out functor.template operator()<T> (...) which is
//necessary due to some arcane c++ rules. Normally
// you would write functor(...), when creating a functor.
struct receiveForcesFunctorAdd {

    size_t _N;
    void* _stream;
    receiveForcesFunctorAdd() {}
    receiveForcesFunctorAdd(size_t N) : _N(N), _stream(nullptr) {}

    //have empty implementation so if CUDA not enabled,
    //we still have a GPU implementation
    template<IPCCommMode M>
    void call(Scalar4* dest, Scalar4* src) {}

};

struct receiveVirialFunctorAdd {

    size_t _N;
    size_t _pitch;
    void* _stream;
    receiveVirialFunctorAdd(){}
    receiveVirialFunctorAdd(size_t N, size_t pitch) : _N(N), _pitch(pitch), _stream(nullptr) {}

    template<IPCCommMode M>
    void call(Scalar* dest, Scalar* src) {}
};

template <IPCCommMode M = IPCCommMode::CPU>
class TensorflowCompute : public ForceCompute
    {
    public:
        //! Constructor
        TensorflowCompute(pybind11::object& py_self, std::shared_ptr<SystemDefinition> sysdef,  std::shared_ptr<NeighborList> nlist,
             Scalar r_cut, unsigned int nneighs, FORCE_MODE force_mode,
             IPCReservation* ipc_reservation, IPCTaskLock* tasklock);

        TensorflowCompute() = delete;

        //!Destructor
        virtual ~TensorflowCompute();

        Scalar getLogValue(const std::string& quantity, unsigned int timestep) override;

        int64_t getForcesBuffer() const;
        int64_t getPositionsBuffer() const;
        int64_t getVirialBuffer() const;
        int64_t getNlistBuffer() const;

        bool isDoublePrecision() const {
            #ifdef SINGLE_PRECISION
            return false;
            #else
            return true;
            #endif //SINGLE_PRECISION
        }

        std::vector<Scalar4> getForcesArray() const;
        std::vector<Scalar4> getNlistArray() const;
        std::vector<Scalar4> getPositionsArray() const;
        std::vector<Scalar> getVirialArray() const;
        unsigned int getVirialPitch() const {return m_virial.getPitch();}

        pybind11::object _py_self; //pybind objects have to be public with current cc flags



    protected:

        //used if particle number changes
        virtual void reallocate();
        //! Take one timestep forward
        virtual void computeForces(unsigned int timestep) override;

        virtual void prepareNeighbors();
        virtual void zeroVirial();

        std::shared_ptr<NeighborList> m_nlist;
        Scalar _r_cut;
        unsigned int _nneighs;
        FORCE_MODE _force_mode;
        std::string m_log_name;
        IPCReservation* _ipcr;
        IPCTaskLock* _tasklock;

        IPCArrayComm<M, Scalar4> _positions_comm;
        IPCArrayComm<M, Scalar4> _forces_comm;
        GPUArray<Scalar4> _nlist_array;
        IPCArrayComm<M, Scalar4> _nlist_comm;
        IPCArrayComm<M, Scalar> _virial_comm;

        receiveVirialFunctorAdd _virial_functor;
        receiveForcesFunctorAdd _forces_functor;
    };

//! Export the TensorflowCompute class to python
void export_TensorflowCompute(pybind11::module& m);

// Third, this class offers a GPU accelerated method in order to demonstrate how to include CUDA code in pluins
// we need to declare a separate class for that (but only if ENABLE_CUDA is set)

#ifdef ENABLE_CUDA

//! A GPU accelerated nonsense particle Compute written to demonstrate how to write a plugin w/ CUDA code
/*! This Compute simply sets all of the particle's velocities to 0 (on the GPU) when update() is called.
*/
class TensorflowComputeGPU : public TensorflowCompute<IPCCommMode::GPU>
    {
    public:
        //! Constructor
        TensorflowComputeGPU(pybind11::object& py_self, std::shared_ptr<SystemDefinition> sysdef,  std::shared_ptr<NeighborList> nlist,
             Scalar r_cut, unsigned int nneighs,
             FORCE_MODE force_mode, IPCReservation* ipc_reservation,
             IPCTaskLock* tasklock);

        void setAutotunerParams(bool enable, unsigned int period) override;
    protected:
	void computeForces(unsigned int timestep) override;
        void reallocate() override;
        void prepareNeighbors() override;
        void zeroVirial() override;
    private:
        std::unique_ptr<Autotuner> m_tuner; // Autotuner for block size
	cudaStream_t _streams[4];
	size_t _nstreams = 4;
    };

//! Export the TensorflowComputeGPU class to python
void export_TensorflowComputeGPU(pybind11::module& m);

template class TensorflowCompute<IPCCommMode::GPU>;
#endif // ENABLE_CUDA

//force implementation
template class TensorflowCompute<IPCCommMode::CPU>;

#endif // _TENSORFLOW_COMPUTE_H_
