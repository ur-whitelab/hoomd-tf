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

#include <hoomd/ForceCompute.h>
#include <hoomd/HOOMDMath.h>
#include <hoomd/ParticleData.h>
#include <hoomd/SystemDefinition.h>
#include <hoomd/md/NeighborList.h>

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


class TensorflowCompute : public ForceCompute
    {
    public:
        //! Constructor
        TensorflowCompute(pybind11::object& py_self, std::shared_ptr<SystemDefinition> sysdef,  std::shared_ptr<NeighborList> nlist,
             Scalar r_cut, unsigned int nneighs, FORCE_MODE force_mode);

        TensorflowCompute() = delete;

        //!Destructor
        virtual ~TensorflowCompute();

        //used if particle number changes
        void reallocate();

        Scalar getLogValue(const std::string& quantity, unsigned int timestep) override;

        int64_t get_forces_buffer() const {
            if(_force_mode == FORCE_MODE::output)
                //if forces are being output, get their location
                return reinterpret_cast<int64_t> (_output_buffer + m_pdata->getN() * (1 + _nneighs));
            return reinterpret_cast<int64_t> (_input_buffer);
        }

        int64_t get_positions_buffer() const {return reinterpret_cast<int64_t> (_output_buffer);}
        int64_t get_virial_buffer() const {return reinterpret_cast<int64_t> (_input_buffer + m_pdata->getN());}
        int64_t get_nlist_buffer() const {return reinterpret_cast<int64_t> (_output_buffer + m_pdata->getN());}

        bool is_double_precision() const {
            #ifdef SINGLE_PRECISION
            return false;
            #else
            return true;
            #endif //SINGLE_PRECISION
        }

        std::vector<Scalar4> get_forces_array() const;
        std::vector<Scalar4> get_nlist_array() const;
        std::vector<Scalar4> get_positions_array() const;

        pybind11::object _py_self; //pybind objects have to be public with current cc flags



    protected:
        //! Take one timestep forward
        void computeForces(unsigned int timestep) override;

        void sendPositions();
        void sendNeighbors(unsigned int timestep);
        void receiveVirial();

        std::shared_ptr<NeighborList> _m_nlist;
        Scalar4* _input_buffer;
        Scalar4* _output_buffer;
        size_t _buffer_size;
        size_t _virial_size;
        Scalar _r_cut;
        unsigned int _nneighs;
        FORCE_MODE _force_mode;
        std::string m_log_name;
    };

//! Export the TensorflowCompute class to python
void export_TensorflowCompute(pybind11::module& m);

// Third, this class offers a GPU accelerated method in order to demonstrate how to include CUDA code in pluins
// we need to declare a separate class for that (but only if ENABLE_CUDA is set)

#ifdef ENABLE_CUDA

//! A GPU accelerated nonsense particle Compute written to demonstrate how to write a plugin w/ CUDA code
/*! This Compute simply sets all of the particle's velocities to 0 (on the GPU) when update() is called.
*/
class TensorflowComputeGPU : public TensorflowCompute
    {
    public:
        //! Constructor
        TensorflowComputeGPU(std::shared_ptr<SystemDefinition> sysdef, pybind11::object py_self);

        //! Take one timestep forward
        virtual void update(unsigned int timestep);
    };

//! Export the TensorflowComputeGPU class to python
void export_TensorflowComputeGPU(pybind11::module& m);

#endif // ENABLE_CUDA

#endif // _TENSORFLOW_COMPUTE_H_
