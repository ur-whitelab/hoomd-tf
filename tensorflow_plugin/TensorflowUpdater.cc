// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TensorflowUpdater.h"
#ifdef ENABLE_CUDA
#include "TensorflowUpdater.cuh"
#endif

#include <iostream>
#include <sys/mman.h>

/*! \file TensorflowUpdater.cc
    \brief Definition of TensorflowUpdater
*/

// ********************************
// here follows the code for TensorflowUpdater on the CPU

/*! \param sysdef System to zero the velocities of
*/
TensorflowUpdater::TensorflowUpdater(std::shared_ptr<SystemDefinition> sysdef, pybind11::object& py_self)
        : Updater(sysdef), _py_self(py_self)
{
    // might need to do something so GPU code doesn't call this
    auto m_exec_conf = sysdef->getParticleData()->getExecConf();
    // create input/output mmap buffer
    assert(m_pdata);
    _input_buffer = static_cast<Scalar4*> (mmap(NULL, m_pdata->getN()*sizeof(Scalar4), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0));
    _output_buffer = static_cast<Scalar4*> (mmap(NULL, m_pdata->getN()*sizeof(Scalar4), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0));
    if(_input_buffer == MAP_FAILED || _output_buffer == MAP_FAILED) {
        perror("Failed to create mmap");
        m_exec_conf->msg->error() << "Failed to create mmap" << std::endl;
    }
    m_exec_conf->msg->notice(2) << "Created mmaped pages for tensorflow updater (" << m_pdata->getN()*sizeof(Scalar4) / 1024.0 << " kB)" << std::endl;
}

TensorflowUpdater::~TensorflowUpdater() {
    // unmap our mmapings
    assert(m_pdata);
    munmap(_input_buffer, m_pdata->getN()*sizeof(Scalar4));
    munmap(_output_buffer, m_pdata->getN()*sizeof(Scalar4));
    _input_buffer = NULL;
    _output_buffer = NULL;
}

/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void TensorflowUpdater::update(unsigned int timestep)
{
    if (m_prof) m_prof->push("TensorflowUpdater");

    _py_self.attr("start_update")();

    // access the particle data for writing on the CPU
    assert(m_pdata);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);

    // zero the velocity of every particle
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_vel.data[i].x = Scalar(0.0);
        h_vel.data[i].y = Scalar(0.0);
        h_vel.data[i].z = Scalar(0.0);
        }


    _py_self.attr("finish_update")();
    if (m_prof) m_prof->pop();
}

/* Export the CPU updater to be visible in the python module
 */
void export_TensorflowUpdater(pybind11::module& m)
    {
    pybind11::class_<TensorflowUpdater, std::shared_ptr<TensorflowUpdater> >(m, "TensorflowUpdater", pybind11::base<Updater>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, pybind11::object& >())
        .def("get_input_buffer", &TensorflowUpdater::get_input_buffer, pybind11::return_value_policy::reference)
        .def("get_output_buffer", &TensorflowUpdater::get_input_buffer, pybind11::return_value_policy::reference)
    ;
    }

// ********************************
// here follows the code for TensorflowUpdater on the GPU

#ifdef ENABLE_CUDA

/*! \param sysdef System to zero the velocities of
*/
TensorflowUpdaterGPU::TensorflowUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef, pybind11::object py_self)
        : TensorflowUpdater(sysdef, py_self)
    {
    }


/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void TensorflowUpdaterGPU::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("TensorflowUpdater");

    // access the particle data arrays for writing on the GPU
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);

    // call the kernel devined in TensorflowUpdater.cu
    gpu_zero_velocities(d_vel.data, m_pdata->getN());

    // check for error codes from the GPU if error checking is enabled
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof) m_prof->pop();
    }

/* Export the GPU updater to be visible in the python module
 */
void export_TensorflowUpdaterGPU(pybind11::module& m)
    {
    pybind11::class_<TensorflowUpdaterGPU, std::shared_ptr<TensorflowUpdaterGPU> >(m, "TensorflowUpdaterGPU", pybind11::base<TensorflowUpdater>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, pybind11::object >())
        .def("get_input_buffer", &TensorflowUpdater::get_input_buffer, pybind11::return_value_policy::reference)
        .def("get_output_buffer", &TensorflowUpdater::get_output_buffer, pybind11::return_value_policy::reference)
    ;
    }

#endif // ENABLE_CUDA
