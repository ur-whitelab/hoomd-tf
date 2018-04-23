// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TensorflowUpdater.h"
#ifdef ENABLE_CUDA
#include "TensorflowUpdater.cuh"
#endif

#include <iostream>

/*! \file TensorflowUpdater.cc
    \brief Definition of TensorflowUpdater
*/

// ********************************
// here follows the code for TensorflowUpdater on the CPU

/*! \param sysdef System to zero the velocities of
*/
TensorflowUpdater::TensorflowUpdater(std::shared_ptr<SystemDefinition> sysdef)
        : Updater(sysdef)
    {
        Session* session;
        auto m_exec_conf = sysdef->getParticleData()->getExecConf();
        Status status = NewSession(SessionOptions(), &session);
        if (!status.ok()) {
             m_exec_conf->msg->notice(5) << "Able to load TF Session" << std::endl;
        } else {
            m_exec_conf->msg->error() << "Failed to load TF Session!" << std::endl;
        }

    }


/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void TensorflowUpdater::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("TensorflowUpdater");

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

    if (m_prof) m_prof->pop();
    }

/* Export the CPU updater to be visible in the python module
 */
void export_TensorflowUpdater(pybind11::module& m)
    {
    pybind11::class_<TensorflowUpdater, std::shared_ptr<TensorflowUpdater> >(m, "TensorflowUpdater", pybind11::base<Updater>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition> >())
    ;
    }

// ********************************
// here follows the code for TensorflowUpdater on the GPU

#ifdef ENABLE_CUDA

/*! \param sysdef System to zero the velocities of
*/
TensorflowUpdaterGPU::TensorflowUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef)
        : TensorflowUpdater(sysdef)
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
        .def(pybind11::init<std::shared_ptr<SystemDefinition> >())
    ;
    }

#endif // ENABLE_CUDA
