// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TensorflowCompute.h"
#ifdef ENABLE_CUDA
#include "TensorflowCompute.cuh"
#endif

#include <iostream>
#include <string.h>
#include <sys/mman.h>

/*! \file TensorflowCompute.cc
    \brief Definition of TensorflowCompute
*/

// ********************************
// here follows the code for TensorflowCompute on the CPU

/*! \param sysdef System to zero the velocities of
*/
TensorflowCompute::TensorflowCompute(
    pybind11::object& py_self,
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist,
    unsigned int nneighs, FORCE_MODE force_mode)
        : ForceCompute(sysdef),
          m_nlist(nlist),
          _py_self(py_self),
          _input_buffer(NULL),
          _output_buffer(NULL),
          _nneighs(nneighs),
          _force_mode(force_mode)
{

    reallocate();
    // connect to the ParticleData to receive notifications when the maximum number of particles changes
     m_pdata->getMaxParticleNumberChangeSignal().connect<TensorflowCompute, &TensorflowCompute::reallocate>(this);

}

void TensorflowCompute::reallocate() {
     // might need to do something so GPU code doesn't call this
    auto m_exec_conf = m_sysdef->getParticleData()->getExecConf();
    // create input/output mmap buffer
    assert(m_pdata);
    //check if allocated
    if(_input_buffer)
        munmap(_input_buffer, _buffer_size*sizeof(Scalar4));
    if(_output_buffer)
        munmap(_output_buffer, _buffer_size*sizeof(Scalar4));
    //set new size
    //             positions         neighbors
    _buffer_size = m_pdata->getN() + m_pdata->getN() * _nneighs;
    if(_force_mode == FORCE_MODE::output) {
        //add space for forces
        _buffer_size += m_pdata->getN();
    }

    //allocate space for virial by making sure buffer is big enough to accomodate.
    //virial is a 3x3 matrix per particle but elements are scalar4s.
    //hoomd internally uses 6 scalars per particle
    _virial_size = m_pdata->getN() * (3 * 3 / 4 + 1);
    _buffer_size = std::max(_buffer_size, m_pdata->getN() + _virial_size);

    _input_buffer = static_cast<Scalar4*> (mmap(NULL, _buffer_size*sizeof(Scalar4), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0));
    _output_buffer = static_cast<Scalar4*> (mmap(NULL, _buffer_size*sizeof(Scalar4), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0));
    if(_input_buffer == MAP_FAILED || _output_buffer == MAP_FAILED) {
        perror("Failed to create mmap");
        m_exec_conf->msg->error() << "Failed to create mmap" << std::endl;
    }
    m_exec_conf->msg->notice(2) << "Created mmaped pages for tensorflow Compute (" << _buffer_size*sizeof(Scalar4) / 1024.0 << " kB)" << std::endl;
    m_exec_conf->msg->notice(2) << "At addresses " << _input_buffer << "," << _output_buffer << std::endl;

    _py_self.attr("restart_tf")();
}

TensorflowCompute::~TensorflowCompute() {
    // unmap our mmapings
    if(_input_buffer) {
        munmap(_input_buffer, _buffer_size*sizeof(Scalar4));
        munmap(_output_buffer, _buffer_size*sizeof(Scalar4));
    }
    _input_buffer = NULL;
    _output_buffer = NULL;
}

/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void TensorflowCompute::computeForces(unsigned int timestep)
{
    if (m_prof) m_prof->push("TensorflowCompute");

    if (m_prof) m_prof->push("TensorflowCompute::Acquire Lock");
    _py_self.attr("start_update")();
    if (m_prof) m_prof->pop();

    //nneighs == 0 send positions only
    sendPositions();
    if(_nneighs > 0)
        sendNeighbors(timestep);

    //TODO: Handle virial (See TablePotential.cc?)
    ArrayHandle<Scalar4> h_force(m_force, access_location::host);
    if(_force_mode == FORCE_MODE::output) //output the forces, instead of using them from input buffer
        memcpy(_output_buffer, h_force.data, sizeof(Scalar4) * m_pdata->getN());

    if (m_prof) m_prof->push("TensorflowCompute::Acquire Barrier (TF Update)");
    _py_self.attr("finish_update")();
    if (m_prof) m_prof->pop();

    //process results from TF
    if (m_prof) m_prof->push("TensorflowCompute::Force Update");
    switch(_force_mode) {
        case FORCE_MODE::overwrite:
            memcpy(h_force.data, _input_buffer, sizeof(Scalar4) * m_pdata->getN());
            break;
        case FORCE_MODE::add:
            for(unsigned int i = 0; i < m_pdata->getN(); i++) {
                h_force.data[i].x += _input_buffer[i].x;
                h_force.data[i].y += _input_buffer[i].y;
                h_force.data[i].z += _input_buffer[i].z;
                //w is potential energy (?) TODO: learn more about this
                h_force.data[i].w += _input_buffer[i].w;

            }
            break;
        case FORCE_MODE::output: //already done above
        case FORCE_MODE::ignore:
            break;
    }

    if (m_prof) m_prof->pop(); //force update
    if (m_prof) m_prof->pop(); //compute
}

void TensorflowCompute::receiveVirial() {
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    Scalar* virial_buffer = reinterpret_cast<Scalar*> (_input_buffer + m_pdata->getN());
    for(unsigned int i = 0; i < m_pdata->getN(); i++) {
        h_virial.data[0*m_virial_pitch + i * 6] += virial_buffer[i * 9 + 0]; //xx
        h_virial.data[1*m_virial_pitch + i * 6] += virial_buffer[i * 9 + 1]; //xy
        h_virial.data[2*m_virial_pitch + i * 6] += virial_buffer[i * 9 + 2]; //xz
        h_virial.data[3*m_virial_pitch + i * 6] += virial_buffer[i * 9 + 4]; //yy
        h_virial.data[4*m_virial_pitch + i * 6] += virial_buffer[i * 9 + 5]; //yz
        h_virial.data[5*m_virial_pitch + i * 6] += virial_buffer[i * 9 + 8]; //zz
    }
}

void TensorflowCompute::sendPositions() {
    // access the particle data for writing on the CPU
    assert(m_pdata);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    //send data to buffer
    memcpy(_output_buffer, h_pos.data, sizeof(Scalar4) * m_pdata->getN());
}

void TensorflowCompute::sendNeighbors(unsigned int timestep) {

    //create ptr at offset to where neighbors go
    Scalar4* buffer = _output_buffer + m_pdata->getN();
    unsigned int* nnoffset = (unsigned int*) calloc(m_pdata->getN(), sizeof(unsigned int));

    //These snippets taken from md/TablePotentials.cc

    // start by updating the neighborlist
    m_nlist->compute(timestep);

    if (m_prof) m_prof->push("TensorflowCompute::sendNeighbors");

    // access the neighbor list
     ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    //need for periodic image correction
    const BoxDim& box = m_pdata->getBox();

    // for each particle
    for (int i = 0; i < (int) m_pdata->getN(); i++) {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        const unsigned int head_i = h_head_list.data[i];

        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        unsigned int j = 0;
        if(_nneighs < size)
            m_exec_conf->msg->error() << "Overflow in nlist!" << std::endl;
        for (; j < std::min(_nneighs, size); j++) {
            // access the index of this neighbor
            unsigned int k = h_nlist.data[head_i + j];

            // calculate dr
            Scalar3 pk = make_scalar3(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
            Scalar3 dx = pk - pi;

            // apply periodic boundary conditions
            dx = box.minImage(dx);
            buffer[i * _nneighs + nnoffset[i]].x = dx.x;
            buffer[i * _nneighs + nnoffset[i]].y = dx.y;
            buffer[i * _nneighs + nnoffset[i]].z = dx.z;
            buffer[i * _nneighs + nnoffset[i]].w = h_pos.data[i].w;
            nnoffset[i]++;

            if(m_nlist->getStorageMode() == NeighborList::half) {
                buffer[k * _nneighs + nnoffset[k]].x =  -dx.x;
                buffer[k * _nneighs + nnoffset[k]].y =  -dx.y;
                buffer[k * _nneighs + nnoffset[k]].z =  -dx.z;
                buffer[k * _nneighs + nnoffset[k]].w =  h_pos.data[k].w;
                nnoffset[k]++;
            }
        }
    }

    for (int i = 0; i < (int) m_pdata->getN(); i++) {
        // fill missing entries
        for (; nnoffset[i] < _nneighs; nnoffset[i]++) {
            buffer[i * _nneighs + nnoffset[i]].x = 0;
            buffer[i * _nneighs + nnoffset[i]].y = 0;
            buffer[i * _nneighs + nnoffset[i]].z = 0;
            buffer[i * _nneighs + nnoffset[i]].w = 0;
        }
    }

    free(nnoffset);

    if (m_prof) m_prof->pop();
}

std::vector<Scalar4> TensorflowCompute::get_positions_array() const {
    std::vector<Scalar4> array(_output_buffer, _output_buffer + m_pdata->getN());
    return array;
}

std::vector<Scalar4> TensorflowCompute::get_nlist_array() const {
    std::vector<Scalar4> array(_output_buffer + m_pdata->getN(), _output_buffer + _buffer_size);
    return array;
}

std::vector<Scalar4> TensorflowCompute::get_forces_array() const {
    std::vector<Scalar4> array(_input_buffer, _input_buffer + _buffer_size);
    return array;
}

/* Export the CPU Compute to be visible in the python module
 */
void export_TensorflowCompute(pybind11::module& m)
    {
    pybind11::class_<TensorflowCompute, std::shared_ptr<TensorflowCompute> >(m, "TensorflowCompute", pybind11::base<ForceCompute>())
        .def(pybind11::init< pybind11::object&, std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, unsigned int, FORCE_MODE>())
        .def("get_positions_buffer", &TensorflowCompute::get_positions_buffer, pybind11::return_value_policy::reference)
        .def("get_nlist_buffer", &TensorflowCompute::get_nlist_buffer, pybind11::return_value_policy::reference)
        .def("get_forces_buffer", &TensorflowCompute::get_forces_buffer, pybind11::return_value_policy::reference)
        .def("get_virial_buffer", &TensorflowCompute::get_virial_buffer, pybind11::return_value_policy::reference)
        .def("get_positions_array", &TensorflowCompute::get_positions_array, pybind11::return_value_policy::take_ownership)
        .def("get_nlist_array", &TensorflowCompute::get_nlist_array, pybind11::return_value_policy::take_ownership)
        .def("get_forces_array", &TensorflowCompute::get_forces_array, pybind11::return_value_policy::take_ownership)
        .def("is_double_precision", &TensorflowCompute::is_double_precision)
    ;
    pybind11::enum_<FORCE_MODE>(m, "FORCE_MODE")
        .value("overwrite", FORCE_MODE::overwrite)
        .value("output", FORCE_MODE::output)
        .value("add", FORCE_MODE::add)
        .value("ignore", FORCE_MODE::ignore)
    ;
    }

// ********************************
// here follows the code for TensorflowCompute on the GPU

#ifdef ENABLE_CUDA

/*! \param sysdef System to zero the velocities of
*/
TensorflowComputeGPU::TensorflowComputeGPU(std::shared_ptr<SystemDefinition> sysdef, pybind11::object py_self)
        : TensorflowCompute(sysdef, py_self)
    {
    }


/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void TensorflowComputeGPU::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("TensorflowCompute");

    // access the particle data arrays for writing on the GPU
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);

    // call the kernel devined in TensorflowCompute.cu
    gpu_zero_velocities(d_vel.data, _buffer_size);

    // check for error codes from the GPU if error checking is enabled
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof) m_prof->pop();
    }

/* Export the GPU Compute to be visible in the python module
 */
void export_TensorflowComputeGPU(pybind11::module& m)
    {
    pybind11::class_<TensorflowComputeGPU, std::shared_ptr<TensorflowComputeGPU> >(m, "TensorflowComputeGPU", pybind11::base<TensorflowCompute>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, pybind11::object &>())
        .def("get_input_buffer", &TensorflowCompute::get_input_buffer)
        .def("get_output_buffer", &TensorflowCompute::get_output_buffer)
    ;
    }

#endif // ENABLE_CUDA
