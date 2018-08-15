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
template<IPCCommMode M>
TensorflowCompute<M>::TensorflowCompute(
    pybind11::object& py_self,
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist,
    Scalar r_cut,
    unsigned int nneighs, FORCE_MODE force_mode)
        : ForceCompute(sysdef),
          _py_self(py_self),
          m_nlist(nlist),
          _r_cut(r_cut),
          _nneighs(nneighs),
          _force_mode(force_mode)
{

    reallocate();
    m_log_name = std::string("tensorflow");
    auto flags = this->m_pdata->getFlags();
    if(_force_mode == FORCE_MODE::overwrite || _force_mode == FORCE_MODE::add) {
        //flags[pdata_flag::isotropic_virial] = 1;
        flags[pdata_flag::pressure_tensor] = 1;
        m_exec_conf->msg->notice(2) <<"Setting flag indicating virial modification will occur" << std::endl;
    }
    // connect to the ParticleData to receive notifications when the maximum number of particles changes
     m_pdata->getMaxParticleNumberChangeSignal().connect<TensorflowCompute, &TensorflowCompute<M>::reallocate>(this);

}
template <IPCCommMode M>
void TensorflowCompute<M>::reallocate() {


    assert(m_pdata);

    //we won't ever override positions,
    //but the recieve method does exist
    //so we'll cast away until I make a version
    //of IPCArrayComm that can't override array
    _positions_comm = IPCArrayComm<M,Scalar4>(const_cast<GPUArray<Scalar4>& > (m_pdata->getPositions()));
    _forces_comm = IPCArrayComm<M,Scalar4>(m_force);
    GPUArray<Scalar4> tmp(_nneighs * m_pdata->getN(), m_exec_conf);
    _nlist_array.swap(tmp);
    _nlist_comm = IPCArrayComm<M,Scalar4>(_nlist_array);
    //pass a larger size because sparse matrix is used in HOOMD
    _virial_comm = IPCArrayComm<M,Scalar>(m_virial, m_pdata->getN() * 9);

    _py_self.attr("restart_tf")();
}


template<IPCCommMode M>
TensorflowCompute<M>::~TensorflowCompute() {

}

/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
template<IPCCommMode M>
void TensorflowCompute<M>::computeForces(unsigned int timestep) {
    if (m_prof) m_prof->push("TensorflowCompute");

    if (m_prof) m_prof->push("TensorflowCompute<M>::Acquire Lock");
    _py_self.attr("start_update")();
    if (m_prof) m_prof->pop();

    //nneighs == 0 send positions only
    _positions_comm.send();
    if(_nneighs > 0) {
        //Update the neighborlist
        m_nlist->compute(timestep);
        if (m_prof) m_prof->push("TensorflowCompute<M>::prepareNeighbors");
        prepareNeighbors();
        if (m_prof) m_prof->pop();
        _nlist_comm.send();
    }


    if (m_prof) m_prof->push("TensorflowCompute<M>::Acquire Barrier (TF Update)");
    _py_self.attr("finish_update")();
    if (m_prof) m_prof->pop();
    if (m_prof) m_prof->push("TensorflowCompute<M>::Force Update");

    switch(_force_mode) {
        //process results from TF
        case FORCE_MODE::overwrite:
            _forces_comm.receive();
            zeroVirial();
            _virial_comm.receiveOp(receiveVirialFunctorAdd(m_pdata->getN(), m_virial_pitch));
            break;
        case FORCE_MODE::add:
            _forces_comm.receiveOp(receiveForcesFunctorAdd(m_pdata->getN()));
            _virial_comm.receiveOp(receiveVirialFunctorAdd(m_pdata->getN(), m_virial_pitch));
            break;
        case FORCE_MODE::output:
            _forces_comm.send();
        case FORCE_MODE::ignore:
            break;
    }

    if (m_prof) m_prof->pop(); //force update
    if (m_prof) m_prof->pop(); //compute
}


template<IPCCommMode M>
void TensorflowCompute<M>::zeroVirial() {
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());
}

template<IPCCommMode M>
void TensorflowCompute<M>::prepareNeighbors() {

    //create ptr at offset to where neighbors go
    ArrayHandle<Scalar4> buffer_array(_nlist_array, access_location::host, access_mode::overwrite);
    Scalar4* buffer = buffer_array.data;
    unsigned int* nnoffset = (unsigned int*) calloc(m_pdata->getN(), sizeof(unsigned int));

    //These snippets taken from md/TablePotentials.cc

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
            if(dx.x * dx.x + dx.y * dx.y + dx.z * dx.z > _r_cut * _r_cut)
                continue;
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
}

template<IPCCommMode M>
Scalar TensorflowCompute<M>::getLogValue(const std::string& quantity, unsigned int timestep) {
    //not really sure why this has to be implemented by this class...
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        this->m_exec_conf->msg->error() << "tensorflow:" <<  quantity << " is not a valid log quantity"
                    << std::endl;
        throw std::runtime_error("Error getting log value");
        }
}


template<IPCCommMode M>
int64_t TensorflowCompute<M>::getForcesBuffer() const { return _forces_comm.getAddress();}
template<IPCCommMode M>
int64_t TensorflowCompute<M>::getPositionsBuffer() const {return _positions_comm.getAddress();}
template<IPCCommMode M>
int64_t TensorflowCompute<M>::getVirialBuffer() const {return _virial_comm.getAddress();}
template<IPCCommMode M>
int64_t TensorflowCompute<M>::getNlistBuffer() const {return _nlist_comm.getAddress();}

template<IPCCommMode M>
std::vector<Scalar4> TensorflowCompute<M>::getPositionsArray() const {return _positions_comm.getArray();}
template<IPCCommMode M>
std::vector<Scalar4> TensorflowCompute<M>::getNlistArray() const {return _nlist_comm.getArray();}
template<IPCCommMode M>
std::vector<Scalar4> TensorflowCompute<M>::getForcesArray() const {return _forces_comm.getArray();}
template<IPCCommMode M>
std::vector<Scalar> TensorflowCompute<M>::getVirialArray() const {return _virial_comm.getArray();}

/* Export the CPU Compute to be visible in the python module
 */
void export_TensorflowCompute(pybind11::module& m)
    {
    pybind11::class_<TensorflowCompute<IPCCommMode::CPU>, std::shared_ptr<TensorflowCompute<IPCCommMode::CPU> > >(m, "TensorflowCompute", pybind11::base<ForceCompute>())
        .def(pybind11::init< pybind11::object&, std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar, unsigned int, FORCE_MODE>())
        .def("getPositionsBuffer", &TensorflowCompute<IPCCommMode::CPU>::getPositionsBuffer, pybind11::return_value_policy::reference)
        .def("getNlistBuffer", &TensorflowCompute<IPCCommMode::CPU>::getNlistBuffer, pybind11::return_value_policy::reference)
        .def("getForcesBuffer", &TensorflowCompute<IPCCommMode::CPU>::getForcesBuffer, pybind11::return_value_policy::reference)
        .def("getVirialBuffer", &TensorflowCompute<IPCCommMode::CPU>::getVirialBuffer, pybind11::return_value_policy::reference)
        .def("getPositionsArray", &TensorflowCompute<IPCCommMode::CPU>::getPositionsArray, pybind11::return_value_policy::take_ownership)
        .def("getNlistArray", &TensorflowCompute<IPCCommMode::CPU>::getNlistArray, pybind11::return_value_policy::take_ownership)
        .def("getForcesArray", &TensorflowCompute<IPCCommMode::CPU>::getForcesArray, pybind11::return_value_policy::take_ownership)
        .def("getVirialArray", &TensorflowCompute<IPCCommMode::CPU>::getVirialArray, pybind11::return_value_policy::take_ownership)
        .def("isDoublePrecision", &TensorflowCompute<IPCCommMode::CPU>::isDoublePrecision)
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
TensorflowComputeGPU::TensorflowComputeGPU(pybind11::object& py_self,
            std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<NeighborList> nlist,
             Scalar r_cut, unsigned int nneighs,
             FORCE_MODE force_mode)
        : TensorflowCompute(py_self, sysdef, nlist, r_cut, nneighs, force_mode)
{

    _nneighs = std::min(this->m_nlist->getNListArray().getPitch(),nneighs);
    std::cout << "set nneighs to be " << _nneighs << " to match GPU nlist array pitch" << std::endl;
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "tensorflow", this->m_exec_conf));
}

void TensorflowComputeGPU::setAutotunerParams(bool enable, unsigned int period)
{
    TensorflowCompute::setAutotunerParams(enable, period);
    m_tuner->setPeriod(period);
    m_tuner->setEnabled(enable);
}

void TensorflowComputeGPU::prepareNeighbors() {

    ArrayHandle<Scalar4> d_nlist_array(this->_nlist_array, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_head_list(this->m_nlist->getHeadList(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    m_tuner->begin();
    gpu_reshape_nlist(d_nlist_array.data,
                      d_pos.data,
                      m_pdata->getN(),
                      _nneighs,
                      m_pdata->getNGhosts(),
                      m_pdata->getBox(),
                      d_n_neigh.data,
                      d_nlist.data,
                      d_head_list.data,
                      this->m_nlist->getNListArray().getPitch(),
                      m_tuner->getParam(),
                      m_exec_conf->getComputeCapability(),
                      m_exec_conf->dev_prop.maxTexture1DLinear,
                      _r_cut);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
}

void TensorflowComputeGPU::zeroVirial() {
    ArrayHandle<Scalar> h_virial(m_virial,access_location::device, access_mode::overwrite);
    cudaMemset(static_cast<void*> (h_virial.data), 0, sizeof(Scalar) * m_virial.getNumElements());
}

/* Export the GPU Compute to be visible in the python module
 */
void export_TensorflowComputeGPU(pybind11::module& m)
    {
    pybind11::class_<TensorflowComputeGPU, std::shared_ptr<TensorflowComputeGPU> >(m, "TensorflowComputeGPU", pybind11::base<ForceCompute>())
        .def(pybind11::init< pybind11::object&, std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar, unsigned int, FORCE_MODE>())
        .def("getPositionsBuffer", &TensorflowComputeGPU::getPositionsBuffer, pybind11::return_value_policy::reference)
        .def("getNlistBuffer", &TensorflowComputeGPU::getNlistBuffer, pybind11::return_value_policy::reference)
        .def("getForcesBuffer", &TensorflowComputeGPU::getForcesBuffer, pybind11::return_value_policy::reference)
        .def("getVirialBuffer", &TensorflowComputeGPU::getVirialBuffer, pybind11::return_value_policy::reference)
        .def("getPositionsArray", &TensorflowComputeGPU::getPositionsArray, pybind11::return_value_policy::take_ownership)
        .def("getNlistArray", &TensorflowComputeGPU::getNlistArray, pybind11::return_value_policy::take_ownership)
        .def("getForcesArray", &TensorflowComputeGPU::getForcesArray, pybind11::return_value_policy::take_ownership)
        .def("getVirialArray", &TensorflowComputeGPU::getVirialArray, pybind11::return_value_policy::take_ownership)
        .def("isDoublePrecision", &TensorflowComputeGPU::isDoublePrecision)
    ;
    }

#endif // ENABLE_CUDA


template<>
void receiveForcesFunctorAdd::call<IPCCommMode::CPU>(Scalar4* dest, Scalar4* src) {
    for(unsigned int i = 0; i < _N; i++) {
        dest[i].x += src[i].x;
        dest[i].y += src[i].y;
        dest[i].z += src[i].z;
        dest[i].w += src[i].w;
    }
}


template<>
void receiveVirialFunctorAdd::call<IPCCommMode::CPU>(Scalar* dest, Scalar* src) {
    for(unsigned int i = 0; i < _N; i++) {
        dest[0 * _pitch + i] += src[i * 9 + 0]; //xx
        dest[1 * _pitch + i] += src[i * 9 + 1]; //xy
        dest[2 * _pitch + i] += src[i * 9 + 2]; //xz
        dest[3 * _pitch + i] += src[i * 9 + 4]; //yy
        dest[4 * _pitch + i] += src[i * 9 + 5]; //yz
        dest[5 * _pitch + i] += src[i * 9 + 8]; //zz
    }
}

#ifdef ENABLE_CUDA
template<>
void receiveForcesFunctorAdd::call<IPCCommMode::GPU>(Scalar4* dest, Scalar4* src) {
    gpu_add_scalar4(dest, src, _N);
}


template<>
void receiveVirialFunctorAdd::call<IPCCommMode::GPU>(Scalar* dest, Scalar* src) {
    gpu_add_virial(dest, src, _N, _pitch);
}
#endif //ENABLE_CUDA