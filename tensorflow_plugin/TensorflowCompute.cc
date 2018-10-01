// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause
// License.

#include "TensorflowCompute.h"
#ifdef ENABLE_CUDA
#include "TensorflowCompute.cuh"
#endif

#include <string.h>
#include <sys/mman.h>
#include <iostream>

/*! \file TensorflowCompute.cc
    \brief Definition of TensorflowCompute
*/
// ********************************
// here follows the code for TensorflowCompute on the CPU

/*! \param sysdef System to zero the velocities of
 */
template <IPCCommMode M>
TensorflowCompute<M>::TensorflowCompute(
    pybind11::object& py_self, std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist, Scalar r_cut, unsigned int nneighs,
    FORCE_MODE force_mode, unsigned int period, IPCReservation* ipc_reservation,
    IPCTaskLock* tasklock)
    : ForceCompute(sysdef),
      _py_self(py_self),
      m_nlist(nlist),
      _r_cut(r_cut),
      _nneighs(nneighs),
      _force_mode(force_mode),
      _period(period),
      _ipcr(ipc_reservation),
      _tasklock(tasklock) {
  m_exec_conf->msg->notice(2)
      << "Starting TensorflowCompute with IPC Memory reservation of "
      << _ipcr->_size << " bytes" << std::endl;
  reallocate();
  m_exec_conf->msg->notice(2) << "completed reallocate" << std::endl;
  m_log_name = std::string("tensorflow");
  auto flags = this->m_pdata->getFlags();
  if (_force_mode == FORCE_MODE::overwrite) {
    // flags[pdata_flag::isotropic_virial] = 1;
    flags[pdata_flag::pressure_tensor] = 1;
    m_exec_conf->msg->notice(2)
        << "Setting flag indicating virial modification will occur"
        << std::endl;
  }
  // connect to the ParticleData to receive notifications when the maximum
  // number of particles changes
  m_pdata->getMaxParticleNumberChangeSignal()
      .connect<TensorflowCompute, &TensorflowCompute<M>::reallocate>(this);
}
template <IPCCommMode M>
void TensorflowCompute<M>::reallocate() {
  assert(m_pdata);

  // we won't ever override positions,
  // but the recieve method does exist
  // so we'll cast away until I make a version
  // of IPCArrayComm that can't override array
  _positions_comm = IPCArrayComm<M, Scalar4>(
      const_cast<GPUArray<Scalar4>&>(m_pdata->getPositions()), _ipcr);
  _forces_comm = IPCArrayComm<M, Scalar4>(m_force, _ipcr);
  GPUArray<Scalar4> tmp(_nneighs * m_pdata->getN(), m_exec_conf);
  _nlist_array.swap(tmp);
  _nlist_comm = IPCArrayComm<M, Scalar4>(_nlist_array, _ipcr);
  CHECK_CUDA_ERROR();
  // virial is made with maxN, not N
  _virial_comm =
      IPCArrayComm<M, Scalar>(m_virial, _ipcr, (size_t)m_pdata->getMaxN() * 9);
  CHECK_CUDA_ERROR();

  // build functors
  _virial_functor = ReceiveVirialFunctorAdd(m_pdata->getN(), m_virial_pitch);
  _forces_functor = ReceiveForcesFunctorAdd(m_pdata->getN());
}

template <IPCCommMode M>
TensorflowCompute<M>::~TensorflowCompute() {
  delete _ipcr;
  delete _tasklock;
}

/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
template <IPCCommMode M>
void TensorflowCompute<M>::computeForces(unsigned int timestep) {

  // We need to offset positions and net forces if we're sending
  // forces because net forces are calculated after all computes (like us),
  // so we don't have access until next step.
  if (timestep % _period != 0) return;
  if (m_prof) m_prof->push("TensorflowCompute");

  // send net forces from last step
  if(_force_mode == FORCE_MODE::output) {
      if(timestep > 0) {
        //get last step's net force and send
        _forces_comm.receiveArray(m_pdata->getNetForce());
        _forces_comm.sendAsync();
        finishUpdate(timestep);
        //we sent it using forces_comm. We need to zero it out
        _forces_comm.memsetArray(0);
      }
  }

  if (m_prof) m_prof->push("TensorflowCompute<M>::Preparing Data for TF");
  // nneighs == 0 send positions only
  _positions_comm.sendAsync();
  if (_nneighs > 0) {
    // Update the neighborlist
    m_nlist->compute(timestep);
    if (m_prof) m_prof->push("TensorflowCompute<M>::reshapeNeighbors");
    prepareNeighbors();
    if (m_prof) m_prof->pop();
    _nlist_comm.sendAsync();
  }

  if (m_prof) m_prof->pop(); //prearing data

  if (_force_mode != FORCE_MODE::output)
    finishUpdate(timestep);

  if (m_prof) m_prof->push("TensorflowCompute<M>::Force Update");

  switch (_force_mode) {
    // process results from TF
    case FORCE_MODE::overwrite:
      _forces_comm.receiveAsync();
      zeroVirial();
      _virial_comm.receiveOp(_virial_functor);
      break;
    case FORCE_MODE::output:
      break;
    case FORCE_MODE::ignore:
      break;
  }

  if (m_prof) m_prof->pop();  // force update
  if (m_prof) m_prof->pop();  // compute
}

template <IPCCommMode M>
void TensorflowCompute<M>::finishUpdate(unsigned int timestep) {
  if (m_prof) m_prof->push("TensorflowCompute<M>::Awaiting TF Update");
  _py_self.attr("finish_update")(timestep);
  //_tasklock->await();
  if (m_prof) m_prof->pop();
}

template <IPCCommMode M>
void TensorflowCompute<M>::zeroVirial() {
  ArrayHandle<Scalar> h_virial(m_virial, access_location::host,
                               access_mode::overwrite);
  memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());
}

template <IPCCommMode M>
void TensorflowCompute<M>::prepareNeighbors() {
  // create ptr at offset to where neighbors go
  ArrayHandle<Scalar4> buffer_array(_nlist_array, access_location::host,
                                    access_mode::overwrite);
  Scalar4* buffer = buffer_array.data;
  //zero out buffer
  memset(buffer, 0, _nneighs * m_pdata->getN() * sizeof(Scalar4));
  unsigned int* nnoffset =
      (unsigned int*)calloc(m_pdata->getN(), sizeof(unsigned int));

  // These snippets taken from md/TablePotentials.cc

  // access the neighbor list
  ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host,
                             access_mode::read);
  ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                      access_location::host, access_mode::read);
  ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                    access_location::host, access_mode::read);
  ArrayHandle<unsigned int> h_head_list(
      m_nlist->getHeadList(), access_location::host, access_mode::read);

  // need for periodic image correction
  const BoxDim& box = m_pdata->getBox();

  // for each particle
  for (int i = 0; i < (int)m_pdata->getN(); i++) {
    // access the particle's position and type (MEM TRANSFER: 4 scalars)
    Scalar3 pi =
        make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
    const unsigned int head_i = h_head_list.data[i];

    // loop over all of the neighbors of this particle
    const unsigned int size = (unsigned int)h_n_neigh.data[i];
    unsigned int j = 0;
    if (_nneighs < size)
      m_exec_conf->msg->error() << "Overflow in nlist!" << std::endl;
    for (; j < std::min(_nneighs, size); j++) {
      // access the index of this neighbor
      unsigned int k = h_nlist.data[head_i + j];

      // calculate dr
      Scalar3 pk =
          make_scalar3(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
      Scalar3 dx = pk - pi;

      // apply periodic boundary conditions
      dx = box.minImage(dx);
      if (dx.x * dx.x + dx.y * dx.y + dx.z * dx.z > _r_cut * _r_cut) continue;
      buffer[i * _nneighs + nnoffset[i]].x = dx.x;
      buffer[i * _nneighs + nnoffset[i]].y = dx.y;
      buffer[i * _nneighs + nnoffset[i]].z = dx.z;
      buffer[i * _nneighs + nnoffset[i]].w = h_pos.data[i].w;
      nnoffset[i]++;

      if (m_nlist->getStorageMode() == NeighborList::half) {
        buffer[k * _nneighs + nnoffset[k]].x = -dx.x;
        buffer[k * _nneighs + nnoffset[k]].y = -dx.y;
        buffer[k * _nneighs + nnoffset[k]].z = -dx.z;
        buffer[k * _nneighs + nnoffset[k]].w = h_pos.data[k].w;
        nnoffset[k]++;
      }
    }
  }

  free(nnoffset);
}

template <IPCCommMode M>
Scalar TensorflowCompute<M>::getLogValue(const std::string& quantity,
                                         unsigned int timestep) {
  // not really sure why this has to be implemented by this class...
  if (quantity == m_log_name) {
    compute(timestep);
    return calcEnergySum();
  } else {
    this->m_exec_conf->msg->error()
        << "tensorflow:" << quantity << " is not a valid log quantity"
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
        .def(pybind11::init< pybind11::object&, std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar, unsigned int, FORCE_MODE, unsigned int,  IPCReservation*, IPCTaskLock*>())
        .def("getPositionsBuffer", &TensorflowCompute<IPCCommMode::CPU>::getPositionsBuffer, pybind11::return_value_policy::reference)
        .def("getNlistBuffer", &TensorflowCompute<IPCCommMode::CPU>::getNlistBuffer, pybind11::return_value_policy::reference)
        .def("getForcesBuffer", &TensorflowCompute<IPCCommMode::CPU>::getForcesBuffer, pybind11::return_value_policy::reference)
        .def("getVirialBuffer", &TensorflowCompute<IPCCommMode::CPU>::getVirialBuffer, pybind11::return_value_policy::reference)
        .def("getPositionsArray", &TensorflowCompute<IPCCommMode::CPU>::getPositionsArray, pybind11::return_value_policy::take_ownership)
        .def("getNlistArray", &TensorflowCompute<IPCCommMode::CPU>::getNlistArray, pybind11::return_value_policy::take_ownership)
        .def("getForcesArray", &TensorflowCompute<IPCCommMode::CPU>::getForcesArray, pybind11::return_value_policy::take_ownership)
        .def("getVirialArray", &TensorflowCompute<IPCCommMode::CPU>::getVirialArray, pybind11::return_value_policy::take_ownership)
        .def("isDoublePrecision", &TensorflowCompute<IPCCommMode::CPU>::isDoublePrecision)
        .def("getVirialPitch", &TensorflowCompute<IPCCommMode::CPU>::getVirialPitch)
    ;
    pybind11::enum_<FORCE_MODE>(m, "FORCE_MODE")
        .value("overwrite", FORCE_MODE::overwrite)
        .value("output", FORCE_MODE::output)
        .value("ignore", FORCE_MODE::ignore)
    ;

    m.def("reserve_memory", &reserve_memory);
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
             FORCE_MODE force_mode, unsigned int period,
             IPCReservation* ipc_reservation, IPCTaskLock* tasklock)
     : TensorflowCompute(py_self, sysdef, nlist, r_cut, nneighs, force_mode, period, ipc_reservation, tasklock)
{

    _nneighs = std::min(m_nlist->getNListArray().getPitch(),nneighs);
    if(_nneighs != nneighs) {
     m_exec_conf->msg->notice(2) << "set nneighs to be " << _nneighs << " to match GPU nlist array pitch" << std::endl;
      reallocate();
    }
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "tensorflow", m_exec_conf));

    //want nlist on stream 0 since a nlist rebuild is
    //called just before prepareNeighbors
    _streams[0] = 0;
    for(unsigned int i = 1; i < _nstreams; i++) {
      cudaStreamCreate(&(_streams[i]));
      CHECK_CUDA_ERROR();
    }
}

void TensorflowComputeGPU::reallocate()  {
  TensorflowCompute::reallocate();
  _nlist_comm.setCudaStream(_streams[0]);
  _virial_comm.setCudaStream(_streams[1]);
  _forces_comm.setCudaStream(_streams[2]);

}

void TensorflowComputeGPU::computeForces(unsigned int timestep)  {
  TensorflowCompute::computeForces(timestep);
  cudaDeviceSynchronize();
}

void TensorflowComputeGPU::setAutotunerParams(bool enable, unsigned int period)
{
    TensorflowCompute::setAutotunerParams(enable, period);
    m_tuner->setPeriod(period);
    m_tuner->setEnabled(enable);
}

void TensorflowComputeGPU::prepareNeighbors() {

    ArrayHandle<Scalar4> d_nlist_array(_nlist_array, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(m_nlist->getNListArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_head_list(m_nlist->getHeadList(), access_location::device, access_mode::read);
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
                      _r_cut,
		      _nlist_comm.getCudaStream());

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
}

void TensorflowComputeGPU::zeroVirial() {
    ArrayHandle<Scalar> h_virial(m_virial,access_location::device, access_mode::overwrite);
    cudaMemsetAsync(static_cast<void*> (h_virial.data), 0, sizeof(Scalar) * m_virial.getNumElements(),_virial_comm.getCudaStream());
}

/* Export the GPU Compute to be visible in the python module
 */
void export_TensorflowComputeGPU(pybind11::module& m)
    {
    pybind11::class_<TensorflowComputeGPU, std::shared_ptr<TensorflowComputeGPU> >(m, "TensorflowComputeGPU", pybind11::base<ForceCompute>())
        .def(pybind11::init< pybind11::object&, std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar, unsigned int, FORCE_MODE, unsigned int, IPCReservation*, IPCTaskLock*>())
        .def("getPositionsBuffer", &TensorflowComputeGPU::getPositionsBuffer, pybind11::return_value_policy::reference)
        .def("getNlistBuffer", &TensorflowComputeGPU::getNlistBuffer, pybind11::return_value_policy::reference)
        .def("getForcesBuffer", &TensorflowComputeGPU::getForcesBuffer, pybind11::return_value_policy::reference)
        .def("getVirialBuffer", &TensorflowComputeGPU::getVirialBuffer, pybind11::return_value_policy::reference)
        .def("getPositionsArray", &TensorflowComputeGPU::getPositionsArray, pybind11::return_value_policy::take_ownership)
        .def("getNlistArray", &TensorflowComputeGPU::getNlistArray, pybind11::return_value_policy::take_ownership)
        .def("getForcesArray", &TensorflowComputeGPU::getForcesArray, pybind11::return_value_policy::take_ownership)
        .def("getVirialArray", &TensorflowComputeGPU::getVirialArray, pybind11::return_value_policy::take_ownership)
        .def("isDoublePrecision", &TensorflowComputeGPU::isDoublePrecision)
        .def("getVirialPitch", &TensorflowComputeGPU::getVirialPitch)
    ;
    }

#endif // ENABLE_CUDA


IPCReservation* reserve_memory(unsigned int natoms, unsigned int nneighs) {
    size_t element = sizeof(Scalar);
    size_t size = 0;
    size_t cuda_size = 0;
    #ifdef ENABLE_CUDA
    element = sizeof(cudaIPC_t);
    cuda_size += element; //positions
    cuda_size += element; //forces
    cuda_size += element; // nlist
    cuda_size += element; //virial
    #endif //ENABLE_CUDA
    size += natoms * element * 4; //positions
    size += natoms * element * 4; //forces
    size += natoms * nneighs * element * 4; // nlist
    size += natoms * 9 * element; //virial

    return new IPCReservation(std::max(cuda_size, size));
}

template<>
void ReceiveForcesFunctorAdd::call<IPCCommMode::CPU>(Scalar4* dest, Scalar4* src) {
    for(unsigned int i = 0; i < _N; i++) {
        dest[i].x += src[i].x;
        dest[i].y += src[i].y;
        dest[i].z += src[i].z;
        dest[i].w += src[i].w;
    }
}


template<>
void ReceiveVirialFunctorAdd::call<IPCCommMode::CPU>(Scalar* dest, Scalar* src) {
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
void ReceiveForcesFunctorAdd::call<IPCCommMode::GPU>(Scalar4* dest, Scalar4* src) {
  gpu_add_scalar4(dest, src, _N, *(static_cast<cudaStream_t*> (_stream)) );
}



template<>
void ReceiveVirialFunctorAdd::call<IPCCommMode::GPU>(Scalar* dest, Scalar* src) {
  gpu_add_virial(dest, src, _N, _pitch, *(static_cast<cudaStream_t*> (_stream)) );
}
#endif //ENABLE_CUDA

