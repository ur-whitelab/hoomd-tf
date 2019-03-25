// Copyright (c) 2018 Andrew White at the University of Rochester
//  This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

#include "TensorflowCompute.h"
#ifdef ENABLE_CUDA
#include "TensorflowCompute.cuh"
#endif

#include <string.h>
#include <sys/mman.h>
#include <iostream>

using namespace hoomd_tf;

/*! \file TensorflowCompute.cc
    \brief Contains code for TensorflowCompute
*/
// ********************************
// here follows the code for TensorflowCompute

/*! \param py_self Python object tfcompute. So that methods can be called
    \param sysdef SystemDefinition this compute will act on. Must not be NULL.
    \param nlist Neighborlist
    \param r_cut Cutoff for processing nlist which is then passed to TF
    \param nneighs Maximum size for neighbors passed to TF
    \param force_mode Whether we should be computed forces in TF or sending them to TF
    \param period The period between TF updates
    \param tasklock Currently unused. When this was multiporcess, this allowed simultaneous updates
 */
template <TFCommMode M>
TensorflowCompute<M>::TensorflowCompute(
    pybind11::object& py_self, std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist, Scalar r_cut, unsigned int nneighs,
    FORCE_MODE force_mode, unsigned int period,
    TaskLock* tasklock)
    : ForceCompute(sysdef),
      _py_self(py_self),
      //Why? Because I cannot get pybind to export multiple inheritance
      //class (HalfStepHook, ForceCompute), so I make a HalfStepHook wrapper
      // that dispatches to my update(..). BUT, I want to call computeForces
      // which is protected in ForceCompute, so I cannot use any type inference.
      // I hate it too
      hook(std::make_shared<HalfStepHookWrapper<TensorflowCompute<M> > >(HalfStepHookWrapper<TensorflowCompute<M> >(*this))),
      m_nlist(nlist),
      _r_cut(r_cut),
      _nneighs(nneighs),
      _force_mode(force_mode),
      _period(period),
      _tasklock(tasklock) {
  m_exec_conf->msg->notice(2)
      << "Starting TensorflowCompute "
      << std::endl;
  reallocate();
  m_exec_conf->msg->notice(2) << "completed reallocate" << std::endl;
  m_log_name = std::string("tensorflow");
  auto flags = this->m_pdata->getFlags();
  if (_force_mode == FORCE_MODE::tf2hoomd) {
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

template <TFCommMode M>
void TensorflowCompute<M>::reallocate() {
  assert(m_pdata);

  // we won't ever override positions,
  // but the recieve method does exist
  // so we'll cast away until I make a version
  // of TFArrayComm that can't override array
  _positions_comm = TFArrayComm<M, Scalar4>(
      const_cast<GlobalArray<Scalar4>&>(m_pdata->getPositions()), "positions");
  _forces_comm = TFArrayComm<M, Scalar4>(m_force, "forces");
  // In cuda, an array of size 0 breaks things. So even if we aren"t using
  // neighborlist we need to make it size > 0
  if (_nneighs > 0) {
    GlobalArray<Scalar4> tmp(std::max(1U, _nneighs * m_pdata->getMaxN()), m_exec_conf);
    _nlist_array.swap(tmp);
    _nlist_comm = TFArrayComm<M, Scalar4>(_nlist_array, "nlist");
  }
  // virial is made with maxN, not N
  GlobalArray<Scalar>  tmp2(9 * m_pdata->getMaxN(), m_exec_conf);
  _virial_array.swap(tmp2);
  _virial_comm =  TFArrayComm<M, Scalar>(_virial_array, "virial");
  _virial_comm.memsetArray(0);
}

template <TFCommMode M>
TensorflowCompute<M>::~TensorflowCompute() {
  delete _tasklock;
}

/*! Perform the needed calculations
    \param timestep Current time step of the simulation
*/
template <TFCommMode M>
void TensorflowCompute<M>::computeForces(unsigned int timestep) {
  if (timestep % _period == 0) {
    if (m_prof) m_prof->push("TensorflowCompute<M>");
    if (m_prof) m_prof->push("TensorflowCompute<M>::Preparing Data for TF");
    // nneighs == 0 send positions only
    if (_nneighs > 0) {
      // Update the neighborlist
      m_nlist->compute(timestep);
      if (m_prof) m_prof->push("TensorflowCompute<M>::reshapeNeighbors");
      prepareNeighbors();
      if (m_prof) m_prof->pop();
    }

    if (m_prof) m_prof->pop(); //prearing data

    // positions and forces are ready. Now we send
    if (_force_mode == FORCE_MODE::hoomd2tf) {
      if(_ref_forces)
        _forces_comm.receiveArray(_ref_forces->getForceArray());
      else
        _forces_comm.receiveArray(m_pdata->getNetForce());
    }

    finishUpdate(timestep);

    if (m_prof) m_prof->push("TensorflowCompute<M>::Force Update");

    // now we receive virial from the update.
    if(_force_mode == FORCE_MODE::tf2hoomd) {
        receiveVirial();
    }
    if (m_prof) m_prof->pop();  // force update

    #ifdef ENABLE_CUDA
    if(M == TFCommMode::GPU)
      cudaDeviceSynchronize();
    #endif // ENABLE_CUDA

    if (m_prof) m_prof->pop();  // compute
  }
}

template <TFCommMode M>
void TensorflowCompute<M>::finishUpdate(unsigned int timestep) {
  if (m_prof) m_prof->push("TensorflowCompute<M>::Awaiting TF Update");
  _py_self.attr("finish_update")(timestep);
  // _tasklock->await();
  if (m_prof) m_prof->pop();
}

template <TFCommMode M>
void TensorflowCompute<M>::receiveVirial() {
  ArrayHandle<Scalar> dest(m_virial, access_location::host,
                               access_mode::overwrite);
  ArrayHandle<Scalar> src(_virial_array, access_location::host,
  access_mode::read);
  for(unsigned int i = 0; i < m_pdata->getN(); i++) {
      dest.data[0 * getVirialPitch() + i] += src.data[i * 9 + 0]; //xx
      dest.data[1 * getVirialPitch() + i] += src.data[i * 9 + 1]; //xy
      dest.data[2 * getVirialPitch() + i] += src.data[i * 9 + 2]; //xz
      dest.data[3 * getVirialPitch() + i] += src.data[i * 9 + 4]; //yy
      dest.data[4 * getVirialPitch() + i] += src.data[i * 9 + 5]; //yz
      dest.data[5 * getVirialPitch() + i] += src.data[i * 9 + 8]; //zz
  }
}

template <TFCommMode M>
void TensorflowCompute<M>::prepareNeighbors() {
  // create ptr at offset to where neighbors go
  ArrayHandle<Scalar4> buffer_array(_nlist_array, access_location::host,
                                    access_mode::overwrite);
  Scalar4* buffer = buffer_array.data;
  //zero out buffer
  memset(buffer, 0, _nlist_array.getNumElements() * sizeof(Scalar4));
  unsigned int* nnoffset =
      (unsigned int*)calloc(m_pdata->getMaxN(), sizeof(unsigned int));

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
      m_exec_conf->msg->error() << "Overflow in nlist! Only " << _nneighs << " space but there are " << size << " neighbors." << std::endl;
    for (; j < size; j++) {

      // access the index of this neighbor
      unsigned int k = h_nlist.data[head_i + j];

      // calculate dr
      Scalar3 pk =
          make_scalar3(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
      Scalar3 dx = pk - pi;

      // apply periodic boundary conditions
      dx = box.minImage(dx);
      if((i * _nneighs + nnoffset[i]) >=  _nlist_array.getNumElements())
        std::cout << "Error: " << i << " " <<  m_pdata->getMaxN() << " " << (i * _nneighs + nnoffset[i]) << " " << _nlist_array.getNumElements() << std::endl;
      if (dx.x * dx.x + dx.y * dx.y + dx.z * dx.z > _r_cut * _r_cut) continue;
      buffer[i * _nneighs + nnoffset[i]].x = dx.x;
      buffer[i * _nneighs + nnoffset[i]].y = dx.y;
      buffer[i * _nneighs + nnoffset[i]].z = dx.z;
      buffer[i * _nneighs + nnoffset[i]].w = h_pos.data[k].w;
      nnoffset[i]++;
      //TODO: Why is k so big?
      if (m_nlist->getStorageMode() == NeighborList::half && k < m_pdata->getN()) {
        buffer[k * _nneighs + nnoffset[k]].x = -dx.x;
        buffer[k * _nneighs + nnoffset[k]].y = -dx.y;
        buffer[k * _nneighs + nnoffset[k]].z = -dx.z;
        buffer[k * _nneighs + nnoffset[k]].w = h_pos.data[i].w;
        nnoffset[k]++;
      }
    }
  }

  free(nnoffset);
}

template <TFCommMode M>
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

//these below are how we communicate memory addresses to TF
template<TFCommMode M>
int64_t TensorflowCompute<M>::getForcesBuffer() const { return _forces_comm.getAddress();}
template<TFCommMode M>
int64_t TensorflowCompute<M>::getPositionsBuffer() const {return _positions_comm.getAddress();}
template<TFCommMode M>
int64_t TensorflowCompute<M>::getVirialBuffer() const {return _virial_comm.getAddress();}
template<TFCommMode M>
int64_t TensorflowCompute<M>::getNlistBuffer() const {return _nlist_comm.getAddress();}

template<TFCommMode M>
std::vector<Scalar4> TensorflowCompute<M>::getPositionsArray() const {return _positions_comm.getArray();}
template<TFCommMode M>
std::vector<Scalar4> TensorflowCompute<M>::getNlistArray() const {return _nlist_comm.getArray();}
template<TFCommMode M>
std::vector<Scalar4> TensorflowCompute<M>::getForcesArray() const {return _forces_comm.getArray();}
template<TFCommMode M>
std::vector<Scalar> TensorflowCompute<M>::getVirialArray() const {return _virial_comm.getArray();}

/* Export the CPU Compute to be visible in the python module
 */
void hoomd_tf::export_TensorflowCompute(pybind11::module& m)
    {

      //need to export halfstephook, since it's not exported anywhere else
    pybind11::class_<HalfStepHook, std::shared_ptr<HalfStepHook> >(m, "HalfStepHook");


    pybind11::class_<TensorflowCompute<TFCommMode::CPU>, std::shared_ptr<TensorflowCompute<TFCommMode::CPU> >, ForceCompute>(m, "TensorflowCompute")
        .def(pybind11::init< pybind11::object&, std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar, unsigned int, FORCE_MODE, unsigned int, TaskLock*>())
        .def("getPositionsBuffer", &TensorflowCompute<TFCommMode::CPU>::getPositionsBuffer, pybind11::return_value_policy::reference)
        .def("getNlistBuffer", &TensorflowCompute<TFCommMode::CPU>::getNlistBuffer, pybind11::return_value_policy::reference)
        .def("getForcesBuffer", &TensorflowCompute<TFCommMode::CPU>::getForcesBuffer, pybind11::return_value_policy::reference)
        .def("getVirialBuffer", &TensorflowCompute<TFCommMode::CPU>::getVirialBuffer, pybind11::return_value_policy::reference)
        .def("getPositionsArray", &TensorflowCompute<TFCommMode::CPU>::getPositionsArray, pybind11::return_value_policy::take_ownership)
        .def("getNlistArray", &TensorflowCompute<TFCommMode::CPU>::getNlistArray, pybind11::return_value_policy::take_ownership)
        .def("getForcesArray", &TensorflowCompute<TFCommMode::CPU>::getForcesArray, pybind11::return_value_policy::take_ownership)
        .def("getVirialArray", &TensorflowCompute<TFCommMode::CPU>::getVirialArray, pybind11::return_value_policy::take_ownership)
        .def("isDoublePrecision", &TensorflowCompute<TFCommMode::CPU>::isDoublePrecision)
        .def("getVirialPitch", &TensorflowCompute<TFCommMode::CPU>::getVirialPitch)
        .def("hook", &TensorflowCompute<TFCommMode::CPU>::getHook)
        .def("setReferenceForces", &TensorflowCompute<TFCommMode::CPU>::setReferenceForces)
    ;
    pybind11::enum_<FORCE_MODE>(m, "FORCE_MODE")
        .value("tf2hoomd", FORCE_MODE::tf2hoomd)
        .value("hoomd2tf", FORCE_MODE::hoomd2tf)
    ;
    }

// ********************************
// here follows the code for TensorflowCompute on the GPU

#ifdef ENABLE_CUDA

TensorflowComputeGPU::TensorflowComputeGPU(pybind11::object& py_self,
            std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<NeighborList> nlist,
             Scalar r_cut, unsigned int nneighs,
             FORCE_MODE force_mode, unsigned int period,
             TaskLock* tasklock)
     : TensorflowCompute(py_self, sysdef, nlist, r_cut, nneighs, force_mode, period, tasklock)
{

    //want nlist on stream 0 since a nlist rebuild is
    //called just before prepareNeighbors
    _streams[0] = 0;
    for(unsigned int i = 1; i < _nstreams; i++) {
      cudaStreamCreate(&(_streams[i]));
      //_streams[i] = 0;
      CHECK_CUDA_ERROR();
    }

    if(_nneighs > 0) {
      _nneighs = std::min(m_nlist->getNListArray().getPitch(),nneighs);
      if(_nneighs != nneighs) {
	m_exec_conf->msg->notice(2) << "set nneighs to be " << _nneighs << " to match GPU nlist array pitch" << std::endl;
      }
    }
    reallocate(); //must be called so streams are correctly set
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "tensorflow", m_exec_conf));
}

void TensorflowComputeGPU::reallocate()  {
  TensorflowCompute::reallocate();
  _nlist_comm.setCudaStream(_streams[0]);
  _virial_comm.setCudaStream(_streams[1]);
  _forces_comm.setCudaStream(_streams[2]);
  _positions_comm.setCudaStream(_streams[3]);

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

void TensorflowComputeGPU::receiveVirial() {
  ArrayHandle<Scalar> h_virial(m_virial, access_location::device, access_mode::overwrite);
  ArrayHandle<Scalar> tf_h_virial(_virial_array, access_location::device, access_mode::read);
  gpu_add_virial(h_virial.data, tf_h_virial.data, m_pdata->getN(), getVirialPitch(), _virial_comm.getCudaStream());
}
/* Export the GPU Compute to be visible in the python module
 */
void hoomd_tf::export_TensorflowComputeGPU(pybind11::module& m)
    {
    pybind11::class_<TensorflowComputeGPU, std::shared_ptr<TensorflowComputeGPU>, ForceCompute>(m, "TensorflowComputeGPU")
        .def(pybind11::init< pybind11::object&, std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar, unsigned int, FORCE_MODE, unsigned int, TaskLock*>())
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
        .def("hook", &TensorflowComputeGPU::getHook)
        .def("setReferenceForces", &TensorflowComputeGPU::setReferenceForces)
    ;
    }

#endif // ENABLE_CUDA
