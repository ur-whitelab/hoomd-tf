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
 */
template <TFCommMode M>
TensorflowCompute<M>::TensorflowCompute(
    pybind11::object& py_self, std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist, Scalar r_cut, unsigned int nneighs,
    FORCE_MODE force_mode, unsigned int period, unsigned int batch_size)
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
      _batch_size(batch_size){
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

  // try to set it
  if(_nneighs > 0) {
    if(m_nlist->getStorageMode() == NeighborList::half) {
      m_nlist->setStorageMode(NeighborList::full);

      m_exec_conf->msg->warning()
          << "Must use full neighbor list! Attempting to set neighbor storage"
          << std::endl;
    }
  }
  // connect to the ParticleData to receive notifications when the maximum
  // number of particles changes
  m_pdata->getMaxParticleNumberChangeSignal()
      .connect<TensorflowCompute, &TensorflowCompute<M>::reallocate>(this);
}

template <TFCommMode M>
void TensorflowCompute<M>::reallocate() {
  assert(m_pdata);

  // we don't want to hold onto the positions, so just pass as shared.
  unsigned int batch_size = _batch_size == 0 ? m_pdata->getMaxN() : _batch_size;
  GlobalArray<Scalar4> tmpPos(batch_size, m_exec_conf);
  _positions_array.swap(tmpPos);
  _positions_comm = TFArrayComm<M, Scalar4>(_positions_array, "positions", m_exec_conf);
  _forces_comm = TFArrayComm<M, Scalar4>(m_force, "forces", m_exec_conf);
  // In cuda, an array of size 0 breaks things. So even if we aren"t using
  // neighborlist we need to make it size > 0
  if (_nneighs > 0) {
    GlobalArray<Scalar4> tmp(std::max(1U, _nneighs * batch_size), m_exec_conf);
    _nlist_array.swap(tmp);
    _nlist_comm = TFArrayComm<M, Scalar4>(_nlist_array, "nlist", m_exec_conf);
  }
  // virial is made with maxN, not N
  GlobalArray<Scalar>  tmpVirial(9 * batch_size, m_exec_conf);
  _virial_array.swap(tmpVirial);
  _virial_comm =  TFArrayComm<M, Scalar>(_virial_array, "virial", m_exec_conf);
  _virial_comm.memsetArray(0);
}

template <TFCommMode M>
TensorflowCompute<M>::~TensorflowCompute() {

}

/*! Perform the needed calculations
    \param timestep Current time step of the simulation
*/
template <TFCommMode M>
void TensorflowCompute<M>::computeForces(unsigned int timestep) {
  int offset, N;
  if (timestep % _period == 0) {
    if (m_prof) m_prof->push("TensorflowCompute");
    // Batch the operations
    // if batch_size == 0, that means do as big as we need to
    unsigned int batch_size = _batch_size == 0 ? m_pdata->getN() : _batch_size;
    for(int i = 0; i <  m_pdata->getN() / batch_size + 1; i++) {
      offset = i * batch_size;
      // compute batch size so that we don't exceed atom number.
      N = std::min(m_pdata->getN() - offset, batch_size);
      std::cout << "Batch: " << i <<  "  " << offset << " " << N << std::endl;
      if (N < 1)
        break;
      // nneighs == 0 send positions only
      if (_nneighs > 0) {
	_nlist_comm.setBatchSize(N * _nneighs);
        // check again
        if(m_nlist->getStorageMode() == NeighborList::half) {
          m_exec_conf->msg->error() << "Must have full neigbhorlist" << std::endl;
          throw std::runtime_error("neighbor list wrong type");
        }
        // Update the neighborlist once
        if(i == 0) m_nlist->compute(timestep);
        if (m_prof) m_prof->push("TensorflowCompute::reshapeNeighbors");
        prepareNeighbors(offset, N);
        if (m_prof) m_prof->pop();
      }

      // get positions
      std::cout << "Receiving positions" << std::endl;
      _positions_comm.receiveArray(m_pdata->getPositions(), offset);

      // Now we prepare forces if we're sending it
      // forces are size  N, not batch size so we only do this on first batch
      if (_force_mode == FORCE_MODE::hoomd2tf && i == 0) {
        std::cout << "Receiving forces" << std::endl;
        if(_ref_forces.empty()) {
          _forces_comm.receiveArray(m_pdata->getNetForce());
        }
        else {
          sumReferenceForces();
        }
      }

      // set batch sizes for communication
      _positions_comm.setBatchSize(N);
      // forces comm is full size because we use m_forces
      _forces_comm.setOffset(offset);
      _forces_comm.setBatchSize(N);

      std::cout << "Finishing step" << std::endl;
      finishUpdate(i, static_cast<float>(N) / m_pdata->getN());

      if (m_prof) m_prof->push("TensorflowCompute::Force Update");

      // now we receive virial from the update.
      if(_force_mode == FORCE_MODE::tf2hoomd) {
          std::cout << "Receiving Virial" << std::endl;
	  _virial_comm.setBatchSize(N * 9);
          receiveVirial(offset, N);
      }
      if (m_prof) m_prof->pop();  // force update

      #ifdef ENABLE_CUDA
      if(M == TFCommMode::GPU)
        cudaDeviceSynchronize();
      #endif // ENABLE_CUDA

    }
    if (m_prof) m_prof->pop();  // compute
    std::cout << "TFCompute Complete" << std::endl;
  }

}

template <TFCommMode M>
void TensorflowCompute<M>::finishUpdate(unsigned int batch_index, float batch_frac) {
  if (m_prof) m_prof->push("TensorflowCompute:Awaiting TF Update");
  _py_self.attr("finish_update")(batch_index, batch_frac);
  if (m_prof) m_prof->pop();
}

template <TFCommMode M>
void TensorflowCompute<M>::sumReferenceForces() {
  _forces_comm.memsetArray(0);
  ArrayHandle<Scalar4> dest(m_force, access_location::host,
                           access_mode::overwrite);

  for (auto const& forces : _ref_forces) {
    ArrayHandle<Scalar4> src(forces->getForceArray(), access_location::host,
                            access_mode::read);
    for (unsigned int i = 0; i < m_pdata->getN(); i++) {
      dest.data[i].x += src.data[i].x;
      dest.data[i].y += src.data[i].y;
      dest.data[i].z += src.data[i].z;
      dest.data[i].w += src.data[i].w;
      std::cout << "Adding ref forces " << i << dest.data[i].w << " "  << std::endl;
    }
  }
}

template <TFCommMode M>
void TensorflowCompute<M>::receiveVirial(unsigned int batch_offset, unsigned int batch_size) {
  ArrayHandle<Scalar> dest(m_virial, access_location::host,
                           access_mode::readwrite);
  ArrayHandle<Scalar> src(_virial_array, access_location::host,
                          access_mode::read);
  for (unsigned int i = 0; i < batch_size; i++) {
    std::cout << "virial " << i + batch_offset << " " << src.data[i * 9 + 0] << std::endl;
    assert(5 * getVirialPitch() + i + batch_offset < m_virial.getNumElements());
    dest.data[0 * getVirialPitch() + i + batch_offset] += src.data[i * 9 + 0];  // xx
    dest.data[1 * getVirialPitch() + i + batch_offset] += src.data[i * 9 + 1];  // xy
    dest.data[2 * getVirialPitch() + i + batch_offset] += src.data[i * 9 + 2];  // xz
    dest.data[3 * getVirialPitch() + i + batch_offset] += src.data[i * 9 + 4];  // yy
    dest.data[4 * getVirialPitch() + i + batch_offset] += src.data[i * 9 + 5];  // yz
    dest.data[5 * getVirialPitch() + i + batch_offset] += src.data[i * 9 + 8];  // zz
  }
}

template <TFCommMode M>
void TensorflowCompute<M>::prepareNeighbors(unsigned int batch_offset, unsigned int batch_size) {
  // create ptr at offset to where neighbors go
  ArrayHandle<Scalar4> buffer_array(_nlist_array, access_location::host,
                                    access_mode::overwrite);
  Scalar4* buffer = buffer_array.data;
  //zero out buffer
  memset(buffer, 0, _nlist_array.getNumElements() * sizeof(Scalar4));
  unsigned int* nnoffset =
      (unsigned int*)calloc(batch_size, sizeof(unsigned int));

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
  int bi = 0;
  for (int i = batch_offset; i < batch_offset + batch_size; i++, bi++) {
    // access the particle's position and type (MEM TRANSFER: 4 scalars)
    Scalar3 pi =
        make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
    const unsigned int head_i = h_head_list.data[i];

    // loop over all of the neighbors of this particle
    const unsigned int size = (unsigned int)h_n_neigh.data[i];
    unsigned int j = 0;

    if (_nneighs < size) {
      m_exec_conf->msg->error() << "Overflow in nlist! Only " << _nneighs << " space but there are " << size << " neighbors." << std::endl;
      throw std::runtime_error("Nlist Overflow");
    }
    for (; j < size; j++) {

      // access the index of this neighbor
      unsigned int k = h_nlist.data[head_i + j];

      // calculate dr
      Scalar3 pk =
          make_scalar3(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
      Scalar3 dx = pk - pi;

      // apply periodic boundary conditions
      dx = box.minImage(dx);
      // if((i * _nneighs + nnoffset[bi]) >=  _nlist_array.getNumElements())
      // std::cerr << "Error: " << bi << " " <<  _batch_size << " " << (bi * _nneighs + nnoffset[bi]) << " " << _nlist_array.getNumElements() << std::endl;
      if (dx.x * dx.x + dx.y * dx.y + dx.z * dx.z > _r_cut * _r_cut) continue;
      buffer[bi * _nneighs + nnoffset[bi]].x = dx.x;
      buffer[bi * _nneighs + nnoffset[bi]].y = dx.y;
      buffer[bi * _nneighs + nnoffset[bi]].z = dx.z;
      buffer[bi * _nneighs + nnoffset[bi]].w = h_pos.data[k].w;
      nnoffset[bi]++;
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
        .def(pybind11::init< pybind11::object&, std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar, unsigned int, FORCE_MODE, unsigned int, unsigned int>())
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
        .def("addReferenceForce", &TensorflowCompute<TFCommMode::CPU>::addReferenceForce)
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
             unsigned int batch_size)
     : TensorflowCompute(py_self, sysdef, nlist, r_cut, nneighs, force_mode, period, batch_size)
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

void TensorflowComputeGPU::prepareNeighbors(unsigned int offset, unsigned int batch_size) {

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
                      offset,
                      batch_size,
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

void TensorflowComputeGPU::receiveVirial(unsigned int batch_offset, unsigned int batch_size) {
  ArrayHandle<Scalar> h_virial(m_virial, access_location::device, access_mode::overwrite);
  ArrayHandle<Scalar> tf_h_virial(_virial_array, access_location::device, access_mode::read);
  gpu_add_virial(h_virial.data + batch_offset, tf_h_virial.data, batch_size, getVirialPitch(), _virial_comm.getCudaStream());
}

void TensorflowComputeGPU::sumReferenceForces() {
  _forces_comm.memsetArray(0);
  ArrayHandle<Scalar4> dest(m_force, access_location::device,
                           access_mode::overwrite);
  for (auto const& forces : _ref_forces) {
    ArrayHandle<Scalar4> src(forces->getForceArray(), access_location::device,
                            access_mode::read);
    gpu_add_scalar4(dest.data, src.data, m_force.getNumElements(), _forces_comm.getCudaStream());
  }
}
/* Export the GPU Compute to be visible in the python module
 */
void hoomd_tf::export_TensorflowComputeGPU(pybind11::module& m)
    {
    pybind11::class_<TensorflowComputeGPU, std::shared_ptr<TensorflowComputeGPU>, ForceCompute>(m, "TensorflowComputeGPU")
        .def(pybind11::init< pybind11::object&, std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar, unsigned int, FORCE_MODE, unsigned int, unsigned int>())
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
        .def("addReferenceForce", &TensorflowComputeGPU::addReferenceForce)
    ;
    }

#endif // ENABLE_CUDA
