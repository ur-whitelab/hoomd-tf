// Copyright (c) 2018 Andrew White at the University of Rochester
//  This file is part of the Hoomd-Tensorflow plugin developed by Andrew White


#ifndef _TENSORFLOW_COMPUTE_H_
#define _TENSORFLOW_COMPUTE_H_

/*! \file TensorflowCompute.h
    \brief Declaration of TensorflowCompute
*/

#include <hoomd/Autotuner.h>
#include <hoomd/ForceCompute.h>
#include <hoomd/HOOMDMath.h>
#include <hoomd/HalfStepHook.h>
#include <hoomd/ParticleData.h>
#include <hoomd/SystemDefinition.h>
#include <hoomd/md/NeighborList.h>
#include "TFArrayComm.h"
#include "TaskLock.h"

// pybind11 is used to create the python bindings to the C++ object,
// but not if we are compiling GPU kernels
#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>
#include <hoomd/extern/pybind/include/pybind11/stl_bind.h>
#endif


namespace hoomd_tf {


  //! A nonsense particle Compute written to demonstrate how to write a plugin
  /*! This Compute simply sets all of the particle's velocities to 0 when update()
  * is called.
  */

 /*!
 * Indicates if forces should be computed by or passed to TF
 */
  enum class FORCE_MODE { tf2hoomd, hoomd2tf };


  template <class T>
  class HalfStepHookWrapper : public HalfStepHook {
    public:T& _f;
    HalfStepHookWrapper(T& f) : _f(f) {}

    void update(unsigned int timestep) override {
      _f.computeForces(timestep);
    }

    // called for half step hook
    void setSystemDefinition(std::shared_ptr<SystemDefinition> sysdef) override {
      //pass
    }

  };

  /*! Template class for TFCompute
  *  \tfparam M If TF is on CPU or GPU.
  *
  */
  template <TFCommMode M = TFCommMode::CPU>
  class TensorflowCompute : public ForceCompute {
  public:
    //! Constructor
    TensorflowCompute(pybind11::object& py_self,
                      std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<NeighborList> nlist, Scalar r_cut,
                      unsigned int nneighs, FORCE_MODE force_mode,
                      unsigned int period,
                      TaskLock* tasklock);

    TensorflowCompute() = delete;

    //! Destructor
    virtual ~TensorflowCompute();


    Scalar getLogValue(const std::string& quantity,
                      unsigned int timestep) override;

    int64_t getForcesBuffer() const;
    int64_t getPositionsBuffer() const;
    int64_t getVirialBuffer() const;
    int64_t getNlistBuffer() const;

    bool isDoublePrecision() const {
  #ifdef SINGLE_PRECISION
      return false;
  #else
      return true;
  #endif  // SINGLE_PRECISION
    }

    std::vector<Scalar4> getForcesArray() const;
    std::vector<Scalar4> getNlistArray() const;
    std::vector<Scalar4> getPositionsArray() const;
    std::vector<Scalar> getVirialArray() const;
    unsigned int getVirialPitch() const { return m_virial.getPitch(); }
    virtual void computeForces(unsigned int timestep) override;
    std::shared_ptr<HalfStepHook> getHook() {
      return hook;
    }

    pybind11::object
        _py_self;  // pybind objects have to be public with current cc flags
    std::shared_ptr<HalfStepHookWrapper<TensorflowCompute<M> > > hook; //need this to add to integrator
  protected:
    // used if particle number changes
    virtual void reallocate();
    //! Take one timestep forward
    virtual void prepareNeighbors();
    virtual void receiveVirial();

    void finishUpdate(unsigned int timestep);

    std::shared_ptr<NeighborList> m_nlist;
    Scalar _r_cut;
    unsigned int _nneighs;
    FORCE_MODE _force_mode;
    unsigned int _period;
    std::string m_log_name;
    TaskLock* _tasklock;

    TFArrayComm<M, Scalar4> _positions_comm;
    TFArrayComm<M, Scalar4> _forces_comm;
    GlobalArray<Scalar4> _nlist_array;
    GlobalArray<Scalar> _virial_array;
    TFArrayComm<M, Scalar4> _nlist_comm;
    TFArrayComm<M, Scalar> _virial_comm;
  };

  //! Export the TensorflowCompute class to python
  void export_TensorflowCompute(pybind11::module& m);


  #ifdef ENABLE_CUDA

  class TensorflowComputeGPU : public TensorflowCompute<TFCommMode::GPU> {
  public:
    //! Constructor
    TensorflowComputeGPU(pybind11::object& py_self,
                        std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<NeighborList> nlist, Scalar r_cut,
                        unsigned int nneighs, FORCE_MODE force_mode,
                        unsigned int period,
                        TaskLock* tasklock);

    void setAutotunerParams(bool enable, unsigned int period) override;

  protected:
    void computeForces(unsigned int timestep) override;
    void reallocate() override;
    void prepareNeighbors() override;
    void receiveVirial() override;

  private:
    std::unique_ptr<Autotuner> m_tuner;  // Autotuner for block size
    cudaStream_t _streams[4];
    size_t _nstreams = 4;
  };

  //! Export the TensorflowComputeGPU class to python
  void export_TensorflowComputeGPU(pybind11::module& m);

  template class TensorflowCompute<TFCommMode::GPU>;
  #endif  // ENABLE_CUDA

  // force implementation
  template class TensorflowCompute<TFCommMode::CPU>;

}

#endif  // _TENSORFLOW_COMPUTE_H_
