// Copyright (c) 2020 HOOMD-TF Developers

#ifndef m_TENSORFLOW_COMPUTE_H_
#define m_TENSORFLOW_COMPUTE_H_

#include <hoomd/Autotuner.h>
#include <hoomd/ForceCompute.h>
#include <hoomd/HOOMDMath.h>
#include <hoomd/HalfStepHook.h>
#include <hoomd/ParticleData.h>
#include <hoomd/SystemDefinition.h>
#include <hoomd/md/NeighborList.h>
#include "TFArrayComm.h"

// pybind11 is used to create the python bindings to the C++ object,
// but not if we are compiling GPU kernels
#ifndef NVCC
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#endif

namespace hoomd_tf
{
    /*! \file TensorflowCompute.h
     *  \brief Declaration of TensorflowCompute
     *
     *  This is the main class of the HOOMD-TF package, responsible for the
     *  communication between HOOMD and TensorFlow, optionally using GPU-GPU
     *  memory transfer in GPU mode.
     *  HOOMD neighbor lists are passed to and from TensorFlow as comm objects.
     *  \sa TFArrayComm class
    */

    //! A nonsense particle Compute written to demonstrate how to write a plugin
    /*! This Compute simply sets all of the particle's velocities to 0 when update()
     * is called.
     */

    /*! FORCE_MODE class
     *  Indicates if forces should be computed by or passed to TF, respectively.
     */
    enum class FORCE_MODE
    {
        tf2hoomd,
        hoomd2tf
    };

    /*! HalfStepHookWrapper class
     *  Wrapper around HOOMD-blue's HalfStepHook class.
     *  Overrides update method to enable call of TF for force computing.
     */
    template <class T>
    class HalfStepHookWrapper : public HalfStepHook
    {
    public:
        T &m_f;
        HalfStepHookWrapper(T &f) : m_f(f) {}

        //! override update from HOOMD to compute TF forces also
        void update(unsigned int timestep) override
        {
            m_f.computeForces(timestep);
        }

        //! called for half step hook
        void setSystemDefinition(std::shared_ptr<SystemDefinition> sysdef) override
        {
            //pass
        }
    };

    /*! Template class for TFCompute
     *  \tfparam M If TF is on CPU or GPU.
     *
     */
    template <TFCommMode M = TFCommMode::CPU>
    class TensorflowCompute : public ForceCompute
    {
    public:
        //! Constructor
        TensorflowCompute(pybind11::object &py_self,
                          std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<NeighborList> nlist,
                          Scalar r_cut,
                          unsigned int nneighs,
                          FORCE_MODE force_mode,
                          unsigned int period,
                          unsigned int batch_size);

        //! No base constructor
        TensorflowCompute() = delete;

        //! Destructor
        virtual ~TensorflowCompute();

        //! Returns log value of specified quantity at chosen timestep
        Scalar getLogValue(const std::string &quantity,
                           unsigned int timestep) override;

        //! Returns address of TFArrayComm object holding forces
        int64_t getForcesBuffer() const;

        //! Returns address of TFArrayComm object holding positions
        int64_t getPositionsBuffer() const;

        //! Returns address of TFArrayComm object holding box
        int64_t getBoxBuffer() const;

        //! Returns address of TFArrayComm object holding virial
        int64_t getVirialBuffer() const;

        //! Returns address of TFArrayComm object holding neighbor list
        int64_t getNlistBuffer() const;

        //! Check what precision level we're using for CUDA purposes
        bool isDoublePrecision() const
        {
#ifdef SINGLE_PRECISION
            return false;
#else
            return true;
#endif // SINGLE_PRECISION
        }

        void setMappedNlist(bool mn, unsigned int cg_typeid_start);

        //! Returns the array of forces from associated TFArrayComm object
        std::vector<Scalar4> getForcesArray() const;

        //! Returns the array of neighbor lists from associated TFArrayComm object
        std::vector<Scalar4> getNlistArray() const;

        //! Returns the array of positions from associated TFArrayComm object
        std::vector<Scalar4> getPositionsArray() const;

        //! Returns the array of box dims from associated TFArrayComm object
        std::vector<Scalar3> getBoxArray() const;

        //! Returns the array of virials from associated TFArrayComm object
        std::vector<Scalar> getVirialArray() const;

        //! Dispatches computation of forces according to FORCE_MODE
        virtual void computeForces(unsigned int timestep) override;

        //! Get the memory pitch of the virial
        unsigned int getVirialPitch() const { return m_virial.getPitch(); }
        std::shared_ptr<HalfStepHook> getHook()
        {
            return hook;
        }

        //! Add a separately computed or tabular force
        void addReferenceForce(std::shared_ptr<ForceCompute> force)
        {
            m_ref_forces.push_back(force);
        }

        //! pybind objects have to be public with current cc flags
        pybind11::object m_py_self;

        //! need this to add to integrator in HOOMD
        std::shared_ptr<HalfStepHookWrapper<TensorflowCompute<M>>> hook;

    protected:
        //! used if particle number changes
        virtual void reallocate();

        //! Set up neighbor list to take one timestep forward
        virtual void prepareNeighbors(unsigned int offset, unsigned int batch_size);

        //! Transfer virial from TF memory location to HOOMD
        virtual void receiveVirial(unsigned int offset, unsigned int batch_size);

        //! Add up all the reference forces from TF to HOOMD
        virtual void sumReferenceForces();

        //! Update box
        void updateBox();

        //! When TF updates are all finished, send word to python
        void finishUpdate(unsigned int offset);

        //! Called at beginning of update
        void startUpdate();

        //! pointer to the neighbor lists of all particles
        std::shared_ptr<NeighborList> m_nlist;

        //! cutoff radius
        Scalar m_r_cut;

        //! max number of neighbors
        unsigned int m_nneighs;

        //! specify which force mode we are using
        FORCE_MODE m_force_mode;

        //! how frequently we actually do the TF update
        unsigned int m_period;

        //! The batch size for sending/receiving TF updates
        unsigned int m_batch_size;

        //! Flag for mapped nlist computation
        bool m_b_mapped_nlist;

        //! Start of CG bead typeids
        unsigned int m_cg_typeid_start;

        //! name of log used in TF
        std::string m_log_name;

        //! vector of reference forces as ForceCompute objects
        std::vector<std::shared_ptr<ForceCompute>> m_ref_forces;

        //! comm object for holding positions
        TFArrayComm<M, Scalar4> m_positions_comm;

        //! array of positions, which is size of batch
        GlobalArray<Scalar4> m_positions_array;

        //! box dims
        TFArrayComm<M, Scalar3> m_box_comm;

        //! box array
        GlobalArray<Scalar3> m_box_array;

        //! comm object for holding forces
        TFArrayComm<M, Scalar4> m_forces_comm;

        //! array of neighbor list values (x, y, z, w)
        GlobalArray<Scalar4> m_nlist_array;

        //! array of virial values
        GlobalArray<Scalar> m_virial_array;

        //! comm object for holding neighbor lists
        TFArrayComm<M, Scalar4> m_nlist_comm;

        //! comm object for holding virials
        TFArrayComm<M, Scalar> m_virial_comm;
    };

    //! Export the TensorflowCompute class to python
    void export_TensorflowCompute(pybind11::module &m);

#ifdef ENABLE_CUDA

    /*! GPU version of TensorflowCompute class
         *
         */
    class TensorflowComputeGPU : public TensorflowCompute<TFCommMode::GPU>
    {
    public:
        //! Constructor
        TensorflowComputeGPU(pybind11::object &py_self,
                             std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<NeighborList> nlist,
                             Scalar r_cut,
                             unsigned int nneighs,
                             FORCE_MODE force_mode,
                             unsigned int period,
                             unsigned int batch_size);

        /*! Set what HOOMD autotuner params to use
             *  \param enable whether to use autotuner
             *  \param period period with which to use autotuner
             */
        void setAutotunerParams(bool enable, unsigned int period) override;

    protected:
        /*! GPU version calls CPU reallocate and resets cudaStreams for comm objects
             *  \sa TensorflowCompute::reallocate()
             */
        void reallocate() override;

        //! invokes a kernel version of prepareNeighbors
        //! \sa TensorflowCompute::prepareNeighbors()
        void prepareNeighbors(unsigned int offset, unsigned int batch_size) override;

        /*! Use a GPU kernel to transfer the virial values
             *  \sa TensorflowCompute::receiveVirial()
             */
        void receiveVirial(unsigned int offset, unsigned int batch_size) override;

        /*! Use a GPU kernel to add up reference forces
             *  \sa TensorflowCompute::sumReferenceForces()
             */
        void sumReferenceForces() override;

    private:
        std::unique_ptr<Autotuner> m_tuner; //! Autotuner for block size
        cudaStream_t m_streams[4];          //! Array of CUDA streams
        size_t m_nstreams = 4;              //! Number of CUDA streams
    };

    //! Export the TensorflowComputeGPU class to python
    void export_TensorflowComputeGPU(pybind11::module &m);

    template class TensorflowCompute<TFCommMode::GPU>;
#endif // ENABLE_CUDA

    //! force implementation even if no CUDA found
    template class TensorflowCompute<TFCommMode::CPU>;

}

#endif // m_TENSORFLOW_COMPUTE_H_
