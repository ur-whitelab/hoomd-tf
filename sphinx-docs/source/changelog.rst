Change Log
==========

v2.0.rc1
-----

*New Features*

- Tensorflow 2.0 now required
- Keras models replace computation graphs
- Training is handled by Keras/TF functions now
- Getting values now longer requires save/load, instead everything is accessible in Python

*Breaking Changes*

- All code must be rewritten following migration guide
- feeddict is no longer supported
- graphbuilder is now SimModel, which must be subclassed
- All graphbuilder methods (e.g., compute_rdf) are now functions
- Code that duplicates Keras functionality is removed:
    - checkpoint management, model save/load
    - saving values, computing means, other metrics
    - saving values over time is now done by tensorboard or other TF/Keras approaches
- How forces are computed must be explicit and virials are no longer implicit (use modify_virial flag)
- You can no longer save tensors, instead output what you would like to save in your model code
- EDS is now layer (EDSLayer)
- Running from a trajectory is now done via a generator
- Mol batching is now a separate class MolSimModel

v1.0.1 (7/27/2020)
----

*Bug fixes*
- Prevented CPU overflow when nlist is too small (and added unit test)
- Adding check on mapping validity

v1.0 (7/20/2020)
----

*JOSS Review*

Hoomd-TF has been published as a `peer-reviewed article <https://joss.theoj.org/papers/5d1323eadec82aabe86c65a403ff8f90>`_ in the
Journal of Open Source Software (JOSS)

*New Features*

- Added pre-built molecular features
- Added MDAnalysis style selection for defining mapping operators

*Enhancements*

- Docs can now be built without HTF install
- mol batching performance is much better
- Simplified variable saving
- More example notebooks and reduced file sizes of example trajectories
- Supports dynamic boxes
- Better EDS bias naming
- Prevents accidentally computing forces from positions, instead of nlist
- Added guards against compiler mismatch
- Added sanity tests to prevent unsupported CPU/GPU modes
- Added benchmarking script
- Added check for nlist overflows on GPU
- Added check for mismatch on saving variables/tensors
- Fixed all compiler warnings
- Added Dockerfile for containerized use

*Bug Fixes*

- Empty tensorboard summaries no longer crash
- Prevented import issues with name clashes between packages and classes

v0.6 (2020-02-21)
-----------------------

*Enhancements*

- Migrated documentation to sphinx
- Added Jupyter notebook examples
- Various documentation improvements
- Added CUDA 10 Support

v0.5 (2019-10-17)
-----------------------

*Bug Fixes*

- Types are now correctly translated to TF

v0.4 (2019-09-25)
-----------------------

*New Features*

- Added experiment directed simulation biasing to `htf`.

*Enhancements*

- Added box dimension to computation graph (`graph.box` and `graph.box_size`)
- Can now wrap position derived distances with `graph.wrap_vector`
- Made it possible to specify period for `out_nodes`

*Bug Fixes*

- Fixed dangling list element in `rev_mol_indices`

v0.3 (2019-07-03)
-----------------------

*Enhancements*

- Batching by molecule now has a atom id to mol id/atom id look-up (`rev_mol_indices`)
- Version string is visible in package
- Example models now take an argument specifying where to save them
- When batching, atom sorting is automatically disabled
- `compute_pairwise_potential` now outputs force as well as potential

*Bug Fixes*

- Computing nlist in TF now correctly sorts when requested
- Fixed Mac OS specific issues for compiling against existing HOOMD-blue install
- Running mean computation variables are now marked as untrainable

v0.2 (2019-06-03)
-----------------------

*New Features*

- Added attach `batch_size` argument enabling batching of TF calls
- Can now batch by molecule, enabling selection/exclusion of molecules
- Added XLA option to improve TF speed
- Now possible to compile the plugin after hoomd-blue install
- Changed name of package to htf instead of tensorflow_plugin

*Enhancements*

- Changed output logging to only output TF items to the tf_manager.log and
- Log-level is now consistent with hoomd
- Added C++ unit tests skeleton in the same format as HOOMD-blue. Compile with -DBUILD_TESTING=ON to use.
- Switched to hoomd-blue cuda error codes
- Added MPI tests with domain decomposition
- Improved style consistency with hoomd-blue
- Cmake now checks for TF and hoomd versions while building hoomd-tf.

v0.1 (2019-04-22)
-----------------

- Made Python packages actual dependencies.
- Switched to using hoomd-blue cuda error codes.
- Removed TaskLock from C++ code.
- Documentation updates
- Included license.
- User can now use specific hoomd forces in the hoomd2tf force mode.
- Added the ability to create a custom nlist.
- Made unit tests stricter and fixed some cuda synchronization bugs.
- Fixed TF GPU Compiling bug.
- Fixed ordering/masking error in mapping nlist and type of neighbor particles in nlist.
- Fixed a bug which caused a seg fault in nonlist settings.
