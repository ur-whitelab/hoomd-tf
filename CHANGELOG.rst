Change Log
==========

v0.3 (2019-07-03)
-----------------------

*Enhancements*

- Batching by molecule now has a atom id to mol id/atom id look-up (`rev_mol_indices`)
- XLA is enabled by default
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
