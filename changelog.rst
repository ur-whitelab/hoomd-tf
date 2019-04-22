# Change Log

## v0.2 (Not yet released)

*New Features*

- Make it possible to disable hoomd-tf compute in script.
- Make code consistent for hoomd-blue plugin.
- Add config options and test XLA benchmarks for CG mapping operators code and other multistep complex code.

*Enhancements*

- Change output logging to only output TF items to the tf_manager.log and also respect the log-level set in hoomd.
- Add C++ tests and coverage analysis.
- Add timing tests for assessing regression in performance.

*Bug Fixes*

- Batch positions/nlist for execution because very large systems cannot fit a complex NN model into memory,

## v0.1 (2019-04-22)

*Note:* only major changes are listed here.

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
