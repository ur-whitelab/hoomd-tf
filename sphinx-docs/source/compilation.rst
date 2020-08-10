.. _compiling:

Compiling
=========

Quick Install -- No GPU support
----------------------------------

Try this to install as quickly as possible,
assuming you have conda

.. code:: bash

  conda create -n htf2 python=3.7
  source activate htf2
  conda install -c conda-forge tbb-devel hoomd
  pip install --upgrade tensorflow
  git clone https://github.com/ur-whitelab/hoomd-tf
  cd hoomd-tf && mkdir build && cd build
  CXX=g++ CC=gcc CMAKE_PREFIX_PATH=$CONDA_PREFIX cmake ..
  make install && cd ..
  python htf/test-py/test_sanity.py

Didn't work? Read below for more detailed instructions

Prerequisites
----------------

The following packages are required to compile:

::

    python >= 3.6
    tensorflow >= 2.0
    hoomd >= 2.5.2
    tbb-devel (only for hoomd 2.8 and above if installed with conda)

tbb-devel is required for HOOMD-blue 2.8 or above when using the
HOOMD-blue conda release. It is not automatically installed when
installing HOOMD-blue, so use ``conda install -c conda-forge
tbb-devel`` to install. The TensorFlow version should be any
TensorFlow 2 release. It is recommended you install via pip:

.. code:: bash

  pip install --upgrade tensorflow

Checkout the `TensorFlow install guide <https://www.tensorflow.org/install>`_
to see details.

Python and GCC requirements
--------------------------------

If you install TensorFlow with pip, as recommended, this
provides a pre-built version of TensorFlow which has
specific GCC and Python versions. When you compile
Hoomd-tf, these must match what is found by cmake. So if your version
of TensorFlow used gcc-7x, then you must have gcc-7x available on your machine.
The cmake script in Hoomd-tf will check for this and tell you if they do not match.


.. _simple_compiling:

Simple Compiling
----------------

Install HOOMD-blue and TensorFlow by your preferred method.
We recommend installing TensorFlow with pip,
as ``pip install --upgrade tensorflow``. HOOMD-blue
distributes prebuilt binaries via conda for both CPU versions (``conda install -c conda-forge hoomd``)
|ss| and GPU versions (``conda install -c conda-forge hoomd=*=*gpu*``). If using GPU, make sure the CUDA
toolkit version between TensorFlow and Hoomd match. |se|
*As of August 2020, there are no GPU compatible CUDA/Hoomd/TF combinations on conda-forge.*
Due to the frequent CUDA version mismatches on conda, it is recommended to compile HOOMD-blue if you
intend to use HOOMD-TF in GPU mode.
You can compile HOOMD-blue using `their instructions
<http://hoomd-blue.readthedocs.io>`_.

**Steps after installing TensorFlow and HOOMD-blue**

.. code:: bash

    git clone https://github.com/ur-whitelab/hoomd-tf
    cd hoomd-tf && mkdir build && cd build
    CXX=g++ CC=gcc cmake ..
    make install

If you are using a conda environment, you may need
to add a ``CMAKE_PREFIX_PATH``, like so:

.. code:: bash

    git clone https://github.com/ur-whitelab/hoomd-tf
    cd hoomd-tf && mkdir build && cd build
    CXX=g++ CC=gcc CMAKE_PREFIX_PATH=/path/to/environment cmake ..
    make install

Check your install by running ``python
htf/test-py/test_sanity.py``.  If you have installed with GPU support, also
check with ``python htf/test-py/_test_gpu_sanity.py``.

.. _compiling_with_hoomd_blue:

Compiling with Hoomd-Blue
-------------------------

Use this method if you need to compile with developer flags on or other
special requirements. Note, these steps are NOT required for GPU support! You can
simply compile HOOMD-blue and follow simple instructions above for GPU support.

.. code:: bash

    git clone --recursive https://bitbucket.org/glotzer/hoomd-blue hoomd-blue

You can check out a specific version of HOOMD-blue now, if desired:

.. code:: bash

    cd hoomd-blue && git checkout tags/v2.5.2

Now we put our plugin in the source directory with a softlink:

.. code:: bash

    git clone https://github.com/ur-whitelab/hoomd-tf
    ln -s $HOME/hoomd-tf/htf $HOME/hoomd-blue/hoomd

Now compile (from hoomd-blue directory). Modify options for speed if
necessary. Set build type to `DEBUG` if you need to troubleshoot.

.. code:: bash

    mkdir build && cd build
    CXX=g++ CC=gcc cmake .. -DCMAKE_BUILD_TYPE=Release \
     -DENABLE_CUDA=ON -DENABLE_MPI=OFF\
     -DBUILD_HPMC=off -DBUILD_CGCMM=off -DBUILD_MD=on\
     -DBUILD_METAL=off -DBUILD_TESTING=off -DBUILD_DEPRECATED=off -DBUILD_MPCD=OFF \
     -DCMAKE_INSTALL_PREFIX=`python -c "import site; print(site.getsitepackages()[0])"`

Now compile with make:

.. code:: bash

    make

Option 1: Put build directory on your python path:

.. code:: bash

    export PYTHONPATH="$PYTHONPATH:`pwd`"

Option 2: Install in your python site-packages

.. code:: bash

    make install

.. _conda_environments:

Conda Environments
------------------

If you are using a conda environment, you may need to force cmake to
find your python environment. This is rare, we only see it on our
compute cluster which has multiple conflicting version of python and
conda. The following additional flags can help with this:

.. code:: bash

    export CMAKE_PREFIX_PATH=/path/to/environment
    CXX=g++ CC=gcc cmake .. \
    -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    -DPYTHON_EXECUTABLE=$(which python)

.. _mbuild_environment:

MBuild Environment
------------------

If you are using mbuild, please follow these additional install steps:

.. code:: bash

    pip install requests networkx matplotlib scipy pandas plyplus lxml mdtraj oset cython
    conda install -c omnia -y openmm parmed
    conda install -c conda-forge --no-deps -y packmol gsd
    pip install --upgrade git+https://github.com/mosdef-hub/foyer git+https://github.com/mosdef-hub/mbuild

.. _hpc_installation:

HPC Installation
----------------------------

These are instructions for our group's cluster (BlueHive), and not for general users. **Feeling Lucky?** Try this for quick results

.. code:: bash

    module load cudnn/10.1-7.6.5 anaconda3/2020.02 openmpi/4.0.4/b1 gcc/7.3.0 cmake git zmq
    export PYTHONNOUSERSITE=True
    conda create -n hoomd-tf python=3.7
    source activate hoomd-tf
    export CMAKE_PREFIX_PATH=/path/to/environment
    python -m pip install tensorflow
    git clone https://github.com/glotzerlab/hoomd-blue
    cd hoomd-blue && mkdir build && cd build
    CXX=g++ CC=gcc cmake .. -DCMAKE_INSTALL_PREFIX=`python -c "import site; print(site.getsitepackages()[0])"` -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=ON -DENABLE_MPI=OFF
    make install && cd
    git clone https://github.com/ur-whitelab/hoomd-tf
    cd hoomd-tf && mkdir build && cd build
    CXX=g++ CC=gcc cmake .. \
      -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
      -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
      -DPYTHON_EXECUTABLE=$(which python)
    make install
    cd .. && python htf/test-py/test_sanity.py

Here are the more detailed steps. Clone the ``hoomd-tf`` repo
and then follow these steps:

Load the modules necessary:

.. code:: bash

    module load cudnn/10.1-7.6.5 anaconda3/2020.02 openmpi/4.0.4/b1 gcc/7.3.0 cmake git zmq

Set-up virtual python environment *ONCE* to keep packages isolated.

.. code:: bash

    conda create -n hoomd-tf python=3.7
    source activate hoomd-tf
    python -m pip install tensorflow

Then whenever you login and *have loaded modules*:

.. code:: bash

    source activate hoomd-tf


Continue following the compiling steps below to complete install.
The simple approach is recommended but **use the following
different cmake step**

.. code:: bash

  export CMAKE_PREFIX_PATH=/path/to/environment
  CXX=g++ CC=gcc cmake ..

If using the hoomd-blue compilation, **use the following
different cmake step**

.. code:: bash

    export CMAKE_PREFIX_PATH=/path/to/environment
    CXX=g++ CC=gcc cmake .. \
    -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DENABLE_MPI=OFF -DBUILD_HPMC=off -DBUILD_CGCMM=off -DBUILD_MD=on \
    -DBUILD_METAL=off -DBUILD_TESTING=off -DBUILD_DEPRECATED=off -DBUILD_MPCD=OFF \
    -DCMAKE_INSTALL_PREFIX=`python -c "import site; print(site.getsitepackages()[0])"`\
    -DNVCC_FLAGS="-ccbin /software/gcc/7.3.0/bin"

.. _optional_dependencies:

Optional Dependencies
----------------------------
Following packages are optional:

.. code:: bash

   MDAnalysis

:py:class:`utils.iter_from_trajectory` uses `MDAnalysis` for trajectory parsing


.. |ss| raw:: html

   <strike>

.. |se| raw:: html

   </strike>&nbsp;