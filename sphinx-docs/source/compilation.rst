.. _bluehive_installation:

BlueHive Installation
=====================

**Feeling Lucky?** Try this for quick results

.. code:: bash

    module load tensorflow/1.15.0/b1 git cmake
    conda create -n hoomd-tf python=3.7
    source activate hoomd-tf
    python -m pip install tensorflow-gpu==1.15.0
    conda install -c conda-forge hoomd==2.5.2
    git clone https://github.com/ur-whitelab/hoomd-tf
    cd hoomd-tf && mkdir build && cd build
    CXX=g++ CC=gcc cmake ..
    make install
    cd .. && python htf/test-py/test_sanity.py

After cloning the ``hoomd-tf`` repo, follow these steps:

Load the modules necessary:

.. code:: bash

    module load tensorflow/1.15.0/b1 git cmake

Set-up virtual python environment *ONCE* to keep packages isolated.

.. code:: bash

    conda create -n hoomd-tf python=3.7
    source activate hoomd-tf
    python -m pip install tensorflow-gpu==1.15.0

Then whenever you login and *have loaded modules*:

.. code:: bash

    source activate hoomd-tf


Continue following the compling steps below to complete install.
The simple approach is recommended but **use the following
different cmake string**

.. code:: bash

  CXX=g++ CC=gcc cmake ..

If using the hoomd-blue compilation, **use the following
different cmake string**

.. code:: bash

    CXX=g++ CC=gcc cmake .. \
    -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DENABLE_MPI=OFF -DBUILD_HPMC=off -DBUILD_CGCMM=off -DBUILD_MD=on \
    -DBUILD_METAL=off -DBUILD_TESTING=off -DBUILD_DEPRECATED=off -DBUILD_MPCD=OFF \
    -DCMAKE_INSTALL_PREFIX=`python -c "import site; print(site.getsitepackages()[0])"`\
    -DNVCC_FLAGS="-ccbin /software/gcc/7.3.0/bin" -DCUDA_ARCH_LIST=70

.. _compiling:

Compiling
=========

The following packages are required to compile:

::

    tensorflow < 2.0
    hoomd-blue >= 2.5.0
    numpy
    tbb-devel (only for hoomd-blue 2.8 and above)

tbb-devel is only required if using the "simple" method below and are
using hoomd-blue 2.8 or above. The tensorflow versions should be any
Tensorflow 1 release. The higher versions, like 1.14, 1.15, will give
lots of warnings about migrating code to Tensorflow 2.0. It is
recommended you install via pip:

.. code:: bash

  pip install tensorflow-gpu==1.15.0

.. _simple_compiling:

Simple Compiling
----------------

This method assumes you already have installed hoomd-blue and
tensorflow. You could do that, for example, via ``conda install -c
conda-forge hoomd==2.5.2 tbb-devel``. Remember that pip is recommneded for installing
tensorflow. Here are steps **after** installing hoomd-blue

.. code:: bash

    git clone https://github.com/ur-whitelab/hoomd-tf
    cd hoomd-tf && mkdir build && cd build
    cmake ..
    make install

That's it! Make sure you have a GCC compiler consistent with the
tensorflow version you have installed (assuming you installed
tensorflow via pip). To see your tensorflow GCC compiler, try
`python -c 'import tensorflow;
print(tensorflow.__compiler_version__)'`

.. _compiling_with_hoomd_blue:

Compiling with Hoomd-Blue
-------------------------

Use this method if you need to compile with developer flags on or other
special requirements.

.. code:: bash

    git clone --recursive https://bitbucket.org/glotzer/hoomd-blue hoomd-blue

We typically use v2.5.2 of hoomd-blue

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
    cmake .. -DCMAKE_BUILD_TYPE=Release \
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

    cmake .. \
    -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DENABLE_MPI=OFF -DBUILD_HPMC=off -DBUILD_CGCMM=off -DBUILD_MD=on \
    -DBUILD_METAL=off -DBUILD_TESTING=off -DBUILD_DEPRECATED=off -DBUILD_MPCD=OFF \
    -DCMAKE_INSTALL_PREFIX=`python -c "import site; print(site.getsitepackages()[0])"`

.. _updating_compiled_code:

Updating Compiled Code
----------------------

If you are developing frequently, add the build directory to your
python path instead of `make install` (only works with hoomd-blue
compiled). Then if you modify C++ code, only run make (not cmake). If
you modify python, just copy over py files (``htf/*py`` to
``build/hoomd/htf``).

.. _mbuild_environment:

MBuild Environment
------------------

If you are using mbuild, please follow these additional install steps:

.. code:: bash

    conda install numpy cython
    pip install requests networkx matplotlib scipy pandas plyplus lxml mdtraj oset
    conda install -c omnia -y openmm parmed
    conda install -c conda-forge --no-deps -y packmol gsd
    pip install --upgrade git+https://github.com/mosdef-hub/foyer git+https://github.com/mosdef-hub/mbuild
