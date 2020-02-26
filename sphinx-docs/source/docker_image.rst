.. _docker_image:

Docker Image for Development
============================

To use the included docker image:

.. code:: bash

    docker build -t hoomd-tf htf

To run the container:

.. code:: bash

    docker run --rm -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
     -v /insert/path/to/htf/:/srv/hoomd-blue/htf hoomd-tf bash

The ``cap--add`` and ``security-opt`` flags are optional and allow
``gdb`` debugging. Install ``gdb`` and ``python3-dbg`` packages to use
``gdb`` with the package.

Once in the container:

.. code:: bash

    cd /srv/hoomd-blue && mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Debug\
        -DENABLE_CUDA=OFF -DENABLE_MPI=OFF -DBUILD_HPMC=off\
         -DBUILD_CGCMM=off -DBUILD_MD=on -DBUILD_METAL=off \
        -DBUILD_TESTING=off -DBUILD_DEPRECATED=off -DBUILD_MPCD=OFF
    make -j2
