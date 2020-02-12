.. _mbuild_environment:

MBuild Environment
==================

If you are using mbuild, please follow these additional install steps:

.. code:: bash

    conda install numpy cython
    pip install requests networkx matplotlib scipy pandas plyplus lxml mdtraj oset
    conda install -c omnia -y openmm parmed
    conda install -c conda-forge --no-deps -y packmol gsd
    pip install --upgrade git+https://github.com/mosdef-hub/foyer git+https://github.com/mosdef-hub/mbuild
