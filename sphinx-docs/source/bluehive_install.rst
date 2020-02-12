.. _bluehive_installation:

BlueHive Installation
=====================

After cloning the ``hoomd-tf`` repo, follow these steps:

Load the modules necessary:

.. code:: bash

    module load git anaconda3/2018.12b cmake sqlite cudnn/9.0-7

Set-up virtual python environment *ONCE* to keep packages isolated.

.. code:: bash

    conda create -n hoomd-tf python=3.6

Then whenever you login and *have loaded modules*:

.. code:: bash

    source activate hoomd-tf

Now that Python is ready, install some pre-requisites:

.. code:: bash

    pip install tensorflow-gpu==1.12

Continue following the compling steps below to complete install.

