---
title: 'HOOMD-TF: GPU-Accelerated, Online Machine Learning in the HOOMD-blue Molecular Dynamics Engine'
tags:
  - Python
  - molecular dynamics
  - machine learning
authors:
  - name: Rainier Barrett
    orcid: 0000-0002-5728-9074
    affiliation: 1
  - name: Maghesree Chakraborty
    orcid: 0000-0001-5706-3027
    affiliation: 1
  - name: Dilnoza B Amirkulova
    orcid: 0000-0001-6961-3081
    affiliation: 1
  - name: Heta A Gandhi
    orcid: 0000-0002-9465-3840
    affiliation: 1
  - name: Geemi Wellawatte
    orcid: 0000-0002-3772-6927
    affiliation: 1
  - name: Andrew D White
    orcid: 0000-0002-6647-3965
    affiliation: 1
affiliations:
 - name: University of Rochester
   index: 1
date: 27 February 2020
bibliography: paper.bib
---

# Statement of Need

Machine learning (ML) is emerging as an essential tool in the molecular
modeling community. The most mature applications are in batch learning with data generated from molecular modeling calculations.
For example, one can run a molecular dynamics (MD) or density functional theory (DFT) simulation to completion, from which the trajectory is then used as
input data to train a deep neural network that reproduces the energetics at a fraction of the computational cost.
Some recent examples include an energy-conserving force-field learned with a custom gradient-domain
model [@ChmielaConservedEnergyMLFF2017], DFT-based neural network force-fields [@SmithDFTNNPotential2017],
and a neural network coarse-grained potential [@WangCGML2019].
A limitation of these methods is that they treat the molecular modeling calculations as a static dataset, whereas molecular modeling calculations can be treated as interrogative functions, opening the door to methods like reinforcement learning or active learning. Thus there is a clear limitation of existing implementations due to their sequential nature of going from calculation to ML. Another practical issue is that neural network force-field implementations often duplicate standard ML frameworks, preventing them from keeping progress with state-of-the art methods. For example [@RuppMLAtomizationE2012] and [@BotuMLQMMD] are benchmark works in this field and custom implementations. This limits the scope and speed of translating ML advances to the molecular modeling community.


This need has led us to develop HOOMD-TF, a flexible direct integration of a standard ML library and standard molecular simulation framework that maintains GPU acceleration. Our goals are to improve the reproducibility of ML methods in molecular simulation, ease translation of ML advances, and remove the need for sequential simulation and ML. This should enable active learning, reinforcement learning, and online learning of molecular simulations. 

There are other applications focusing on ML in molecular modeling, such as [DeepChem](https://www.deepchem.io/) and [@Aspuru-GuzikMaterialsDiscovery2015],
which are largely concerned with property prediction and representation. There is also a similar work to HOOMD-TF by
[@EastmanOpenMMNN2018], [OpenMM-NN](https://github.com/openmm/openmm-nn), which allows the use of pre-trained TensorFlow models in [OpenMM](http://openmm.org/) [@PandeOpenMM2013]. In contrast,
this work fills the niche of online model training, while also allowing pre-trained model imports in MD simulation,
coarse-grained force-field learning, collective variable calculation and manipulation, and force-field biasing.

# Summary

The HOOMD-TF package pairs the TensorFlow ML library [@tensorflow2015whitepaper] with the HOOMD-blue
simulation engine [@AndersonHOOMD2019] to allow for flexible online ML and tensor calculations 
during HOOMD-blue simulations. Since both TensorFlow and HOOMD-blue are GPU-accelerated, HOOMD-TF
was designed with a GPU-GPU communication scheme that minimizes 
latency between GPU memory to preserve execution speed.

HOOMD-TF enables online ML in MD simulations with the support of the
suite of tools available through TensorFlow. It can be used for force matching,
calculation of arbitrary collective variables (CVs), force-field biasing or learning using said CVs, and analysis using tensor calculations. These tasks can be performed either online during a simulation or offline using a saved trajectory. This is accomplished by using TensorFlow tensors 
filled with particle positions and neighbor lists from HOOMD-blue. This also allows the use of TensorFlow's
derivative propagation to perform biasing with arbitrary CVs, provided that they can be expressed as a tensor operation of either the neighbor list or particle positions. Another application of this software is learning coarse-grained force-fields with either
neural networks or other ML models. The ability to run force matching calculations online makes
the coarse-graining workflow straightforward in HOOMD-TF. Since HOOMD-blue can use external force-fields and TensorFlow
can learn as the simulation is running, learning can be terminated as soon as the force-matching algorithm converges,
requiring only one simulation iteration.

HOOMD-TF uses TensorFlow to save and load models, and is therefore compatible with pre-trained TensorFlow models. TensorFlow's TensorBoard
utility can also be used to track and examine model training and performance. HOOMD-TF can be used independent of HOOMD-blue by using trajectories via the MDAnalysis framework [@MDAnalysis2011; @MDAnalysis2016]. This allows for previously-trained TensorFlow
models to be used on trajectories that were produced by other MD engines, analysis of new CVs
from a previously-run simulation, and training of models from trajectories.

Overall, HOOMD-TF makes online ML in MD simulations possible with little additional effort, and
eases the use of TensorFlow models on MD trajectories for both machine learning and analysis.
The ability to tightly integrate trained ML models in HOOMD-TF can enable their use in simulations 
by removing the need for custom implementations and improve reproducibility in the field. The online functionality of HOOMD-TF enables the use of simulations as interrogable models rather than static data generators, allowing direct use in an active and/or reinforcement learning framework.
TensorFlow computation graphs allow for transparent and simple model designation with a high
 degree of customizability, replicability, and efficiency.

# Accessing the Software

HOOMD-TF is freely available under the MIT license on [github](https://github.com/ur-whitelab/hoomd-tf). 
The documentation is hosted on [readthedocs.io](https://hoomd-tf.readthedocs.io/en/latest/).


# Acknowledgements

We would like to thank Joshua Anderson and Jens Glaser for instructive conversations about the HOOMD-blue architecture. We thank the Center for Integrated Research Computing (CIRC) at the University of Rochester for providing computational resources and technical support. This work was supported by the National Science Foundation (CBET‐1751471  and CHE-1764415).

# References