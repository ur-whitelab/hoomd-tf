# Hoomd-TF

This plugin enables the use of TensorFlow to compute forces in a Hoomd-blue simulation. You can also compute other quantities, like collective variables, and do learning.

Table of Contents
=================

   * [Quickstart Tutorial](#quickstart-tutorial)
   * [Building the Graph](#building-the-graph)
      * [Molecule Batching](#molecule-batching)
      * [Computing Forces](#computing-forces)
      * [Neighbor Lists](#neighbor-lists)
      * [Virial](#virial)
      * [Finalizing the Graph](#finalizing-the-graph)
      * [Printing](#printing)
      * [Period of out nodes](#period-of-out-nodes)
      * [Variables and Restarts](#variables-and-restarts)
      * [Saving and Loading Variables](#saving-and-loading-variables)
      * [Optional: Keras Layers for Model Building](#optional-keras-layers-for-model-building)
      * [Complete Examples](#complete-examples)
      * [Lennard-Jones with 1 Particle Type](#lennard-jones-with-1-particle-type)
   * [Using a Graph in a Simulation](#using-a-graph-in-a-simulation)
      * [Batching](#batching)
      * [Bootstraping Variables](#bootstraping-variables)
         * [Bootstrapping Variables from Other Models](#bootstrapping-variables-from-other-models)
   * [Utilities](#utilities)
      * [RDF](#rdf)
      * [Pairwise Potential and Forces](#pairwise-potential-and-forces)
      * [Biasing with EDS](#biasing-with-eds)
   * [Coarse-Graining Utilities](#coarse-graining-utilities)
      * [Find Molecules](#find-molecules)
      * [Sparse Mapping](#sparse-mapping)
      * [Center of Mass](#center-of-mass)
      * [Compute Mapped Neighbor List](#compute-mapped-neighbor-list)
   * [Tensorboard](#tensorboard)
      * [Saving Scalars in Tensorboard](#saving-scalars-in-tensorboard)
      * [Viewing when TF is running on remote server](#viewing-when-tf-is-running-on-remote-server)
      * [Viewing when TF is running in container](#viewing-when-tf-is-running-in-container)
   * [Interactive Mode](#interactive-mode)
   * [Docker Image for Development](#docker-image-for-development)
   * [Tests](#tests)
   * [Bluehive Install](#bluehive-install)
   * [Compiling](#compiling)
      * [Conda Environments](#conda-environments)
      * [Updating Compiled Code](#updating-compiled-code)
   * [MBuild Environment](#mbuild-environment)
   * [Running on Bluehive](#running-on-bluehive)
   * [Known Issues](#known-issues)
      * [Using Positions](#using-positions)
      * [Exploding Gradients](#exploding-gradients)
         * [Small Training Rates](#small-training-rates)
         * [Safe Norm](#safe-norm)
         * [Clipping Gradients](#clipping-gradients)
      * [Neighbor Lists](#neighbor-lists)

# Quickstart Tutorial

To compute a `1 / r` pairwise potential with Hoomd-TF:

```python
import hoomd.htf as htf
import tensorflow as tf

########### Graph Building Code ###########
graph = htf.graph_builder(64) # max neighbors = 64
pair_energy = graph.nlist_rinv # nlist_rinv is neighbor 1 / r
particle_energy = tf.reduce_sum(pair_energy, axis=1) # sum over neighbors
forces = graph.compute_forces(energy) # compute forces
graph.save('my_model', forces)

########### Hoomd-Sim Code ################
hoomd.context.initialize()
# this will start TensorFlow, so it goes
# in a with statement for clean exit
with htf.tfcompute('my_model') as tfcompute:
    # create a square lattice
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=4.0),
                                        n=[3,3])
    nlist = hoomd.md.nlist.cell()
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.nve(group=hoomd.group.all())
    tfcompute.attach(nlist, r_cut=3.0)
    hoomd.run(1e3)
```

This creates a computation graph whose energy function is `2 / r` and also computes forces and virial for the simulation. The `2` is because the neighborlists in Hoomd-TF are *full* neighborlists (double counted). The Hoomd-blue code starts a simulation of a 9 particle square lattice and simulates it for 1000 timesteps under the potential defined in our Hoomd-TF model. The general process of using Hoomd-TF is to build a TensorFlow computation graph, load the graph, and then attach the graph. See below for more detailed information about Hoomd-TF.

# Building the Graph

To construct a graph, create a `graph_builder`:

```python
import hoomd.htf as htf
graph = htf.graph_builder(NN, output_forces)
```

where `NN` is the maximum number of nearest neighbors to consider (can be 0) and
`output_forces` indicates if the graph will output forces to use in
the simulation. After building the `graph`, it will have three tensors
as attributes to use in constructing the TensorFlow graph: `nlist`,
`positions`, `box`, `box_size`, and `forces`. `nlist` is an `N` x `NN` x 4 tensor
containing the nearest neighbors. An entry of all zeros indicates that
less than `NN` nearest neighbors where present for a particular
particle. The 4 right-most dimensions are `x,y,z` and `w`, which is
the particle type. Particle type is an integer starting at 0. Note
that the `x,y,z` values are a vector originating at the particle and
ending at its neighbor. `positions` and `forces` are `N` x 4
tensors. `forces` *only* is available if the graph does not output
forces via `output_forces=False`. `box` is a 3x3 tensor containing the low box coordinate, 
high box coordinate, and then tilt factors. `box_size` contains just the box length
in each dimension. 

## Molecule Batching

It may be simpler to have positions or neighbor lists or forces arranged by molecule.
For example, you may want to look at only a particular bond or subset of atoms in a molecule.
To do this, you can call `graph.build_mol_rep(MN)`, where `MN` is the maximum number of
atoms in a molecule. This will create the following new attributes: `mol_positions`,
`mol_nlist`, and `mol_forces` (if your graph has `output_forces=False`). These new
attributes are dimension `M x MN x ...`  where `M` is the number of molecules and
`MN` is the atom index within the molecule. If your molecule has less than `MN`,
extra entries will be zeros. You can defnie a molecule to be whatever you want and
atoms need not be only in one molecule. Here's an example to compute a water angle,
where I'm assuming that the oxygens are the middle atom.

```python
import hoomd.htf as htf
graph = htf.graph_builder(0)
graph.build_mol_rep(3)
# want slice for all molecules (:)
# want h1 (0), o (1), h2(2)
# positions are x,y,z,w. We only want x,y z (:3)
v1 = graph.mol_positions[:, 2, :3] - graph.mol_positions[:, 1, :3]
v2 = graph.mol_positions[:, 0, :3] - graph.mol_positions[:, 1, :3]
# compute per-molecule dot product and divide by per molecule norm
c = tf.einsum('ij,ij->i', v1, v2) / tf.norm(v1, axis=1) / tf.norm(v2 axis=1)
angles = tf.math.acos(c)
```

## Computing Forces

If your graph is outputting forces, you may either compute forces and pass them to `graph_builder.save(...)` or have them computed via automatic differentiation of a potential energy. Call `graph_builder.compute_forces(energy)` where `energy` is a scalar or tensor that depends on `nlist` and/or `positions`. A tensor of forces will be returned as sum(-dE / dn) - dE / dp where the sum is over the neighbor list. For example, to compute a `1 / r` potential:

```python
graph = htf.graph_builder(N - 1)
#remove w since we don't care about types
nlist = graph.nlist[:, :, :3]
#get r
r = tf.norm(nlist, axis=2)
#compute 1. / r while safely treating r = 0.
# halve due to full nlist
rij_energy = 0.5 * graph.safe_div(1, r)
#sum over neighbors
energy = tf.reduce_sum(rij_energy, axis=1)
forces = graph.compute_forces(energy)
```

See in the above example that we have used the
`graph_builder.safe_div(numerator, denominator)` function which allows
us to safely treat a `1 / 0` due to using nearest neighbor distances,
which can arise because `nlist` contains 0s for when less than `NN`
nearest neighbors are found. Note that because `nlist` is a *full*
neighbor list, you should divide by 2 if your energy is a sum of
pairwise energies.

## Neighbor lists

As mentioned above, there is `graph.nlist`, which is an `N x NN x 4` neighobr lists. You can
access masked versions of this with `graph.masked_nlist(self, type_i=None, type_j=None, nlist=None, type_tensor=None)`
where `type_i/type_j` are optional integers that specify the type of the origin (`type_i`) or neighobr (`type_j`). The `nlist` argument
allows you to pass in your own neighbor list and `type_tensor` allows you to specify your own list of types,
if different than what is given by hoomd-blue. You can also access `graph.nlist_rinv` which gives a pre-computed
`1 / r` `N x NN` matrix.

## Virial

The virial is computed and added to the graph if you use the
`compute_forces` function and your energy has a non-zero derivative
with respect to `nlist`. You may also explicitly pass the virial when
saving, or pass `None` to remove the automatically calculated virial.

## Finalizing the Graph

To finalize and save your graph, you must call the `graph_builder.save(directory, force_tensor=forces, virial = None, out_node=None)` function. `force_tensor` should be your computed forces, either as computed by your graph or as the output from `compute_energy`. If your graph is not outputting forces, then you must provide a tensor which will be computed, `out_nodes`, at each timestep. Your forces should be an `N x 4` tensor with the 4th column indicating per-particle potential energy. The virial should be an `N x 3 x 3` tensor.

## Printing

If you would like to print out the values from nodes in your graph, you can
add a print node to the `out_nodes`. For example:

```python
...graph building code...
forces = graph.compute_forces(energy)
print_node = tf.Print(energy, [energy], summarize=1000)
graph.save(force_tensor=forces, model_directory=name, out_nodes=[print_node])
```

The `summarize` keyword sets the maximum number of numbers to print. Be wary of printing thousands of numbers per step.

## Period of out nodes

You can modify how often tensorflow is called via the `tfcompute.attach` command. You can also have more granular control of operations/tensors passed to `out_nodes` by changing the type to a list whose first element is the tensor and the second argument is the period at which it is computed. For example:

```python
...graph building code...
forces = graph.compute_forces(energy)
avg_force = tf.reduce_mean(forces, axis=-1)
print_node = tf.Print(energy, [energy], summarize=1000)
graph.save(force_tensor=forces, model_directory=name, out_nodes=[[print_node, 100], [avg_force, 25]])
```

This will print the energy every 100 steps and compute the average force every 25 steps (although it is unused). Note that these two ways of affecting period both apply. So if the above graph was attached with `tfcompute.attach(..., period=25)` then the `print_node` will be computed every 2500 steps. 

## Variables and Restarts

In TensorFlow, variables are trainable parameters. They are required parts of your graph when doing learning. Each `saving_period` (set as arg to `tfcompute.attach`), they are written to your model directory. Note that when a run is started, the latest values of your variables are loaded from your model directory. *If you are starting a new run but you previously ran your model, the old variable values will be loaded.* To prevent this unexpectedly loading old checkpoints, if you run `graph_builder.save` it will move out all old checkpoints. This behavior means that if you want to restart, you should not re-run `graph_builder.save` in your restart script *or* pass `move_previous = False` as a parameter if you re-run `graph_builder.save`.

Variables are how you can save data. They can be accumulated between steps. Be sure to set them to be `trainable=False` if you are also doing learning but would like to accumulate in variables. For example, you can have a variable for running mean. You can load these variables with the `htf.load_variables` command. See next section for details.

## Saving and Loading Variables

`graph_builder` has a convenience function to compute the running mean of some property:

```python
# set-up graph to compute energy
...
# we name our variable avg-energy
graph.running_mean(energy, 'avg-energy')
# run the simulation
...
```

You may then load the variable after the simulation using the following syntax, which creates a dictionary with entries [`avg-energy`].

```python
variables  = htf.load_variables(model_dir, ['avg-energy'])
```

The `load_variables` is general and can be used to load trained, non-trained, or averaged variables.

## Optional: Keras Layers for Model Building

Currently HOOMD-TF supports Keras layers in model building. We do not yet support Keras `Model.compile()` or `Model.fit()`. This example shows how to set up a neural network model using Keras layers.

```python
import tensorflow as tf
from tensorflow.keras import layers
import hoomd.htf as htf

NN = 64
N_hidden_nodes = 5
graph = htf.graph_builder(NN, output_forces=False)
r_inv = graph.nlist_rinv
input_tensor = tf.reshape(r_inv, shape=(-1,1), name='r_inv')
#we don't need to explicitly make a keras.Model object, just layers
input_layer = layers.Input(tensor=input_tensor)
hidden_layer = layers.Dense(N_hidden_nodes)(input_layer)
output_layer = layers.Dense(1, input_shape=(N_hidden_nodes,))(hidden_layer)
#do not call Model.compile, just use the output in the TensorFlow graph
nn_energies = tf.reshape(output_layer, [-1, NN])
calculated_energies = tf.reduce_sum(nn_energies, axis=1, name='calculated_energies')
calculated_forces = graph.compute_forces(calculated_energies)
#cost and optimizer must also be set through TensorFlow, not Keras
cost = tf.losses.mean_squared_error(calculated_forces, graph.forces)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
#save using graph.save, not Keras Model.compile
graph.save(model_directory='/tmp/keras_model/', out_nodes=[ optimizer])

```

The model can then be loaded and trained as normal. Note that `keras.models.Model.fit()` is not currently supported. You must train using `htf.tfcompute()` as explained in the next section.

## Complete Examples

See [htf/models](htf/models)

## Lennard-Jones with 1 Particle Type

```python
graph = hoomd.htf.graph_builder(NN)
#use convenience rinv
r_inv = graph.nlist_rinv
p_energy = 4.0 / 2.0 * (r_inv**12 - r_inv**6)
#sum over pairwise energy
energy = tf.reduce_sum(p_energy, axis=1)
forces = graph.compute_forces(energy)
graph.save(force_tensor=forces, model_directory='/tmp/lj-model')
```

# Using a Graph in a Simulation

You may use a saved TensorFlow model via:

```python
import hoomd, hoomd.md
import hoomd.htf as htf

...hoomd initialization code...
with htf.tfcompute(model_dir) as tfcompute:

    nlist = hoomd.md.nlist.cell()
    tfcompute.attach(nlist, r_cut=3)

    ...other hoomd code...

    hoomd.run(...)

```

where `model_dir` is the directory where the TensorFlow model was saved, `nlist` is a hoomd neighbor list object and `r_cut` is the maximum distance for to consider particles as being neighbors. `nlist` is optional and is not required if your graph doesn't use the `nlist` object (you passed `NN = 0` when building your graph).

## Batching

If you used per-molecule positions or nlist in your graph, you can either
rely on hoomd-tf to find your molecules by traversing the bonds in your system (default)
or you can specify what are molecules in your system. They are passed via `attach(..., mol_indices=[[..]])`. The `mol_indices` are a, possibly ragged, 2D python list where each
element in the list is a list of atom indices for a molecule. For example, `[[0,1], [1]]` means
that there are two molecules with the first containing atoms 0 and 1 and the second containing atom 1. Note that the molecules can be different size and atoms can exist in multiple molecules.

If you do not call `build_mol_rep` while building your graph, you can optionally split your batches to be smaller than the entire system. This is set via the `batch_size` integer argument to `attach`. This can help for high-memory systems where you cannot spare the GPU memory to have each tensor be the size of your system.

## Bootstraping Variables

If you have trained variables previously and would like to load them into the current TensorFlow graph, you can use the `bootstrap` and `bootstrap_map` arguments. `bootstrap` should be a checkpoint file path or model directory path (latest checkpoint is used) containing variables which can be loaded into your tfcompute graph. Your model will be built, then all variables will be initialized, and then your bootstrap checkpoint will be loaded and no variables will be reloaded even if there exists a checkpoint in the model directory (to prevent overwriting your bootstrap variables). `bootstrap_map` is an optional additional argument that will have keys that are variable names in the `bootstrap` checkpoint file and values that are names in the tfcompute graph. This can be used when your variable names do not match up. Here are two example demonstrating with and without a `bootstrap_map`:

Here's an example that creates some variables that could be trained offline without Hoomd. In this example, they just use their initial values.

```python
import tensorflow as tf

#make some variables
v = tf.Variable(8.0, name='epsilon')
s = tf.Variable(2.0, name='sigma')

#initialize and save them
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, '/tmp/bootstrap/model')
```

We load them in the hoomd run script:
```python
with hoomd.htf.tfcompute(model_dir,
    bootstrap='/tmp/bootstrap/model') as tfcompute:
    ...
```

Here's how we would load them in the hoomd run script if we want to change
the names of the variables:
```python
# here the pretrained variable parameters will replace variables with a different name
with hoomd.htf.tfcompute(model_dir,
    bootstrap='/tmp/bootstrap/model',
    bootstrap_map={'lj-epsilon':'epsilon', 'lj-sigma':'sigma'}) as tfcompute:
    ...
```

### Bootstrapping Variables from Other Models

Here's an example of bootstrapping where you train with Hoomd-TF and then load the variables into a different model:

```python
# build_models.py
import tensorflow as tf
import hoomd.htf as htf

def make_train_graph(NN, directory):
    # build a model that fits the energy to a linear term
    graph = htf.graph_builder(NN, output_forces=False)
    # get r
    nlist = graph.nlist[:, :, :3]
    r = graph.safe_norm(nlist, axis=2)
    # build energy model
    m = tf.Variable(1.0, name='m')
    b = tf.Variable(0.0, name='b')
    predicted_particle_energy = tf.reduce_sum(m * r + b, axis=1)
    # get energy from hoomd
    particle_energy = graph.forces[:, 3]
    # make them match
    loss = tf.losses.mean_squared_error(particle_energy, predicted_particle_energy)
    optimize = tf.train.AdamOptimizer(1e-3).minimize(loss)
    graph.save(model_directory=directory, out_nodes=[optimize])

def make_force_graph(NN, directory):
    # this model applies the variables learned in the example above
    # to compute forces
    graph = htf.graph_builder(NN)
    # get r
    nlist = graph.nlist[:, :, :3]
    r = graph.safe_norm(nlist, axis=2)
    # build energy model
    m = tf.Variable(1.0, name='m')
    b = tf.Variable(0.0, name='b')
    predicted_particle_energy = tf.reduce_sum(m * r + b, axis=1)
    forces = graph.compute_forces(predicted_particle_energy)
    graph.save(force_tensor=forces, model_directory=directory)
make_train_graph(64, 16, '/tmp/training')
make_force_graph(64, 16, '/tmp/inference')
```

Here is how we run the training model:
```python
#run_train.py
import hoomd, hoomd.md
import hoomd.htf as htf


hoomd.context.initialize()

with htf.tfcompute('/tmp/training') as tfcompute:
    rcut = 3.0
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                       n=[8,8])
    nlist = hoomd.md.nlist.cell(check_period = 1)
    lj = hoomd.md.pair.lj(rcut, nlist)
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.nve(
        group=hoomd.group.all()).randomize_velocities(kT=0.2, seed=42)

    tfcompute.attach(nlist, r_cut=rcut)
    hoomd.run(100)
```

Load the variables trained in the training run into the model which computes forces:

```python
#run_inference.py
import hoomd, hoomd.md
import hoomd.htf as htf

hoomd.context.initialize()
with htf.tfcompute('/tmp/inference',
        bootstrap='/tmp/training') as tfcompute:
    rcut = 3.0
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                       n=[8,8])
    nlist = hoomd.md.nlist.cell(check_period = 1)
    #notice we no longer compute forces with hoomd
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.nve(
        group=hoomd.group.all()).randomize_velocities(kT=0.2, seed=42)

    tfcompute.attach(nlist, r_cut=rcut)
    hoomd.run(100)
```

# Utilities

There are a few convenience functions in `hoomd.htf` and the `graph_builder` class for plotting potential energies of pairwise potentials and constructing CG mappings.

## RDF

To compute an RDF, use the `graph.compute_rdf(...)` method:

```python
# set-up graph to compute energy
...
rdf = graph.compute_rdf([1,10], 'rdf', nbins=200)
graph.running_mean(rdf, 'avg-rdf')
# run the simulation
...
variables  = htf.load_variables(model_dir, ['avg-rdf'])
print(variables)
```
## Pairwise Potential and Forces

To compute pairwise potential, use the `graph.compute_pairwise_potential(...)` method:

```python
...
r = numpy.arange(1, 10, 1)
potential, forces = htf.compute_pairwise_potential('/path/to/model', r, potential_tensor)
...
```

## Biasing with EDS

To apply [Experiment Directed Simulation](https://www.tandfonline.com/doi/full/10.1080/08927022.2019.1608988) biasing to a system:

```python
eds_alpha = htf.eds_bias(cv, set_point=3.0, period=100)
eds_energy = eds_alpha * cv
eds_forces = graph.compute_forces(eds_energy)
graph.save('eds-graph', eds_forces)
```

where `htf.eds_bias(cv, set_point, period, learning_rate, cv_scale, name)` is the function that computes your lagrange multiplier/eds coupling that you use to bias your simulation. It may be useful 
to also take the average of `eds_alpha` so that you can use it in a subsequent simulation:

```python
avg_alpha = graph.running_mean(eds_alpha, name='avg-eds-alpha')
.....
# after simulation
vars = htf.load_variables('model/directory', ['avg-eds-alpha'])
print(vars['avg-eds-alpha'])
```

# Coarse-Graining Utilities

## Find Molecules
To go from atom index to particle index, use the `hoomd.htf.find_molecules(...)` method:
```python
# The method takes in a hoomd system as an argument.
...
molecule_mapping_index = hoomd.htf.find_molecules(system)
...

```

## Sparse Mapping

The `sparse_mapping(...)` method creates the necessary indices and values for defining a sparse tensor in tensorflow that is a mass-weighted MxN mapping operator where M is the number of coarse-grained particles and N is the number of atoms in the system. In the example,`mapping_per_molecule` is a list of k x n matrices where k is the number of coarse-grained sites for each molecule and n is the number of atoms in the corresponding molecule. There should be one matrix per molecule. Since the example is for a 1 bead mapping per molecule the shape is 1 x n. The ordering of the atoms should follow the output from the find_molecules method. The variable `molecule_mapping_index` is the output from the `find_molecules(...)` method.

```python
#The example is shown for 1 coarse-grained site per molecule.
...
molecule_mapping_matrix = numpy.ones([1, len(molecule_mapping_index[0])], dtype=np.int)
mapping_per_molecule = [molecule_mapping_matrix for _ in molecule_mapping_index]
cg_mapping = htf.sparse_mapping(mapping_per_molecule, \
	     			molecule_mapping_index, system = system)
...
```

## Center of Mass

The `center_of_mass(...)` method maps the given positions according to the specified mapping operator to coarse-grain site positions considering periodic boundary condition. The coarse grain site position is placed at the center of mass of its constituent atoms.

```python

...
mapped_position = htf.center_of_mass(graph.positions[:,:3], cg_mapping, system)
#cg_mapping is the output from the sparse_matrix(...) method and indicates how each molecule is mapped.
...

```

## Compute Mapped Neighbor List
The `compute_nlist(...)` method returns the neighbor list for the mapped coarse-grained particles. In the example, `mapped_position` is the mapped particle positions obeying the periodic boundary condition as returned by the `center_of_mass(...) method`, `rcut` is the cut-off radius and `NN` is the number of nearest neighbors to be considered for the coarse-grained system.
```python
...
mapped_nlist= htf.compute_nlist(mapped_position, rcut, NN, system)
...

```

# Tensorboard

You can visualize your models with tensorboard. First, add
`write_tensorboard=True` the TensorFlow plugin constructor. This will
add a new directory called `tensorboard` to your model directory.

After running, you can launch tensorboard like so:

```bash
tensorboard --logdir=/path/to/model/tensorboard
```

and then visit `http://localhost:6006` to view the graph.

## Saving Scalars in Tensorboard

If you would like to save a scalar over time, like total energy or training loss, you can use the Tensorboard functionality. Add scalars to the Tensorboard summary during the build step:

```python
tf.summary.scalar('total-energy', tf.reduce_sum(particle_energy))
```

and then add the `write_tensorboard=True` flag during the `tfcompute` initialize. The period of tensorboard writes is controlled by the `saving_period` flag to the `tfcompute.attach` command. View the Tensorboard section below to see how to view the resulting scalars.

## Viewing when TF is running on remote server

If you are running on a server, before launching tensorboard use this ssh command to login:

```bash
ssh -L 6006:[remote ip or hostname]:6006 username@remote
```

and then you can view after launching on the server via your local web browser.

## Viewing when TF is running in container

If you are running docker, you can make this port available a few different ways. The first is
to get the IP address of your docker container (google how to do this if not default), which is typically `172.0.0.1`, and then
visit `http://172.0.0.1:6006` or equivalent if you have a different IP address for your container.

The second option is to use port forwarding. You can add a port forward flag, `-p 6006:6006`, when running the container which
will forward traffic from your container's 6006 port to the host's 6006 port. Again, then you can visit `http://localhost:6006` (linux)
or `http://127.0.0.1:6006` (windows).

The last method, which usually works when all others fail, is to have all the container's traffic be on the host. You can do this by
adding the flag `--net=host` to the run command of the container. Then you can visit  `http://localhost:6006`.

# Interactive Mode

Experimental, but you can trace your graph in realtime in a simulation. Add both the `write_tensorboard=True` to
the constructor and the `_debug_mode=True` flag to `attach` command. You then open another shell and connect by following
the online instructions for interactive debugging from Tensorboard.

# Docker Image for Development

To use the included docker image:

```bash
docker build -t hoomd-tf htf
```

To run the container:

```bash
docker run --rm -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
 -v /insert/path/to/htf/:/srv/hoomd-blue/htf hoomd-tf bash
```

The `cap--add` and `security-opt` flags are optional and allow `gdb`
debugging. Install `gdb` and `python3-dbg` packages to use `gdb` with
the package.

Once in the container:

```bash
cd /srv/hoomd-blue && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug\
    -DENABLE_CUDA=OFF -DENABLE_MPI=OFF -DBUILD_HPMC=off\
     -DBUILD_CGCMM=off -DBUILD_MD=on -DBUILD_METAL=off \
    -DBUILD_TESTING=off -DBUILD_DEPRECATED=off -DBUILD_MPCD=OFF
make -j2
```

# Tests

To run the unit tests:

```bash
pytest ../htf/test-py/
```

# Bluehive Install

After cloning the `hoomd-tf` repo, follow these steps:

Load the modules necessary:

```bash
module load git anaconda3/2018.12b cmake sqlite cudnn/9.0-7
```

Set-up virtual python environment *ONCE* to keep packages isolated.

```bash
conda create -n hoomd-tf python=3.6
```

Then whenever you login and *have loaded modules*:

```bash
source activate hoomd-tf
```

Now that Python is ready, install some pre-requisites:

```bash
pip install tensorflow-gpu==1.12
```

Continue following the compling steps below to complete install.

# Compiling

The following packages are required to compile:
```
tensorflow < 2.0
hoomd-blue >= 2.5.0
numpy
tbb-devel
```

tbb-devel is only required if using the "simple" method below. The tensorflow
versions should be any Tensorflow 1 release. The higher versions, like 1.14, 1.15, 
will give lots of warnings about migrating code to Tensorflow 2.0.

## Simple Compiling

This method assumes you already have installed hoomd-blue. You could do that,
for example, via `conda install -c conda-forge hoomd`. Here are the complete steps
including that

```bash
pip install tensorflow==1.14.0
conda install -c conda-forge hoomd tbb-devel
git clone https://github.com/ur-whitelab/hoomd-tf
cd hoomd-tf && mkdir build && cd build
cmake ..
make install
```

That's it! You can perform this in a conda environment if desired as well. 

## Compiling with Hoomd-Blue

Use this method if you need to compile with developer flags on or other special requirements.

```bash
git clone --recursive https://bitbucket.org/glotzer/hoomd-blue hoomd-blue
```

We are on release v2.5.1 of hoomd-blue

```bash
cd hoomd-blue && git checkout tags/v2.5.1
```

Now we put our plugin in the source directory with a softlink:

```bash
git clone https://github.com/ur-whitelab/hoomd-tf
ln -s $HOME/hoomd-tf/htf $HOME/hoomd-blue/hoomd
```

Now compile (from hoomd-blue directory). Modify options for speed if necessary.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
 -DENABLE_CUDA=ON -DENABLE_MPI=OFF\
 -DBUILD_HPMC=off -DBUILD_CGCMM=off -DBUILD_MD=on\
 -DBUILD_METAL=off -DBUILD_TESTING=off -DBUILD_DEPRECATED=off -DBUILD_MPCD=OFF
```

Now compile with make:

```bash
make
```

Put build directory on your python path:

```bash
export PYTHONPATH="$PYTHONPATH:`pwd`"
```

## Conda Environments

If you are using a conda environment, you may need to force CMAKE to find your
python environment. This is rare, we only see it on our compute cluster which has multiple conflicting version of python and conda. The following additional flags can help with this:

```bash
cmake .. \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DPYTHON_EXECUTABLE=$(which python) \
-DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DENABLE_MPI=OFF -DBUILD_HPMC=off -DBUILD_CGCMM=off -DBUILD_MD=on -DBUILD_METAL=off -DBUILD_TESTING=off -DBUILD_DEPRECATED=off -DBUILD_MPCD=OFF
```

## Updating Compiled Code

Note: if you modify C++ code, only run make (not cmake). If you modify python, just copy over py files (`htf/*py` to `build/hoomd/htf`)

# MBuild Environment

If you are using mbuild, please follow these additional install steps:

```bash
conda install numpy cython
pip install requests networkx matplotlib scipy pandas plyplus lxml mdtraj oset
conda install -c omnia -y openmm parmed
conda install -c conda-forge --no-deps -y packmol gsd
pip install --upgrade git+https://github.com/mosdef-hub/foyer git+https://github.com/mosdef-hub/mbuild
```

# Running on Bluehive

This command works for interactive gpu use:

```bash
interactive -p awhite -t 12:00:00 --gres=gpu
```


# Known Issues

## Using Positions

Hoomd re-orders positions to improve performance. If you are using CG mappings that rely on ordering of positions, be sure to disable this:

```python
c = hoomd.context.initialize()
c.sorter.disable()
```

## Exploding Gradients

There is a bug in norms (https://github.com/tensorflow/tensorflow/issues/12071) that somtimes prevents optimizers to work well with TensorFlow norms. Note that this is only necessary if you're summing up gradients, like what is commonly done in computing gradients in optimizers. This isn't usually an issue for just computing forces. There are three ways to deal with this:

### Small Training Rates

When Training something like a Lennard-Jones potential or other `1/r` potential, high gradients are possible. You can prevent expoding gradients by using small learning rates and ensuring variables are initialized so that energies are finite.


### Safe Norm

There is a workaround (`graph_builder.safe_norm`) in Hoomd-TF. There is almost no performance penalty, so it is fine to replace `tf.norm` with `graph_builder.safe_norm` throughout. This method adds a small amount to all the norms though, so if you rely on some norms being zero it will not work well.

### Clipping Gradients

Another approach is to clip gradients instead of using safe_norm:
```python
optimizer = tf.train.AdamOptimizer(1e-4)
gvs = optimizer.compute_gradients(cost)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)
```


### Neighbor Lists

Using a max-size neighbor list is non-ideal, especially in CG simulations where density is non-uniform.

&copy; Andrew White at University of Rochester
