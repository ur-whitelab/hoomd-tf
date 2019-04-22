# TensorFlow Plugin

This plugin allows using TensorFlow to compute forces in a simulation
or to compute other quantities, like collective variables to fit a
potential for coarse-graining. You must first construct your
tensorlfow graph using the `tensorflow_plugin.graph_builder` class and
then add the `tfcompute` compute to your hoomd simulation. See Known Issues at the bottom for important notes.

## Requirements

```
tensorflow == 1.12
hoomd-blue == 2.5.1 (must compile, cannot use conda. TODO)
numpy
```

## Building Graph

To construct a graph, construct a graphbuilder:

```python
from hoomd.tensorflow_plugin import graph_builder
graph = graph_builder(NN, output_forces)
```

where `NN` is the maximum number of nearest neighbors to consider, and
`output_forces` indicates if the graph will output forces to use in
the simulation. After building the `graph`, it will have three tensors
as attributes to use in constructing the TensorFlow graph: `nlist`,
`positions`, and `forces`. `nlist` is an `N` x `NN` x 4 tensor
containing the nearest neighbors. An entry of all zeros indicates that
less than `NN` nearest neighbors where present for a particular
particle. The 4 right-most dimensions are `x,y,z` and `w`, which is
the particle type. Particle type is an integer starting at 0. Note
that the `x,y,z` values are a vector originating at the particle and
ending at its neighbor. `positions` and `forces` are `N` x 4
tensors. `forces` *only* is available if the graph does not output
forces via `output_forces=False`.

### Computing Forces

If you graph is outputting forces, you may either compute forces and pass them to `graph_builder.save(...)` or have them computed via automatic differentiation of a potential energy. Call `graph_builder.compute_forces(energy)` where `energy` is a scalar or tensor that depends on `nlist` and/or `positions`. A tensor of forces will be returned as sum(-dE / dn) - dE / dp where the sum is over the neighbor list. For example, to compute a `1 / r` potential:

```python
graph = hoomd.tensorflow_plugin.graph_builder(N - 1)
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

### Virial

The virial is computed and added to the graph if you use the
`compute_forces` function and your energy has a non-zero derivative
with respect to `nlist`. You may also explicitly pass the virial when
saving, or pass `None` to remove the automatically calculated virial.

### Finalizing the Graph

To finalize and save your graph, you must call the `graph_builder.save(directory, force_tensor=forces, virial = None, out_node=None)` function. `force_tensor` should be your computed forces, either as computed by your graph or as the output from `compute_energy`. If your graph is not outputting forces, then you must provide a tensor which will be computed, `out_node`, at each timestep. Your forces should be an `N x 4` tensor with the 4th column indicating per-particle potential energy. The virial should be an `N x 3 x 3` tensor.

### Printing

If you would like to print out the values from nodes in your graph, you can
add a print node to the `out_nodes`. For example:

```python
...graph building code...
forces = graph.compute_forces(energy)
print_node = tf.Print(energy, [energy], summarize=1000)
graph.save(force_tensor=forces, model_directory=name, out_nodes=[print_node])
```

The `summarize` keyword sets the maximum number of numbers to print. Be wary of printing thousands of numbers per step.

### Saving Scalars in Tensorboard

If you would like to save a scalar over time, like total energy or training loss, you can use the Tensorboard functionality. Add scalars to the Tensorboard summary during the build step:

```python
tf.summary.scalar('total-energy', tf.reduce_sum(particle_energy))
```

and then add the `write_tensorboard=True` flag during the `tfcompute` initialize. The period of tensorboard writes is controlled by the `saving_period` flag to the `tfcompute.attach` command. View the Tensorboard section below to see how to view the resulting scalars.

### Variables and Restarts

In TensorFlow, variables are trainable parameters. They are required parts of your graph when doing learning. Each `saving_period` (set as arg to `tfcompute.attach`), they are written to your model directory. Note that when a run is started, the latest values of your variables are loaded from your model directory. *If you are starting a new run but you previously ran your model, the old variable values will be loaded.* To prevent this unexpectedly loading old checkpoints, if you run `graphbuilder.save` it will move out all old checkpoints. This behavior means that if you want to restart, you should not re-run `graphbuild.save` or pass `move_previous = False` as a parameter.

Variables are how you can save data without using Tensorboard. They can be accumulated between steps. Be sure to set them to be `trainable=False` if you are also doing learning but would like to accumulate in variables. For example, you can have a variable for running mean.

### Optional: Keras Layers for Model Building

Currently HOOMD-TF supports Keras layers in model building. We do not yet support Keras `Model.compile()` or `Model.fit()`. This example shows how to set up a neural network model using Keras layers.

```python
import tensorflow as tf
import keras
import hoomd.tensorflow_plugin as htf

NN = 64
N_hidden_nodes = 5
graph = htf.graph_builder(NN, output_forces=False)
r_inv = graph.nlist_rinv
input_tensor = tf.reshape(r_inv, shape=(-1,1), name='r_inv')
#we don't need to explicitly make a keras.Model object, just layers
input_layer = keras.layers.Input(tensor=input_tensor)
hidden_layer = keras.layers.Dense(N_hidden_nodes)(input_layer)
output_layer = keras.layers.Dense(1, input_shape=(N_hidden_nodes,))(hidden_layer)
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

The model can then be loaded and trained as normal. Note that `keras.models.Model.fit()` is not currently supported. You must train using `tensorflow_plugin.tfcompute()` as explained in the next section.

### Complete Examples

See [tensorflow_plugin/models](tensorflow_plugin/models)

### Lennard-Jones with 1 Particle Type

```python
graph = hoomd.tensorflow_plugin.graph_builder(NN)
#use convenience rinv
r_inv = graph.nlist_rinv
p_energy = 4.0 / 2.0 * (r_inv**12 - r_inv**6)
#sum over pairwise energy
energy = tf.reduce_sum(p_energy, axis=1)
forces = graph.compute_forces(energy)
graph.save(force_tensor=forces, model_directory='/tmp/lj-model')
```

## Using Graph in a Simulation

You may use a saved TensorFlow model via:

```python
import hoomd, hoomd.md
import hoomd.tensorflow_plugin

with hoomd.tensorflow_plugin.tfcompute(model_dir) as tfcompute:

    ...hoomd initialization code...

    nlist = hoomd.md.nlist.cell()
    tfcompute.attach(nlist, r_cut=3)

    ...other hoomd code...

    hoomd.run(...)

```

where `model_dir` is the directory where the TensorFlow model was saved, `nlist` is a hoomd neighbor list object and `r_cut` is the maximum distance for to consider particles as being neighbors. `nlist` is optional and is not required if your graph doesn't use the `nlist` object.

### Bootstraping Variables

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

Now here's how we would load them in the hoomd run script:
```python
with hoomd.tensorflow_plugin.tfcompute(model_dir,
    bootstrap='/tmp/bootstrap/model') as tfcompute:
    ...
```

Now here's how we would load them in the hoomd run script if we want to change
the names of the variables:
```python
# here the pretrained variable parameters will replace variables with a different name
with hoomd.tensorflow_plugin.tfcompute(model_dir,
    bootstrap='/tmp/bootstrap/model',
    bootstrap_map={'lj-epsilon':'epsilon', 'lj-sigma':'sigma'}) as tfcompute:
    ...
```

#### Bootstrapping Variables from Other Models

Here's an example of bootstrapping where you train with Hoomd and then load the variables into a different model:

```python
# build_models.py
import tensorflow as tf
import hoomd.tensorflow_plugin

def make_train_graph(NN, directory):
    # build a model that fits the energy to a linear term
    graph = hoomd.tensorflow_plugin.graph_builder(NN, output_forces=False)
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
    graph = hoomd.tensorflow_plugin.graph_builder(NN)
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

Now here is how we run the training model:
```python
#run_train.py
import hoomd, hoomd.md, hoomd.tensorflow_plugin


with hoomd.tensorflow_plugin.tfcompute('/tmp/training') as tfcompute:
    hoomd.context.initialize()
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

Now we load the variables trained in the training run into the model which computes forces:

```python
#run_inference.py
with hoomd.tensorflow_plugin.tfcompute('/tmp/inference',
        bootstrap='/tmp/training') as tfcompute:
    hoomd.context.initialize()
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

## Utilities

There are a few convenience functions in the `hoomd.tensorflow_plugin.utils` for plotting potential energies of pairwise potentials and constructing CG mappings.


### Running mean

To compute the running mean of some property, use `graph.running_mean(...)`
and load it with `load_variables`:

```python
# set-up graph to compute energy
...
graph.running_mean(energy, 'avg-energy')
# run the simulation
...
variables  = hoomd.tensorflow_plugin.load_variables(model_dir, ['avg-energy'])
print(variables)
```

### RDF

To compute an RDF, use the `graph.compute_rdf(...)` method:

```python
# set-up graph to compute energy
...
rdf = graph.compute_rdf([1,10], 'rdf', nbins=200)
graph.running_mean(rdf, 'avg-rdf')
# run the simulation
...
variables  = hoomd.tensorflow_plugin.load_variables(model_dir, ['avg-rdf'])
print(variables)
```


## Coarse-Graining Utilities

### Find Molecules
To go from atom index to particle index, use the `hoomd.tensorflow_plugin.find_molecules(...)` method:
```python
# The method takes in a hoomd system as an argument.
...
molecule_mapping_index = hoomd.tensorflow_plugin.find_molecules(system)
...

```

### Sparse Mapping

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

### Center of Mass

The `center_of_mass(...)` method maps the given positions according to the specified mapping operator to coarse-grain site positions considering periodic boundary condition. The coarse grain site position is placed at the center of mass of its constituent atoms.


```python

...
mapped_position = htf.center_of_mass(graph.positions[:,:3], cg_mapping, system)
#cg_mapping is the output from the sparse_matrix(...) method and indicates how each molecule is mapped.
...

```

### Compute Mapped Neighbor List
The `compute_nlist(...)` method returns the neighbor list for the mapped coarse-grained particles. In the example, mapped_position is the mapped particle positions obeying the periodic boundary condition as returned by the `center_of_mass(...) method`, `rcut` is the cut-off radius and `NN` is the number of nearest neighbors to be considered for the coarse-grained system.
```python
...
mapped_nlist= htf.compute_nlist(mapped_position, rcut, NN, system)
...

```

## Tensorboard

You can visualize your models with tensorboard. First, add
`write_tensorboard=True` the TensorFlow plugin constructor. This will
add a new directory called `tensorboard` to your model directory.

After running, you can launch tensorboard like so:

```bash
tensorboard --logdir=/path/to/model/tensorboard
```

and then visit `http://localhost:6006` to view the graph.

### Viewing when TF is running on remote server

If you are running on a server, before launching tensorboard use this ssh command to login:

```bash
ssh -L 6006:[remote ip or hostname]:6006 username@remote
```

and then you can view after launching on the server via your local web browser.

### Viewing when TF is running in container

If you are running docker, you can make this port available a few different ways. The first is
to get the IP address of your docker container (google how to do this if not default), which is typically `172.0.0.1`, and then
visit `http://172.0.0.1:6006` or equivalent if you have a different IP address for your container.

The second option is to use port forwarding. You can add a port forward flag, `-p 6006:6006`, when running the container which
will forward traffic from your container's 6006 port to the host's 6006 port. Again, then you can visit `http://localhost:6006` (linux)
or `http://127.0.0.1:6006` (windows).

The last method, which usually works when all others fail, is to have all the container's traffic be on the host. You can do this by
adding the flag `--net=host` to the run command of the container. Then you can visit  `http://localhost:6006`.

## Interactive Mode

Experimental, but you can trace your graph in realtime in a simulation. Add both the `write_tensorboard=True` to
the constructor and the `_debug_mode=True` flag to `attach` command. You then open another shell and connect by following
the online instructions for interactive debugging from Tensorboard.

## Docker Image for Development

To use the included docker image:

```bash
docker build -t hoomd-tf tensorflow_plugin
```

To run the container:

```bash
docker run --rm -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
 -v /insert/path/to/tensorflow_plugin/:/srv/hoomd-blue/tensorflow_plugin hoomd-tf bash
```

The `cap--add` and `security-opt` flags are optional and allow `gdb`
debugging. Install `gdb` and `python3-dbg` packages to use `gdb` with
the package.

Once in the container:

```bash
cd /srv/hoomd-blue && mkdir build && cd build
cmake .. -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_BUILD_TYPE=Debug\
     -DCMAKE_C_FLAGS=-march=native \
    -DENABLE_CUDA=OFF -DENABLE_MPI=OFF -DBUILD_HPMC=off\
     -DBUILD_CGCMM=off -DBUILD_MD=on -DBUILD_METAL=off \
    -DBUILD_TESTING=off -DBUILD_DEPRECATED=off -DBUILD_MPCD=OFF
make -j2
```

## Tests

To run the unit tests:

```bash
pytest ../tensorflow_plugin/test-py/
```

MPI is not currently supported for units tests for the last ~10 or so commits!

TODO: Fix this! Models need to be built only on root node.


## Bluehive Install

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

## Compiling

```bash
git clone --recursive https://bitbucket.org/glotzer/hoomd-blue hoomd-blue
```

We are on release v2.5.1 of hoomd-blue

```bash
cd hoomd-blue && git checkout tags/v2.5.1
```

Put our plugin in the source directory. Make a softlink:

```bash
ln -s $HOME/hoomd-tf/tensorflow_plugin $HOME/hoomd-blue/hoomd
```

Now compile (from hoomd-blue directory). Modify options for speed if necessary.

```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=ON -DENABLE_MPI=OFF\
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

### Updating Compiled Code

Note: if you modify C++ code, only run make (not cmake). If you modify python, just copy over py files (`tensorflow_plugin/*py` to `build/hoomd/tensorflow_plugin`)

## MBuild Environment

If you are using mbuild, please follow these additional install steps:

```bash
conda install numpy cython
pip install requests networkx matplotlib scipy pandas plyplus lxml mdtraj oset
conda install -c omnia -y openmm parmed
conda install -c conda-forge --no-deps -y packmol gsd
pip install --upgrade git+https://github.com/mosdef-hub/foyer git+https://github.com/mosdef-hub/mbuild
```

## Running on Bluehive

Because hoomd-tf requires at least two threads to run, you must ensure your bluehive reservation allows two threads. This command works for interactive gpu use:

```bash
interactive -p awhite -t 12:00:00 -N 1 --ntasks-per-node 24 --gres=gpu
```


## Known Issues

### Using Positions

Hoomd re-orders positions to improve performance. If you are using CG mappings that rely on ordering of positions, be sure to disable this:

```python
c = hoomd.context.initialize()
c.sorter.disable()
```

### Exploding Gradients

There is a bug in norms (https://github.com/tensorflow/tensorflow/issues/12071) that makes it impossible to use optimizers with TensorFlow norms. To get around this, use the builtin workaround (`graphbuilder.safe_norm`). Note that this is only necessary if you're summing up gradients, like what is commonly done in computing gradients in optimizers. There is almost no performance penalty, so it is fine to replace `tf.norm` with `graphbuilder.safe_norm` throughout.

You can also clip gradients instead of using safe_norm:
```python
optimizer = tf.train.AdamOptimizer(1e-4)
gvs = optimizer.compute_gradients(cost)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)
```
which will also prevent exploding gradients. Remember that if training something like a Lennard-Jones potential or other `1/r` potential that high gradients are possible. Use small learning rates and probably clip the grads.

### Neighbor Lists

Using a max-size neighbor list is non-ideal, especially in CG simulations where density is non-uniform.

&copy; Andrew White at University of Rochester
