# Tensorflow Plugin


This plugin allows using tensorflow to compute forces in a simulation
or to compute other quantities, like collective variables to fit a
potential for coarse-graining. You must first construct your
tensorlfow graph using the `tensorflow_plugin.graph_builder` class and
then add the `tfcompute` compute to your hoomd simulation.

## Building Graph

To construct a graph, construct a graphbuilder:

```python
from hoomd.tensorflow_plugin import graph_builder
graph = graph_builder(N, NN, output_forces)
```

where `N` is the number of particles in the simulation, `NN` is the maximum number of nearest neighbors to consider, and `output_forces` indicates if the graph will output forces to use in the simulation. After building the `graph`, it will have three tensors as attributes to use in constructing the tensorflow graph: `nlist`, `positions`, and `forces`. `nlist` is an `N` x `NN` x 4 tensor containing the nearest neighbors. An entry of all zeros indicates that less than `NN` nearest neighbors where present for a particular particle. The 4 right-most dimensions are `x,y,z` and `w`, which is the particle type. Note that the `x,y,z` values are a vector originating at the particle and ending at its neighbor. `positions` and `forces` are `N` x 4 tensors. `forces` *only* is available if the graph does not output forces via `output_forces=False`.

### Computing Forces

If you graph is outputting forces, you may either compute forces and pass them to `graph_builder.save(...)` or have them computed via automatic differentiation of a potential energy. Call `graph_builder.compute_forces(energy)` where `energy` is a scalar or tensor that depends on `nlist` and/or `positions`. A tensor of forces will be returned as sum(-dE / dn) - dE / dp where the sum is over the neighbor list. For example, to compute a `1 / r` potential:

```python
graph = hoomd.tensorflow_plugin.graph_builder(N, N - 1)
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

To finalize and save your graph, you must call the `graph_builder.save(directory, force_tensor=forces, virial = None, out_node=None)` function. `force_tensor` should be your computed forces, either as computed by your graph or as the output from `compute_energy`. If your graph is not outputting forces, then you must provide a tensor which will be computed, `out_node`, at each timestep. Your forces should be an `N x 4` tensor with the 4th column indicating per-particle potential energy. The virial should be an `N` x 3 x 3 tensor.

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

### Saving Scalars

If you would like to save a scalar over time, like total energy or training loss, you can use the Tensorboard functionality. Add scalars to the Tensorboard summary during the build step:

```python
tf.summary.scalar('total-energy', tf.reduce_sum(particle_energy))
```

and then add the `_write_tensorboard=True` flag during the `tfcompute` initialize. The period of tensorboard writes is controlled by the `saving_period` flag to the `tfcompute.attach` command. View the Tensorboard section below to see how to view the resulting scalars.

### Variables and Restarts

In tensorflow, variables are trainable parameters. They are required parts of your graph when doing learning. Each `saving_period` (set as arg to `tfcompute.attach`), they are written to your model directory. Note that when a run is started, the latest values of your variables are loaded from your model directory. *If you are starting a new run but you previously ran your model, the old variable values will be loaded.* Thus it is necessary to completely delete your model directory and rebuild if you don't want previously trained variables to be loaded. This behavior means though that restarts will work correctly and if you are re-using a trained model, the newest values will be loaded.

### Complete Examples

See `tensorflow_plugin/models/test-models/build.py` for more.

### Lennard-Jones

```python
graph = hoomd.tensorflow_plugin.graph_builder(N, NN)
nlist = graph.nlist[:, :, :3]
#get r
r = tf.norm(nlist, axis=2)
#compute 1 / r while safely treating r = 0.
#pairwise energy. Double count -> divide by 2
p_energy = 4.0 / 2.0 * (graph.safe_div(1., r**12) - graph.safe_div(1., r**6))
#sum over pairwise energy
energy = tf.reduce_sum(p_energy, axis=1)
forces = graph.compute_forces(energy)
graph.save(force_tensor=forces, model_directory='/tmp/lj-model')
```

## Using Graph in a Simulation

You may use a saved tensorflow model via:

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

where `model_loc` is the directory where the tensorflow model was saved, `nlist` is a hoomd neighbor list object, `r_cut` is the maximum distance for to consider particles as being neighbors, and `force_mode` is a string that indicates how to treat forces. A value of `'output'` indicates forces will be output from hoomd and input into the tensorflow model. `'add'` means the forces output from the tensorflow model should be added with whatever forces are computed from hoomd, for example if biasing a simulation. `'ignore'` means the forces will not be modified and are not used the tensorflow model, for example if computing collective variables that do not depend on forces. `'overwrite'` means the forces from the tensorflow model will overwrite the forces from hoomd, for example if the tensorflow model is computing the forces instead.

### Bootstraping Variables

If you have trained variables previously and would like to load them into the current tensorflow graph, you can use the `bootstrap` and `bootstrap_map` arguments. `bootstrap` should be a checkpoint file containing variables which can be loaded into your tfcompute graph. It will be called, then all variables will be initialized and no variables will be reloaded even if there exists a checkpoint in the model directory (to prevent overwriting your bootstrap variables). `bootstrap_map` is an optional additional argument that will have keys that are variable names in the `bootstrap` checkpoint file and values that are names in the tfcompute graph. This can be used when your variable names do not match up. Here are two example demonstrating with and without a `bootstrap_map`:

First, here's an example that creates some variables (note these would be trained in a real example)

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

### Examples

See `tensorflow_plugin/test-py/test_tensorflow.py`.

### Note on Building and Executing Tensorflow Models in Same Script

Due to the side-effects of importing tensorflow, you must build and save your graph in a separate python process first before running it hoomd.

### Interprocess Communication

*You must be on a system with at least two threads so that the tensorflow and hoomd process can run concurrently.*

## Tensorboard

You can visualize your models with tensorboard. First, add
`_write_tensorboard=True` the tensorflow plugin constructor. This will
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

Experimental, but you can trace your graph in realtime in a simulation. Add both the `_write_tensorboard=True` to
the constructor and the `_debug_mode=True` flag to `attach` command. You then open another shell and connect by following
the online instructions for interactive debugging via Tensorboard.

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

The `cap--add` and `security-opt` flags are optional and allow `gdb` debugging.

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

To run the unit tests, first run `python tensorflow_plugin/models/test-models/build.py` to build the graphs used in the tests. Then run

```bash
python tensorflow_plugin/test-py/test_tensorflow.py [test_class].[test_name]
```

to run a unit test. Note that only one test can be run at a time due to the way gpu contexts/forks occur. Some of the tests also have side-effects so you may also need to rebuild your example models directory.

If you change C++/C code, remake. If you modify python code, copy the new version to the build directory.

## Bluehive Install

Load the modules necessary:

```bash
module load anaconda cmake sqlite cuda cudnn git
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
pip install tensorflow-gpu
```

Continue following the compling steps below to complete install.

## Compiling

```bash
git clone --recursive https://bitbucket.org/glotzer/hoomd-blue hoomd-blue
```

Put our plugin in the source directory. Make a softlink:

```bash
ln -s $HOME/hoomd-tf/tensorflow_plugin $HOME/hoomd-blue/hoomd
```

Now compile (from hoomd-blue directory). Modify options for speed if necessary.

```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=ON -DENABLE_MPI=OFF\
 -DBUILD_HPMC=off -DBUILD_CGCMM=off -DBUILD_MD=on\
 -DBUILD_METAL=off -DBUILD_TESTING=off -DBUILD_DEPRECATED=off -DBUILD_MPCD=OFF
```

Now compile with make

```bash
make
```

Put build directory on your python path:

```bash
export PYTHONPATH="$PYTHONPATH:`pwd`"
```

Note: if you modify C++ code, only run make (not cmake). If you modify python, just copy over py files.

## Running on Bluehive

Because hoomd-tf requires at least two threads to run, you must ensure your bluehive reservation allows two threads. This command works for interactive gpu use:

```bash
interactive -p awhite -t 12:00:00 -N 1 --ntasks-per-node 24 --gres=gpu
```

## Issues

* Use GPU event handles -> Depends on TF While
* Domain decomposition testing -> Low priority
* Write better source doc -> Style
* Make ipc2tensor not stateful (use resource manager) -> Low priority
    Not sure if even correct, since hoomd will handle decomposition
* TF while -> Next optimization, Determined to be very difficult and unclear if necessary
* Multigpu for training via server/worker mode

### C++

balance between tf/myself/hoomd
C++ class -> Camel
C++ methods -> camel
C++ variables -> snake
C++ types -> camel _t
C++ class variables -> snake prefix
POD struct -> (c++ types) (since that is cuda style)
C++ functions -> snake (?) because they are only used in py or gpu kernels

### Python

py class ->snake

## Examples

Just made up, not sure if they work

### Force-Matching

```python
import tensorflow as tf
import hoomd.tensorflow_plugin
graph = hoomd.tensorflow_plugin.graph_builder(N, NN)
#we want to get mapped forces!
#map = tf.Variable(tf.ones([N, M]))
#zero_map_enforcer = ...
#restricted_map = zero_map_enforcer * map
# + add some normalization
map = tf.Placeholder((N, M), dtype=tf.float32)
#forces from HOOMD are fx,fy,fz,pe where pe is potential energy of particle
forces = graph.forces[:, :, :3]
mapped_forces = map * forces #think -> N x 3 * N x M
# sum_i m_ij * f_ik = cf_jk
mapped_forces = tf.einsum('ij,ik->jk', map, forces)
#get model forces
mapped_positions = tf.einsum('ij,ik->jk', map, graph.positions[:, :3])
#get mapped neighbor list
dist_r = tf.reduce_sum(mapped_positions * mapped_positions, axis=1)
# turn dist_r into column vector
dist_r = tf.reshape(dist_r, [-1, 1])
mapped_distances = dist_r - 2*tf.matmul(mapped_positions,
    tf.transpose(mapped_positions)) + tf.transpose(dist_r)
#compute our model forces on CG sites
#our model -> f(r) ->  r * w = f
#      0->0.5,0.5->1,1->1.5,1.5->infty
# r -> hr = [ 0,       0.1,    0.8,      0.1]
#f hr * w
# distance at each grid point from r
#send through RElu
grid = tf.range(0.5, 10, 0.1, dtype=tf.float32)
#want an N x N x G
grid_dist = grid - tf.tile(mapped_distances, grid.shape[0])
#one of the N x N grid distances -> [0 - r, 0.5 - r , 1.0 - r, 1.5 - r, 2.0 - r]
#want to do f(delta r) -> [0,1]
clip_high = tf.Variable(1, name='clip-high')
grid_clip = tf.clip_by_value(tf.abs(grid_dist), 0, clip_high)
#afterwards -> r = 1.4, [0, 0.9, 0.4, 0.1, 0.6,]
#r = 1.3, [0, 0.8, 0.3, 0.2, 0.7]
#TODO -> see if Prof White assumption is correct -> sum of grid_clip = 2 * clip-high
grid_normed = grid_clip / 2 / clip_high
force_weights = tf.Variable(tf.ones(grid.shape), name='force-weights')
#N x N x G * G x 1 = N x N
#TODO: we need actual rs
model_force_mag = tf.matmul(grid_normed, force_weights)
#once fixed....
model_forces = ....
error = tf.reduce_sum(tf.norm(mapped_forces - model_forces, axis=1), axis=0)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(error)

#need to tell tf to run optimizer
graph.save('/tmp/force_matching', out_nodes=[optimizer])
```

To run the model

```python
import hoomd, hoomd.md
from hoomd.tensorflow_plugin import tfcompute
tfcompute = tfcompute('/tmp/force_matching')

....setup simulation....
nlist = hoomd.md.nlist.cell()
tfcompute.attach(nlist, r_cut=r_cut, force_mode='output')
hoomd.run(1000)
```
