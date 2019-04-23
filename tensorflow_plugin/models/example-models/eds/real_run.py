import hoomd, hoomd.md, hoomd.tensorflow_plugin
import tensorflow as tf
tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
#output_aver_r_running_mean.txt
with hoomd.tensorflow_plugin.tfcompute('manuscrpt/eds') as tfcompute:
    hoomd.context.initialize()
    N = 64
    NN = 30
    rcut = 3.0
    system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                       n=[8,8])
    nlist = hoomd.md.nlist.cell(check_period = 1)
    lj = hoomd.md.pair.lj(rcut, nlist)
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    hoomd.md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.nve(
        group=hoomd.group.all()).randomize_velocities(kT=0.2, seed=42)
    hoomd.analyze.log(filename='eds.log',
                      quantities = ['momentum', 'temperature', 'time'],
                      period=100,overwrite=True)

    tfcompute.attach(nlist, r_cut=rcut)
    #hoomd.dump.gsd(filename="trajectory_eds.gsd", period=10)
    hoomd.dump.gsd(filename="saveall_running.gsd", overwrite=True, period=100, group=hoomd.group.all(), dynamic=['attribute', 'momentum', 'topology'])
    hoomd.run(200000)
