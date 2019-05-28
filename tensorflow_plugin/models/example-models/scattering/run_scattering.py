import tensorflow as tf
import hoomd
from hoomd.tensorflow_plugin import tfcompute
model_dir = '/scratch/hgandhi/scattering/'


with hoomd.tensorflow_plugin.tfcompute(model_dir, _mock_mode=False,
                                       write_tensorboard=False) as tfcompute:
    hoomd.context.initialize("")
    N = 8*8
    NN = N - 1
    r_cut = 1
    hoomd.init.create_lattice(unitcell=hoomd.lattice.sc(a=1.0), n=4)
    nlist = hoomd.md.nlist.cell(check_period=5)
    lj = hoomd.md.pair.lj(r_cut=r_cut, nlist=nlist)
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    hoomd.md.integrate.mode_standard(dt=0.005)
    all = hoomd.group.all()
    hoomd.md.integrate.langevin(group=all, kT=0.2, seed=42)
    tfcompute.attach(nlist, r_cut=r_cut, force_mode='ignore')
    hoomd.analyze.log(filename='scattering.log',
                      quantities=['momentum', 'temperature'],
                      period=10,
                      overwrite=True)
    hoomd.dump.gsd("trajectory.gsd", period=2e3, group=all, overwrite=True)
    hoomd.run(100)
    positions_array = tfcompute.get_positions_array()
    tfcompute.get_nlist_array()
    # print('positions array is {}'.format(positions_array))
