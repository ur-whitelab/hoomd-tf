import tensorflow as tf
import hoomd
import gsd.hoomd
import pickle
import hoomd.md
import hoomd_ff
from hoomd.htf import tfcompute
import sys
if(len(sys.argv) != 2):
    print('Usage: build_scattering.py [model_dir]')
    exit(0)

model_dir = sys.argv[1]
struct_dir = '/scratch/hgandhi/run_scattering/'

with gsd.hoomd.open(struct_dir + 'water.gsd') as g:
    frame = g[0]
    boxd = frame.configuration.box[0]

with open(struct_dir + 'water.pickle', 'rb') as f:
    param_sys, kwargs = pickle.load(f)


with hoomd.htf.tfcompute(model_dir, _mock_mode=False,
                         write_tensorboard=False) as tfcompute:
    hoomd.context.initialize("")
    r_cut = 4.0
    N = 5000
    NN = 256
    system = hoomd.init.read_gsd(struct_dir + 'water.gsd')
    nlist = hoomd.md.nlist.cell(check_period=1)
    hoomd_ff.pair_coeffs(frame, param_sys, nlist, r_cut=4.0)
    # set-up pppm
    charged = hoomd.group.all()
    pppm = hoomd.md.charge.pppm(nlist=nlist, group=charged)
    pppm.set_params(Nx=32, Ny=32, Nz=32, order=6, rcut=4.0)
    # set-up bonds
    hoomd_ff.bond_coeffs(frame, system, param_sys)
    # set-up angles
    hoomd_ff.angle_coeffs(frame, param_sys)
    # set-up dihedrals
    hoomd_ff.dihedral_coeffs(frame, param_sys)
    # free particles from rigid bodies since rigid doesn't quite work for us
    for i in range(frame.particles.N):
        system.particles[i].body = -1
    all = hoomd.group.all()
    kT = 1.9872 / 1000
    fire = hoomd.md.integrate.mode_minimize_fire(dt=0.5 / 48.9,
                                                 ftol=1e-4, Etol=1e-8)
    nve = hoomd.md.integrate.nve(group=all)
    for i in range(5):
        if not(fire.has_converged()):
            hoomd.run(100)
    nve.disable()
    # Now NVT
    hoomd.md.integrate.mode_standard(dt=2 / 48.9)
    nvt = hoomd.md.integrate.nvt(group=all, kT=298 * kT, tau=100 / 48.9)
    nvt.randomize_velocities(1234)
    tfcompute.attach(nlist, r_cut=r_cut, period=1)
    # hoomd.analyze.log(filename='scattering.log',
    #                  quantities=['momentum', 'temperature'],
    #                  period=10,
    #                  overwrite=True)
    hoomd.dump.gsd("trajectory.gsd", period=10, group=all, overwrite=True)
    hoomd.run(1e4)
    positions_array = tfcompute.get_positions_array()
    tfcompute.get_nlist_array()
    # print('positions array is {}'.format(positions_array))
