import hoomd
import hoomd.tensorflow_plugin as htf
import unittest


class test_find_molecules(unittest.TestCase):
    def test_find_molecules(self):
        # build system using example from hoomd
        hoomd.context.initialize()
        snapshot = hoomd.data.make_snapshot(N=10,
                                        box=hoomd.data.boxdim(Lx=10, Ly=0.5, Lz=0.5),
                                        particle_types=['A', 'B'],
                                        bond_types=['polymer'])
        snapshot.particles.position[:] = [[-4.5, 0, 0], [-3.5, 0, 0],
                                        [-2.5, 0, 0], [-1.5, 0, 0],
                                        [-0.5, 0, 0], [0.5, 0, 0],
                                        [1.5, 0, 0], [2.5, 0, 0],
                                        [3.5, 0, 0], [4.5, 0, 0]]


        snapshot.particles.typeid[0:7]=0
        snapshot.particles.typeid[7:10]=1


        snapshot.bonds.resize(9)
        snapshot.bonds.group[:] = [[0,1], [1, 2], [2,3],
                                [3,4], [4,5], [5,6],
                                [6,7], [7,8], [8,9]]
        snapshot.replicate(1,3,3)
        system = hoomd.init.read_snapshot(snapshot)
        # test out mapping
        mapping = htf.find_molecules(system)
        assert len(mapping) == 9
        assert len(mapping[0]) == 10



