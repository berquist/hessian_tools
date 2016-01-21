import sys
import math
import numpy as np
import numpy.linalg as npl
from cclib.parser import ccopen
import periodic_table as pt

np.set_printoptions(linewidth=200)

planck = 6.6260755e-34
bohr2ang = 0.529177249
bohr2m = bohr2ang * 1.0e-10
hartree2joule = 4.35974434e-18
speed_of_light = 299792458
avogadro = 6.0221413e+23
# to go from atomic units to wavenumbers
vib_constant = math.sqrt((avogadro * hartree2joule * 1000) /
                         (bohr2m * bohr2m)) / \
    (2 * math.pi * speed_of_light * 100)
zpe_prefactor = 0.5 * planck * (100 * speed_of_light) * avogadro / 4184

def com(positions, masses):
    mass_sum = sum(masses)
    CMx = sum(positions[:, 0] * masses) / mass_sum
    CMy = sum(positions[:, 1] * masses) / mass_sum
    CMz = sum(positions[:, 2] * masses) / mass_sum
    return (CMx, CMy, CMz)

def zpe(real_frequencies):
    zpe = zpe_prefactor * sum(real_frequencies)
    return zpe

if __name__ == '__main__':

    filename = sys.argv[1]

    print('=' * 78)
    print('filename:', filename)

    job = ccopen(filename)
    data = job.parse()

    hessian_scf = data.hessian

    # Before obtaining frequencies from the SCF Hessian, we must remove any
    # blocks that correspond to ghost atoms, then mass-weight it.

    gh_indices = list(i for i in range(len(data.atomnos))
                      if data.atomnos[i] < 1)
    atom_indices = list(i for i in range(len(data.atomnos))
                        if data.atomnos[i] > 0)
    hessian_dim = 3 * len(atom_indices)
    hessian_scf_noghost = np.empty(shape = (hessian_dim, hessian_dim))

    # Transfer non-ghost blocks to the new Hessian:
    for newidrows, oldidrows in enumerate(atom_indices):
        for newidcols, oldidcols in enumerate(atom_indices):
            hessian_scf_noghost[(3*newidrows):(3*newidrows)+3,
                                (3*newidcols):(3*newidcols)+3] \
                = hessian_scf[(3*oldidrows):(3*oldidrows)+3,
                              (3*oldidcols):(3*oldidcols)+3]

    atomicnums = np.array(list(data.atomnos[i] for i in atom_indices))
    atommasses = np.array(list(pt.Mass[pt.Element[i]] for i in atomicnums))
    atomcoords = np.array(list(data.atomcoords[-1][i] for i in atom_indices))
    atomcom = com(atomcoords, atommasses)

    # Mass-weight the new Hessian:
    hessian_mw_noghost = np.empty(shape = (hessian_dim, hessian_dim))
    for i in range(hessian_dim):
        for j in range(hessian_dim):
            hessian_mw_noghost[i, j] = hessian_scf_noghost[i, j] / \
                                       math.sqrt(atommasses[i//3] *
                                                 atommasses[j//3])

    eigvals, eigvecs = npl.eigh(hessian_mw_noghost)
    # trash the lowest 6 modes
    eigvals, eigvecs = eigvals[6:], eigvecs[6:, :]
    frequencies = np.lib.scimath.sqrt(eigvals) * vib_constant
    real_frequencies = list(freq.real for freq in frequencies
                            if freq.real > 0.0)
    print('real frequencies:',
          real_frequencies)
    print('ZPE (kcal/mol):',
          zpe(real_frequencies))
    print('ZPE (Hartree/particle):',
          zpe(real_frequencies) / 627.50947414)
    print('elec energy (no ZPE) (Hartree/particle):',
          data.scfenergies[-1] / 27.21138505)
