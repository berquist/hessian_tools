import math
import numpy as np
import numpy.linalg as npl

amu2g = 1.6605402e-24
amu2kg = amu2g / 1000.0
bohr2ang = 0.529177249
bohr2m = bohr2ang * 1.0e-10
hartree2joule = 4.35974434e-18
planck = 6.6260755e-34
# planck = 6.62606957e-34
# pi = math.acos(-1.0)
pi = math.pi
planckbar = planck/(2*pi)
speed_of_light = 299792458
avogadro = 6.0221413e+23
rot_constant = planck/(8*pi*pi*speed_of_light)
vib_constant = math.sqrt((avogadro*hartree2joule*1000) /
                         (bohr2m*bohr2m)) / \
    (2*pi*speed_of_light*100)

hessian = np.zeros(shape=(9, 9))

hessian[:,0:6] = [
[ 0.7920702,  0.0000000,  -0.0000000,  -0.3960351,  -0.0000000,  -0.3210686],
[ 0.0000000,  -0.0601381,   0.0000000,  -0.0000000,   0.0300691,  -0.0000000],
[-0.0000000,   0.0000000,   0.5151325,  -0.2058185,  -0.0000000,  -0.2575663],
[-0.3960351,  -0.0000000,  -0.2058185,   0.4148874,   0.0000000,   0.2634436],
[-0.0000000,   0.0300691,  -0.0000000,   0.0000000,  -0.0194027,   0.0000000],
[-0.3210686,  -0.0000000,  -0.2575663,   0.2634436,   0.0000000,   0.2453350],
[-0.3960351,   0.0000000,  0.2058185, -0.0188523,   0.0000000,   0.0576250],
[ 0.0000000,   0.0300691,  -0.0000000,  -0.0000000,  -0.0106664,   0.0000000],
[ 0.3210686,  -0.0000000,  -0.2575663,  -0.0576250,   0.0000000,   0.0122312]
]

hessian[:,6:] = [
[-0.3960351,   0.0000000,   0.3210686],
[ 0.0000000,   0.0300691,  -0.0000000],
[ 0.2058185,  -0.0000000,  -0.2575663],
[-0.0188523,  -0.0000000,  -0.0576250],
[ 0.0000000,  -0.0106664,   0.0000000],
[ 0.0576250,   0.0000000,   0.0122312],
[ 0.4148874,  -0.0000000,  -0.2634436],
[-0.0000000,  -0.0194027,   0.0000000],
[-0.2634436,   0.0000000,   0.2453350]
]

print('SCF Hessian:')
print(hessian)

masses = np.array([15.99491, 1.00783, 1.00783])
natom = len(masses)

hessian_mw = np.zeros(shape=(3*natom, 3*natom))
hessian_mw2 = np.zeros(shape=(3*natom, 3*natom))
hessian_mw3 = np.zeros(shape=(3*natom, 3*natom))

for i in range(natom):
    for j in range(natom):
        mi = masses[i]
        mj = masses[j]
        mimj = math.sqrt(mi*mj)
        hessian_mw[(i*natom)+0][(j*natom)+0] = hessian[(i*natom)+0][(j*natom)+0] / mimj
        hessian_mw[(i*natom)+0][(j*natom)+1] = hessian[(i*natom)+0][(j*natom)+1] / mimj
        hessian_mw[(i*natom)+0][(j*natom)+2] = hessian[(i*natom)+0][(j*natom)+2] / mimj
        hessian_mw[(i*natom)+1][(j*natom)+0] = hessian[(i*natom)+1][(j*natom)+0] / mimj
        hessian_mw[(i*natom)+1][(j*natom)+1] = hessian[(i*natom)+1][(j*natom)+1] / mimj
        hessian_mw[(i*natom)+1][(j*natom)+2] = hessian[(i*natom)+1][(j*natom)+2] / mimj
        hessian_mw[(i*natom)+2][(j*natom)+0] = hessian[(i*natom)+2][(j*natom)+0] / mimj
        hessian_mw[(i*natom)+2][(j*natom)+1] = hessian[(i*natom)+2][(j*natom)+1] / mimj
        hessian_mw[(i*natom)+2][(j*natom)+2] = hessian[(i*natom)+2][(j*natom)+2] / mimj

for i in range(3*natom):
    for j in range(3*natom):
        hessian_mw2[i][j] = hessian[i][j] / math.sqrt(masses[i//3]*masses[j//3])

for i in range(natom):
    for j in range(natom):
        hessian_mw3[(i*natom):(i*natom)+3][(j*natom):(j*natom)+3] = hessian[(i*natom):(i*natom)+3][(j*natom):(j*natom)+3] / math.sqrt(masses[i]*masses[j])

print('Mass-weighted Hessian (method 1):')
print(hessian_mw)
print('Mass-weighted Hessian (method 2):')
print(hessian_mw2)
print('Mass-weighted Hessian (method 3):')
print(hessian_mw3)

eigvals, eigvecs = npl.eigh(hessian_mw)

frequencies = np.lib.scimath.sqrt(eigvals).real * vib_constant
