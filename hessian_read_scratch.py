#!/usr/bin/env python3

from __future__ import print_function

import os.path

import math
import numpy as np

import scripts.periodic_table as pt


amu2g = 1.6605402e-24
amu2kg = amu2g / 1000.0
bohr2ang = 0.529177249
bohr2m = bohr2ang * 1.0e-10
hartree2joule = 4.35974434e-18
planck = 6.6260755e-34
pi = math.pi
planckbar = planck/(2*pi)
speed_of_light = 299792458
avogadro = 6.0221413e+23
rot_constant = planck/(8*pi*pi*speed_of_light)
vib_constant = math.sqrt((avogadro*hartree2joule*1000) /
                         (bohr2m*bohr2m)) / \
    (2*pi*speed_of_light*100)


def getargs():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('scratchdir')
    parser.add_argument('--outputfile')

    args = parser.parse_args()

    return args


def complex_as_negative(arr):
    return arr.real - arr.imag


def read_file(filename):

    lines = []

    with open(filename) as fh:
        line = fh.readline()
        while line:
            line = line.strip()
            if line:
                lines.append(line)
            line = fh.readline()

    return lines


def read_scratch_hessian(filename):

    lines = read_file(filename)

    hessvalues = []

    tokens = lines[1].split()
    dimension = int(tokens[1])

    for line in lines[2:]:
        tokens = line.split()
        if line == '$end':
            break
        for token in tokens:
            hessvalues.append(float(token))

    hessian = np.zeros(shape=(dimension, dimension))

    c1, c2 = 0, 0
    for value in hessvalues:
        hessian[c1][c2] = hessian[c2][c1] = value
        if c2 < c1:
            c2 += 1
        elif c2 == c1:
            c1 += 1
            c2 = 0

    return hessian


def read_scratch_molecule(filename):

    # Assume that it isn't a fragment job and take everything except
    # the $ lines and the charge/multiplicity.
    lines = read_file(filename)[2:-1]

    atomsyms = [line.split()[0] for line in lines]

    return atomsyms


def atomsyms2atomnums(atomsyms):

    return [pt.AtomicNum[atomsym] for atomsym in atomsyms]


def atomsyms2atommasses(atomsyms):

    return [pt.Mass[atomsym] for atomsym in atomsyms]


def hessian_mass_weight_1(hessian, atommasses):

    hessian_mw = np.zeros(shape=hessian.shape)

    natom = len(atommasses)
    assert hessian.shape[0] // 3 == natom

    for i in range(natom):
        for j in range(natom):
            mi = atommasses[i]
            mj = atommasses[j]
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

    return hessian_mw


def hessian_mass_weight_2(hessian, atommasses):

    hessian_mw = np.zeros(shape=hessian.shape)

    natom = len(atommasses)
    assert hessian.shape[0] // 3 == natom

    for i in range(3*natom):
        for j in range(3*natom):
            hessian_mw[i][j] = hessian[i][j] / math.sqrt(atommasses[i//3]*atommasses[j//3])

    return hessian_mw


def hessian_mass_weight_3(hessian, atommasses):

    hessian_mw = np.zeros(shape=hessian.shape)

    natom = len(atommasses)
    assert hessian.shape[0] // 3 == natom

    for i in range(natom):
        for j in range(natom):
            hessian_mw[(i*natom):(i*natom)+3][(j*natom):(j*natom)+3] = hessian[(i*natom):(i*natom)+3][(j*natom):(j*natom)+3] / math.sqrt(atommasses[i]*atommasses[j])

    return hessian_mw


def hessian_mass_weight_4(hessian, atommasses):

    hessian_mw = np.zeros(shape=hessian.shape)

    nrows, ncols = hessian_mw.shape

    for i in range(nrows):
        for j in range(ncols):
            _denom = math.sqrt(atommasses[i // 3] * atommasses[j // 3])
            hessian_mw[i, j] = hessian[i, j] / _denom

    return hessian_mw


def main(args):

    hessian = read_scratch_hessian(os.path.join(args.scratchdir, 'HESS'))
    atomsyms = read_scratch_molecule(os.path.join(args.scratchdir, 'molecule'))
    atommasses = atomsyms2atommasses(atomsyms)
    hessian_mw = hessian_mass_weight_4(hessian, atommasses)
    eigvals, eigvecs = np.linalg.eigh(hessian_mw)
    frequencies = np.lib.scimath.sqrt(eigvals)[6:] * vib_constant
    print(complex_as_negative(frequencies))

    if args.outputfile:
        from cclib.parser import ccopen
        job = ccopen(args.outputfile)
        data = job.parse()
        hessian_outputfile = data.hessian
        hessian_outputfile_mw = hessian_mass_weight_4(hessian_outputfile, atommasses)
        eigvals_outputfile, eigvecs_outputfiles = np.linalg.eigh(hessian_outputfile_mw)
        frequencies_outputfile_hess = np.lib.scimath.sqrt(eigvals_outputfile)[6:] * vib_constant
        frequencies_outputfile  = data.vibfreqs
        print(complex_as_negative(frequencies_outputfile_hess))
        print(frequencies_outputfile)

if __name__ == '__main__':

    args = getargs()
    main(args)
