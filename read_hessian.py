#!/usr/bin/env python

# Reads Hessian from Q-Chem output and converts it to a MatLab file

import sys
import numpy as np
import scipy.io
import numpy.linalg

if (len(sys.argv) != 3):
    print "Usage: read_hessian  input-file   output-file"
    exit(0)


file = open(sys.argv[1], 'r')

NAtoms = 0
XYZ = []
H = np.zeros((3,3))
for line in file:
    # Read number of atoms
    if "Standard Nuclear Orientation (Angstroms)" in line:
        for line in file:
            if "Molecular Point Group" in line:
                print "NAtoms = ", NAtoms
                break

            tokens = line.split()
            if len(tokens) == 5:
                try:
                    XYZ.append(float(tokens[2]))
                    XYZ.append(float(tokens[3]))
                    XYZ.append(float(tokens[4]))
                    NAtoms = NAtoms + 1
                except ValueError:
                    print ""

            #if "There are" in line:
            #    break

    if "Hessian of the SCF Energy" in line:

        H = np.zeros((3*NAtoms,3*NAtoms))

        col = 0
        col_previous = 0
        row = 0
        block = 0
        for line in file:
            if "Gradient time:" in line:
                break

            tokens = line.split()
            if (tokens[2].isdigit()):
                row = 0
                block = block + 1
                col = col + col_previous
                continue

            #col = (block - 1) * (len(tokens) - 1)
            col_previous = len(tokens) - 1

            for i in range(len(tokens) - 1):
                #print '%-4d  %-4d   %16.10f' % (row, col + i, float(tokens[1+i]))
                H[row,col+i] = float(tokens[1+i])

            row = row + 1


file.close()
print "#XYZ = ", len(XYZ)
print "H.shape = ", H.shape

#for i in r
#print H.shape
#print H
#scipy.io.savemat("test.txt", H)
#scipy.io.savemat(sys.argv[2], mdict={'arr': H})

Hinv = np.linalg.pinv(H)
print "Hinv.shape = ", Hinv.shape
#scipy.io.savemat(sys.argv[2], mdict={'arr': H})
#np.savetxt(sys.argv[2], Hinv, fmt='%18e', delimiter=' ')


#e, x = np.linalg.eig(Hinv)

#print x
#print e

#for i in range(3):
#    for j in range(3*NAtoms):
#        print '%-4d  %-4d   %20.10e' % (i, j, Hinv[i,j])

###
print "Constructing Jacobian ..."
###


Ncart = 3*NAtoms
Npairs = NAtoms * NAtoms

J = np.zeros((Ncart, Npairs))
R = np.zeros((NAtoms, NAtoms))

for k in range(Ncart):

    xk = XYZ[k]
    ij = 0

    for i in range(NAtoms):

        xi = XYZ[3*i]
        yi = XYZ[3*i+1]
        zi = XYZ[3*i+2]

        for j in range(NAtoms):

            if i != j:

                xj = XYZ[3*j]
                yj = XYZ[3*j+1]
                zj = XYZ[3*j+2]

                rij = ( (xi - xj)**2.0 + (yi - yj)**2.0 + (zi - zj)**2.0 )**0.5

                J_k_ij = xk / rij
                R[i,j] = rij
            else:
                J_k_ij = 0.0
                R[i,j] = 0.0

            J[k,ij] = J_k_ij
            ij = ij + 1


print "  J.shape = ", J.shape


###
print "Coordinate transformation ..."
###

Jinv = np.linalg.pinv(J)
#print Jinv
#print "J^(-1) * J: ", np.dot(Jinv, J)
print "  Jinv.shape = ", Jinv.shape

# Hessian in internal (r_ij) coordinates:
# (H_int) = (d2E/drab drcd) = (dxi/drab) (d2E/dxi dxj) (dxj/drcd)
H_int = np.dot(Jinv, np.dot(H, Jinv.T))
print "  H_internal.shape = ", H_int.shape


###
print "Calculating (pseudo-)inverse Hessian Hpinv ..."
###

Hpinv = np.linalg.pinv(H_int)
print "  Hpinv.shape = ", Hpinv.shape
print "Hpinv = ", Hpinv
print


###
print "Writing Hpinv to disk ..."
thresh = 4.0
print "Printing only if distance is < ", thresh
###

fOut = open(sys.argv[2], 'w')

# Print full matrix ...
#for i in range(NAtoms):
#    for j in range(NAtoms):
#        #if i == j:
#        #    continue
#        ij = i*NAtoms + j
#        #fOut.write('Hpinv[%d,%d]:\n' % (i, j))
#        for k in range(NAtoms):
#            for l in range(NAtoms):
#                #if k == l:
#                #    continue
#                kl = k*NAtoms + l
#                #fOut.write('  %-4d,%-4d   %20.10e\n' % (k, l, abs(Hpinv[ij,kl])))
#                fOut.write('%-4d    %-4d   %20.10e\n' % (ij, kl, abs(Hpinv[ij,kl])))

# ... and just the significant diagonals
for i in range(NAtoms):
    for k in range(NAtoms):
        if i == k:
            continue
        ij = i*NAtoms + k
        kl = k*NAtoms + i
        if R[i,k] < thresh:
            fOut.write('%-4d    %-4d   %20.10e\n' % (i, k, abs(Hpinv[ij,kl])))

fOut.close()
