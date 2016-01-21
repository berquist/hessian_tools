import sys
import numpy as np

if __name__ == '__main__':

    filename = sys.argv[1]

    lines = []
    hessvalues = []

    with open(filename) as hessfile:
        line = hessfile.readline()
        while line:
            line = line.strip()
            if line:
                lines.append(line)
            line = hessfile.readline()

    tokens = lines[1].strip().split()
    dimension = int(tokens[1])

    for line in lines[2:]:
        tokens = line.strip().split()
        if line == '$end':
            break
        for token in tokens:
            hessvalues.append(float(token))

    hessian = np.zeros(shape = (dimension, dimension))

    c1, c2 = 0, 0
    for value in hessvalues:
        hessian[c1][c2] = hessian[c2][c1] = value
        if c2 < c1:
            c2 += 1
        elif c2 == c1:
            c1 += 1
            c2 = 0
