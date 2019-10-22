import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
parser.add_argument('-o', type=str, default=None)
args = parser.parse_args()

print("Loading data from:\n   %s" % args.file)
data = np.loadtxt(args.file, comments='%', delimiter=',')

data_sorted = np.sort(data[:, 1:], axis=1)
ind_drop = np.argmax(np.all(np.isnan(data_sorted),axis=0))-1
data_crop = data_sorted[:,:ind_drop]

if args.o is None:
    output = args.file
else:
    output = args.o

print("Saving cleaned data to:\n   %s" % output)
np.savetxt(output, np.hstack((data[:,0].reshape((1,-1)).T, data_crop)), delimiter=',', fmt='%8f')
