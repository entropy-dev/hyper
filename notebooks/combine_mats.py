#!/usr/bin/env python3


from scipy.io import loadmat
from os import listdir
from os.path import join
from tqdm import tqdm
from h5py import File

path_in = 'testfiles/'
path_out = 'result.mat'

filenames = [join(path_in, i) for i in sorted(listdir(path_in)) if i.endswith('.mat')]

size = (len(filenames), 214, 407, 3)

f = File(path_out, 'w')
dset = f.create_dataset("data", size, dtype='f', compression="gzip", compression_opts=2)

for i,p in enumerate(tqdm(filenames)):
	dset[i] = loadmat(p)['data']
