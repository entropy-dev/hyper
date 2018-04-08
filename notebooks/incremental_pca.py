from pickle import dump
from h5py import File
from sklearn.decomposition import IncrementalPCA
from tqdm import trange

# global variables
batch_size = 1024 ** 2
input_path = '/data/nir_data/2017_05_22/NoSpectralCorrection/ShuffeledSpectra/HyperspectralDataNir20170522.hdf5'
output_path = '/data/nir_data/2017_05_22/NoSpectralCorrection/incremental_pca.pickle'
text_px = 0.02

# input data
f = File(input_path, 'r')
data = f['data']
num_samples = data.shape[0] * (1.0 - text_px)

# model
model = IncrementalPCA(n_components=3, batch_size=batch_size)

# fitting
for i in trange(0, int(num_samples // batch_size)):
    model.partial_fit(data[i * batch_size: (i + 1) * batch_size])

# output data
with open(output_path, 'wb') as output_file:
    dump(model, output_file)

# testing
# TODO

