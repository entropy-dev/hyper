import tensorflow as tf
from tqdm import tqdm
import numpy as np
from os import listdir
from os.path import join
from scipy.io import loadmat
from h5py import File


# global variables (config)
root_path = '/data/nir_data/2017_05_22/WithSpectralCorrection/'
data_path = root_path + 'PreprocessedImages/'
model_path = root_path + 'results/training01_tiedStructured/models/model_ae_tied_complex_loss-36129.meta'
output_path = root_path + 'results/training01_tiedStructured/inference.hdf5'

in_shape = (214, 407, 25)
compute_shape = (214 * 407, 25)
latent_shape = (214, 407, 3)

name_input = "x"
name_latent = "encoding_01/Tanh"
name_output = "decoder_01/Tanh"

# helper functions
norm = np.vectorize(lambda x: (x / 255) - 0.5)
denorm = np.vectorize(lambda x: (x + 0.5) * 255)


input_files = [join(data_path, i) for i in listdir(data_path) if i.endswith('.mat')]
num_files = len(input_files)

# create output
f = File(output_path, 'w')
ds_input = f.create_dataset('input_data', (num_files, *in_shape), np.float)
ds_latent = f.create_dataset('latent_data', (num_files, *latent_shape), np.float)
ds_output = f.create_dataset('output_data', (num_files, *in_shape), np.float)

with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
    tf.train.import_meta_graph(model_path).restore(sess, model_path[:-5])

    net_input = [i for i in sess.graph.get_operations() if name_input in i.name][0].outputs[0]
    net_latent = [i for i in sess.graph.get_operations() if name_latent in i.name][0].outputs[0]
    net_output = [i for i in sess.graph.get_operations() if name_output in i.name][0].outputs[0]

    for i, p in enumerate(tqdm(input_files)):
        data_input = loadmat(p)['data']

        feed_data = norm(data_input.reshape(compute_shape))

        data_latent, data_output = sess.run([net_latent, net_output], feed_dict={net_input: feed_data})

        data_latent = data_latent.reshape(latent_shape)
        data_output = denorm(data_output.reshape(in_shape))

        ds_input[i] = data_input
        ds_latent[i] = data_latent
        ds_output[i] = data_output

