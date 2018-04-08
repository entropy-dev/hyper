import tensorflow as tf
from tqdm import tqdm
import numpy as np
from os import listdir
from os.path import join
from scipy.io import loadmat, savemat
import pickle


# helper function
def compute_properties(data_in, data_out, data_latent, hist_edges=None):
    def local_properties(data, hist_bins):
        data_abs = np.abs(data)

        r_abs = data_abs.sum()
        r_squ = np.square(data).sum()
        r_mean = np.mean(data)
        r_sigma = np.std(data)
        r_min = data_abs.min()
        r_max = data_abs.max()
        r_hist, _ = np.histogram(data, bins=hist_bins)

        return r_abs, r_squ, r_mean, r_sigma, r_min, r_max, r_hist

    def volume_properties(volume, hist_bins):
        # overall
        result = [local_properties(volume, hist_bins)]

        # bands
        for band in np.array(volume.T, order='C'):
            result.append(local_properties(band, hist_bins))

        return result

    if not hist_edges:
        hist_edges = np.concatenate(([-2], np.linspace(-1, 1, 256 + 1), [2]))

    return [
        volume_properties(data_in, hist_edges),  # -.5 - .5
        volume_properties(data_out, hist_edges),  # -1 - 1
        volume_properties(data_latent, hist_edges),  # -1 - 1
        volume_properties(data_in - data_out, hist_edges),  # -2 - 2 #
    ]


# global variables (config)
data_path = '/data/nir_data/2017_05_22/NoSpectralCorrection/PreprocessedImages/'
model_path = '/data/nir_data/2017_05_22/NoSpectralCorrection/results/training03_tiedStructuredImproved/models/model_ae_tied_complex_loss-71400.meta'
output_path = '/data/nir_data/2017_05_22/NoSpectralCorrection/results/training03_tiedStructuredImproved/Props.pickle'

in_shape = (214, 407, 25)
compute_shape = (214 * 407, 25)
latent_shape = (214, 407, 3)

name_input = "x"
name_latent = "encoding_01/Tanh"
name_output = "decoder_01/Tanh"

norm = np.vectorize(lambda x: (x / 255) - 0.5)
denorm = np.vectorize(lambda x: (x + 0.5) * 255)

input_files = [(join(data_path, i), i) for i in listdir(data_path) if i.endswith('.mat')]

with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
    tf.train.import_meta_graph(model_path).restore(sess, model_path[:-5])

    net_input = [i for i in sess.graph.get_operations() if name_input in i.name][0].outputs[0]
    net_latent = [i for i in sess.graph.get_operations() if name_latent in i.name][0].outputs[0]
    net_output = [i for i in sess.graph.get_operations() if name_output in i.name][0].outputs[0]

    property_buffer = []

    for p, s in tqdm(input_files):
        data_input = loadmat(p)['data']

        feed_data = norm(data_input.reshape(compute_shape))

        data_latent, data_output = sess.run([net_latent, net_output], feed_dict={net_input: feed_data})

        data_latent = data_latent.reshape(latent_shape)
        data_output = denorm(data_output.reshape(in_shape))

        property_buffer.append(compute_properties(data_input, data_output, data_latent))

with open(output_path, 'wb') as f:
    pickle.dump(property_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)

