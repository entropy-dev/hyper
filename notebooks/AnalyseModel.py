import tensorflow as tf
from tqdm import tqdm
import numpy as np
from os import listdir
from os.path import join
from scipy.io import loadmat, savemat

data_path = '/data/nir_data/2017_05_22/NoSpectralCorrection/PreprocessedImages/'
model_path =  '/data/nir_data/2017_05_22/NoSpectralCorrection/results/training03_tiedStructuredImproved/models/model_ae_tied_complex_loss-71400.meta'
output_path = '/data/nir_data/2017_05_22/NoSpectralCorrection/results/training03_tiedStructuredImproved/processed_cubes/'

in_shape = (214, 407, 25)
compute_shape = (214 * 407, 25)
out_shape = (214, 407, 3)

norm = np.vectorize(lambda x: (x / 255) - 0.5)

input_files = [(join(data_path, i),i) for i in listdir(data_path) if i.endswith('.mat')]

with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
    tf.train.import_meta_graph(model_path).restore(sess, model_path[:-5])
    net_output = [i for i in sess.graph.get_operations() if "encoding_01/Tanh" in i.name][0].outputs[0]
    net_input = [i for i in sess.graph.get_operations() if "x" in i.name][0].outputs[0]

    for p,s in tqdm(input_files):

        in_data = norm(loadmat(p)['data'].reshape(compute_shape))

        result = sess.run(net_output, feed_dict={net_input: in_data})


        savemat(join(output_path, s), {'data' : result.reshape(out_shape)})

