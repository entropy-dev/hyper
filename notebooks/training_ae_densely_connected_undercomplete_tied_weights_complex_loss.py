import tensorflow as tf
from densely_connected_autoencoder_undercomplete_tied_weights_complex_loss import autoencoder
from DatasetStreamer import HyperspectralDatasetStreamer
from tqdm import tqdm
from os import makedirs
from os.path import dirname, exists
import numpy as np
from datetime import datetime

batch_size = 100000
n_epochs = 5
learning_rate = 0.001
ae_sizes = [25, 10, 10, 3]
structure_loss_rate = 0.02
weight_loss_rate = 0.0001

#path = '/data/nir_data/2017_05_22/NoSpectralCorrection/ShuffeledSpectra/HyperspectralDataNir20170522.hdf5'
path = '/data/nir_data/2017_05_22/WithSpectralCorrection/ShuffeledSpectra/HyperspectralDataNir20170522.hdf5'
test_px = 0.05


log_dir = '/tmp/logs/'
model_path =  '/tmp/models/model_ae_densely_tied_complex_loss'

makedirs(log_dir, exist_ok=True)
makedirs(dirname(model_path), exist_ok=True)

data = HyperspectralDatasetStreamer(path, test_px)

norm = np.vectorize(lambda x: (x / 255) - 0.5)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
g = tf.Graph()

with g.as_default():
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ae = autoencoder(g, ae_sizes)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope("overall_loss"):
            balance = tf.Variable(structure_loss_rate, name='structure_loss_rate', trainable=False)
            wsloss = balance * ae['loss'] * batch_size
            weight_loss = ae['weights_norm'] * weight_loss_rate

            overall_loss = ae['cost'] + wsloss + weight_loss

            tf.summary.scalar('weight_loss', weight_loss)
            tf.summary.scalar('structur_loss', wsloss)
            tf.summary.scalar('reconstruction_loss', ae["cost"])
            tf.summary.scalar('overall_loss', overall_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(overall_loss, global_step=global_step)

        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, g)

        tf.global_variables_initializer().run()

        current_time_step = datetime.now()

        print('\n\n\n')

        glob_rng = tqdm(range(n_epochs))
        glob_test_loss = np.nan


        for epoch_i in glob_rng:
            glob_rng.set_description('Epoch {:02} - {} - test loss {:.8f}'.format(epoch_i, 'training', glob_test_loss))

            rng = tqdm(data.train_epoch(batch_size), total=1457236638//batch_size)
            for x in rng:
                s,_,cst = sess.run([merged, optimizer, ae['cost']], feed_dict={ae['x']: norm(x)})
                rng.set_description('Cost: {:.8f}'.format(cst / batch_size))
                writer.add_summary(s, tf.train.global_step(sess, global_step))

                if (datetime.now() - current_time_step).seconds >= 600:
                    saver.save(sess, model_path, global_step=tf.train.global_step(sess, global_step))
                    current_time_step = datetime.now()

            rng.close()

            glob_rng.set_description('Epoch {:02} - {} - test loss {:.8f}'.format(epoch_i, 'saving', glob_test_loss))
            saver.save(sess, model_path, global_step=tf.train.global_step(sess, global_step))

            glob_rng.set_description('Epoch {:02} - {} - test loss {:.8f}'.format(epoch_i, 'testing', glob_test_loss))
            result = [sess.run([ae['cost']], feed_dict={ae['x']: norm(x)})[0] for x in data.test_epoch(batch_size)]
            glob_test_loss = sum(result) / (len(result)*batch_size)
