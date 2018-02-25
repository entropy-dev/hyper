import tensorflow as tf
from autoencoder_undercomplete_tied_weights_complex_loss import autoencoder
from DatasetStreamer import HyperspectralDatasetStreamer
from tqdm import tqdm
from os import makedirs
from os.path import dirname, exists
import numpy as np
from datetime import datetime

batch_size = 100000
n_epochs = 5
learning_rate = 0.001
ae_sizes = [25, 10, 3]
structure_loss_rate = 0.1

path = '/home/sflorian92/output_data/HyperspectralDataNir20170522.hdf5'
test_px = 0.02


log_dir = '/tmp/logs/'
model_path =  '/tmp/models/model_ae_tied_complex_loss'

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

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            ae['cost']+structure_loss_rate*ae['loss'], global_step=global_step)
        tf.summary.scalar('cost', ae["cost"])
        tf.summary.scalar('loss', ae["loss"])
        
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, g)

        tf.global_variables_initializer().run()

        current_time_step = datetime.now()

        for epoch_i in tqdm(range(n_epochs)):
            print("\nRunning epoch: {:02}\n".format(epoch_i))

            rng = tqdm(data.train_epoch(batch_size))
            for x in rng:
                s,_,cst = sess.run([merged, optimizer, ae['cost']], feed_dict={ae['x']: norm(x)})
                rng.set_description('Cost: {:.8f}'.format(cst / batch_size))
                writer.add_summary(s, global_step)
                
                if (datetime.now() - current_time_step).seconds >= 600:
                    saver.save(sess, model_path, global_step=global_step)
                    current_time_step = datetime.now()
            
            rng.close()

            print("\nSaving to {}.\n".format(model_path.format(epoch_i)))
            saver.save(sess, model_path, global_step=global_step)

            print("\nTesting.\n")
            result = [sess.run([ae['cost']], feed_dict={ae['x']: norm(x)})[0] for x in data.test_epoch(batch_size)]
            print("\nCost: {:.8f}\n".format(
                    sum(result) / (len(result)*batch_size)
                ))
