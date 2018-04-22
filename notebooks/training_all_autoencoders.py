from argparse import ArgumentParser
from datetime import datetime
from os import makedirs
from os.path import dirname

import numpy as np
import tensorflow as tf
from DatasetStreamer import HyperspectralDatasetStreamer
import densely_connected_autoencoder_undercomplete_tied_weights_complex_loss
from tqdm import tqdm


def get_inputs():
    ae_index_help = """ Id of the autoencoder used for training:
    0: tied weights
    1: tied weights, complex loss
    2: free weights
    3: free weights, complex loss
    4: free weights, complex loss, densely connected
    """

    parser = ArgumentParser(description='This script trains hyperspectral autoencoder.')
    parser.add_argument('-a', '--autoencoder-id', help=ae_index_help, required=True, type=int)
    parser.add_argument('-b', '--batch-size', help='Batch used for training and testing', default=100000, type=int)
    parser.add_argument('-d', '--dataset', help='Dataset to train on', required=True)
    parser.add_argument('-e', '--epochs', help='Number of epochs to train', default=5, type=int)
    parser.add_argument('-l', '--log-path', help='Path with filename to save logs to.', required=True, type=str)
    parser.add_argument('-m', '--model-path', help='Path with filename to save models to.', required=True, type=str)
    parser.add_argument('-r', '--learning-rate', help='Learning rate used for training', default=0.0001, type=float)
    parser.add_argument('-s', '--shape', help='Shape of the autoencoder e.g. "25,10,10,3"', required=True)
    parser.add_argument('-t', '--test', help='Percentage of the train data to test on.', default=0.1, type=float)

    return parser.parse_args()


def get_model(model_id, sizes, graph):
    if model_id is 0:  # tied weights
        pass
    elif model_id is 1:  # tied weights, complex loss
        pass
    elif model_id is 2:  # free weights
        pass
    elif model_id is 3:  # free weights, complex loss
        pass
    elif model_id is 4:  # free weights, complex loss, densely connected
        with graph.as_default():
            ae = densely_connected_autoencoder_undercomplete_tied_weights_complex_loss.autoencoder(g, sizes)

            weight_loss_rate = 0.0001
            structure_loss_rate = 0.02
            batch_size = tf.shape(ae['x'])[0]

            with tf.variable_scope("overall_loss"):
                balance = tf.Variable(structure_loss_rate, name='structure_loss_rate', trainable=False)
                wsloss = balance * ae['loss'] * batch_size
                weight_loss = ae['weights_norm'] * weight_loss_rate

                overall_loss = ae['cost'] + wsloss + weight_loss

                tf.summary.scalar('weight_loss', weight_loss)
                tf.summary.scalar('structur_loss', wsloss)
                tf.summary.scalar('reconstruction_loss', ae["cost"])
                tf.summary.scalar('overall_loss', overall_loss)

            merged = tf.summary.merge_all()

            return ae['x'], overall_loss, merged
    else:
        raise RuntimeError("Unknown model with id: {}".format(model_id))


def get_data(path, test_px):
    return HyperspectralDatasetStreamer(path, test_px)


def train(data, graph, model, batch_size, n_epochs, learning_rate, log_path, model_path):
    model_input, model_loss, model_summary = model
    makedirs(log_path, exist_ok=True)
    makedirs(dirname(model_path), exist_ok=True)

    norm = np.vectorize(lambda vec: (vec / 255) - 0.5)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)

    with graph.as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model_loss, global_step=global_step)
            saver = tf.train.Saver()
            writer = tf.summary.FileWriter(log_path, graph)
            tf.global_variables_initializer().run()
            current_time_step = datetime.now()

            overall_range = tqdm(range(n_epochs))
            overall_test_loss = np.nan

            for epoch_i in overall_range:
                overall_range.set_description('Epoch {:02} - {} - test loss {:.8f}'.format(epoch_i, 'training',
                                                                                           overall_test_loss))

                epoch_range = tqdm(data.train_epoch(batch_size), total=data.get_train_size() // batch_size)
                for x in epoch_range:
                    s, _, cst = sess.run([model_summary, optimizer, model_loss], feed_dict={model_input: norm(x)})
                    epoch_range.set_description('Cost: {:.8f}'.format(cst / batch_size))
                    writer.add_summary(s, tf.train.global_step(sess, global_step))

                    if (datetime.now() - current_time_step).seconds >= 600:
                        saver.save(sess, model_path, global_step=tf.train.global_step(sess, global_step))
                        current_time_step = datetime.now()

                epoch_range.close()

                overall_range.set_description('Epoch {:02} - {} - test loss {:.8f}'.format(epoch_i, 'saving',
                                                                                           overall_test_loss))
                saver.save(sess, model_path, global_step=tf.train.global_step(sess, global_step))

                overall_range.set_description('Epoch {:02} - {} - test loss {:.8f}'.format(epoch_i, 'testing',
                                                                                           overall_test_loss))
                result = [sess.run(model_loss, feed_dict={model_input: norm(x)}) for x in data.test_epoch(batch_size)]
                overall_test_loss = sum(result) / (len(result) * batch_size)

def main(data_path, test_px, network_shape, network_id, batch_size, epochs, learning_rate, log_path, model_path):
    g = tf.Graph()
    n = get_model(network_id, network_shape, g)
    data = get_data(data_path, test_px)
    train(data, g, n, batch_size, epochs, learning_rate, log_path, model_path)


if __name__ == '__main__':
    inputs = get_inputs()
    ae_shape = [int(i) for i in inputs.shape.split(',')]

    main(data_path=inputs.dataset, test_px=inputs.test,network_shape=ae_shape, network_id=inputs.autoencoder_id,
         batch_size=inputs.batch_size, epochs=inputs.epochs, learning_rate=inputs.learning_rate,
         log_path=inputs.log_path, model_path=inputs.model_path)
