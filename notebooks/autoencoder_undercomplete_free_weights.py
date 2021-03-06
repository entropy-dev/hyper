import tensorflow as tf
import numpy as np
import math

# Autoencoder definition
def autoencoder(graph, dimensions=[784, 512, 256, 64]):
    """Build a deep autoencoder with free weights.
    Parameters
    ----------
    graph : tf.Graph
        The tensorflow computation graph
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    
    with graph.as_default():

        # input to the network
        x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
        current_input = x

        # Build the encoder
        for layer_i, n_output in enumerate(dimensions[1:]):
            n_input = int(current_input.get_shape()[1])
            W = tf.Variable(
                tf.random_uniform([n_input, n_output],
                                  -1.0 / math.sqrt(n_input),
                                  1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([n_output]))
            output = tf.nn.tanh(tf.matmul(current_input, W) + b)
            current_input = output

        # latent representation
        z = current_input

        # Build the decoder using the same weights
        for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
            n_input = int(current_input.get_shape()[1])
            W = tf.Variable(
                tf.random_uniform([n_input, n_output],
                                  -1.0 / math.sqrt(n_input),
                                  1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([n_output]))
            output = tf.nn.tanh(tf.matmul(current_input, W) + b)
            current_input = output

        # now have the reconstruction through the network
        y = current_input

        # cost function measures pixel-wise difference
        cost = tf.reduce_sum(tf.square(y - x))
        return {'x': x, 'z': z, 'y': y, 'cost': cost}
