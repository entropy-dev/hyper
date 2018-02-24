import tensorflow as tf
import numpy as np
import math

# Autoencoder definition
def autoencoder(graph, dimensions=[784, 512, 256, 64]):
    """Build a deep autoencoder with tied weights.
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
        encoder = []
        for layer_i, n_output in enumerate(dimensions[1:]):
            with tf.variable_scope("encoding_{:02}".format(layer_i)):
                n_input = int(current_input.get_shape()[1])
                W = tf.Variable(
                    tf.random_uniform([n_input, n_output],
                                      -1.0 / math.sqrt(n_input),
                                      1.0 / math.sqrt(n_input)))
                b = tf.Variable(tf.zeros([n_output]))
                encoder.append(W)
                output = tf.nn.tanh(tf.matmul(current_input, W) + b)
                current_input = output

        # latent representation
        z = current_input

        # assert suitable respresentation
        with tf.variable_scope("structure_loss"):
            mean, std = tf.nn.moments(x, [0,1,2])
            structure_loss = tf.norm(mean) + tf.norm(std-0.25) # std(tanh(norm_pdf)) ~ 0.6
            
        encoder.reverse()

        # Build the decoder using the same weights
        for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
            with tf.variable_scope("decoder_{:02}".format(layer_i)):
                W = tf.transpose(encoder[layer_i])
                b = tf.Variable(tf.zeros([n_output]))
                output = tf.nn.tanh(tf.matmul(current_input, W) + b)
                current_input = output

        # now have the reconstruction through the network
        y = current_input

        # cost function measures pixel-wise difference
        with tf.variable_scope("reconstruction_loss"):
            cost = tf.reduce_sum(tf.square(y - x))
        return {'x': x, 'z': z, 'y': y, 'cost': cost, 'loss' : structure_loss}
