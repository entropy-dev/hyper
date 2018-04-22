import tensorflow as tf
import numpy as np
import math

def dense_layer(input_a, input_b, n_output, name, weights=None, activation=tf.nn.tanh):
    with tf.variable_scope(name):
        layer_input = tf.concat([input_a, input_b], 1) if input_b is not None else input_a
        n_input = int(layer_input.get_shape()[1])
        
        
        W = tf.Variable(tf.random_uniform([n_input, n_output],
                                          -1.0 / math.sqrt(n_input),
                                          1.0 / math.sqrt(n_input))) if weights is None else weights
        b = tf.Variable(tf.zeros([n_output]))
        output = activation(tf.matmul(layer_input, W) + b)
        
        return layer_input, output, W, b
        

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
        weights = []
        biases = []
        last_input = None
        
        for layer_i, n_output in enumerate(dimensions[1:]):
            last_input, output, W, b = dense_layer(current_input, last_input, n_output, "encoding_{}".format(layer_i))
            current_input = output            
            weights.append(W)
            biases.append(b)

        # latent representation
        z = current_input

        # assert suitable respresentation
        with tf.variable_scope("structure_loss"):
            mean, std = tf.nn.moments(z, [0])
            structure_loss = tf.reduce_sum(tf.square(mean)) + tf.reduce_sum(tf.square(std-0.1)) # std(tanh(norm_pdf)) ~ 0.6

        last_input = None
        # Build the decoder using the same weights
        for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
            last_input, output, W, b = dense_layer(current_input, last_input, n_output, "decoding_{}".format(layer_i))
            current_input = output            
            weights.append(W)
            biases.append(b)

        # now have the reconstruction through the network
        y = current_input

        # cost function measures pixel-wise difference
        with tf.variable_scope("reconstruction_loss"):
            cost = tf.reduce_sum(tf.square(y - x))
            
        with tf.variable_scope('weights_norm') as scope:
            weights_norm = tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in weights]))

        return {'x': x, 'z': z, 'y': y,
                'cost': cost, 'loss' : structure_loss,
                'weights' : weights, 'biases' : biases,
                'weights_norm': weights_norm}
