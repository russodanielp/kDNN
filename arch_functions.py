import numpy as np
import tensorflow as tf

from collections import OrderedDict


def layer_divide(input_matrix, num_fragments, num_node_list):
    """
    Divide input matrix into layers.
    Divide input matrix into layers.
    :param input_matrix: Matrix of chemical fragments and bioprofile
    :param num_fragments: Number of fragments in input layer
    :param num_node_list: List containing number of nodes in each hidden layer
    :return: List of pandas Series containing the data for each layer of the network
    """

    data = []

    start = 0
    end = 0
    num_nodes = [num_fragments] + num_node_list

    for nodes in num_nodes:
        end += nodes
        data.append(input_matrix.iloc[:, start:end])
        start = end

    return data


def create_architecture(num_chemicals, layer_data):
    """
    Set up architecture for kDNN
    :param num_chemicals: Number of chemicals in training set
    :param layer_data: Known data to feed into each network layer
    :return: feed_dict as dictionary, output,
    """

    placeholders = []
    initial_weights = []

    # print(num_chemicals)
    # print(layer_data[0].shape[1])

    placeholders.append(tf.placeholder(tf.float32, [num_chemicals, layer_data[0].shape[1]], name='chemical_inf'))
    initial_chem_weights = np.full((layer_data[0].shape[1], layer_data[1].shape[1]), fill_value=0.1, dtype='float32')
    initial_weights.append(tf.Variable(initial_chem_weights, name='chemical_W'))

    placeholder_names = [f'layer_{num}_bioassays' for num in range(1, len(layer_data) - 1)]
    weight_names = [f'layer_{num}_output_W' for num in range(1, len(layer_data) - 1)]
    counter = 1
    layer_output = None

    while counter < (len(layer_data) - 1):
        num_bioassays = layer_data[counter].shape[1]
        prev_idx = counter - 1
        layer_bioassays = tf.placeholder(tf.float32, [num_chemicals, num_bioassays], name=placeholder_names[prev_idx])
        dropout = tf.nn.dropout(layer_bioassays, 0.5)
        placeholders.append(dropout)
        initial_output = np.full((num_bioassays, layer_data[counter + 1].shape[1]), fill_value=0.1, dtype='float32')
        layer_output_weights = tf.Variable(initial_output, name=weight_names[prev_idx])
        initial_weights.append(layer_output_weights)

        if layer_output is None:
            layer_output = tf.multiply(tf.matmul(placeholders[prev_idx], initial_weights[prev_idx]), dropout)

        else:
            layer_output = tf.multiply(tf.matmul(layer_output, initial_weights[prev_idx]), dropout)

        counter += 1

    output = tf.nn.sigmoid(tf.matmul(layer_output, initial_weights[-1]), 'output')
    placeholders.append(tf.placeholder(tf.float32, [num_chemicals, 1]))

    return OrderedDict(zip(placeholders, layer_data)), output, initial_weights
