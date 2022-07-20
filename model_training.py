import os

import numpy as np
import pandas as pd
import tensorflow as tf

from arch_functions import create_architecture, layer_divide
from generate_input import load_data

from argparse import ArgumentParser
from datetime import datetime

# define input parameters
parser = ArgumentParser(description='Train deep vAOPs using in vitro and in vivo data')
parser.add_argument('-af', '--assay_file', metavar='af', type=str,
                    help='Name of file containing list of assays of interest')
parser.add_argument('-ds', '--dataset', metavar='ds', type=str,
                    help='Dataset sdf file name (do not include .sdf extension)')
parser.add_argument('-ep', '--endpoint', metavar='ep', type=str,
                    help='Molecule property containing in vivo outcome of interest')
parser.add_argument('-f', '--fingerprints', metavar='f', type=str,
                    help='Fingerprints to use as model input')
parser.add_argument('-i', '--identifier', metavar='i', type=str,
                    help='Name of unique identifier field in sdf file')
parser.add_argument('-nn', '--num_nodes', metavar='nn', type=str,
                    help='Csv string with the number of nodes in each hidden layer, in order')

# collect user inputs into variables
args = parser.parse_args()
assay_file = args.assay_file
dataset = args.dataset
endpoint = args.endpoint
directory = 'data'
fingerprints = args.fingerprints
identifier = args.identifier
num_nodes = [int(num) for num in args.num_nodes.split(',')]

# load training data (in vitro and in vivo)
profiles, in_vivo = load_data(dataset, directory, assay_file, endpoint, fingerprints, identifier=identifier)

# define number of chemicals and fragments being used for training
num_chemicals = profiles.shape[0]
num_fragments = profiles.shape[1] - sum(num_nodes)

# divide data into layers for the kDNN model
split_data = layer_divide(profiles, num_fragments, num_nodes)
print(split_data[0].shape, split_data[1].shape)
split_data.append(in_vivo.values.reshape((profiles.shape[0], 1)))
feed_dict, output, weights = create_architecture(num_chemicals, split_data)
y_true = list(feed_dict.items())[-1][0]

# define training parameters (cost function, metric function, overfitting protection)
cost = tf.reduce_mean(tf.keras.backend.binary_crossentropy(y_true, output))
regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
accuracy = tf.reduce_mean(tf.keras.metrics.binary_accuracy(y_true, output) + 0.001 * sum(regularization_losses))
lr_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
cost_summary = tf.summary.scalar('cost', cost)
accuracy_summary = tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

# set training algorithm to stochastic gradient descent
optimizer = tf.train.MomentumOptimizer(lr_placeholder, 0.9, use_nesterov=False).minimize(cost)

# setup TensorFlow logs
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = f'{root_logdir}/run-{now}'

# initialize variables used to monitor and save model training
lr_wait = 0
stop_wait = 0
best_cost = np.inf
lr = 0.01

init = tf.global_variables_initializer()

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
train = True
epoch = 0
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    while train:
        # optimize in single epoch
        feed_dict[lr_placeholder] = lr
        sess.run(optimizer, feed_dict=feed_dict)
        current_cost = sess.run(cost, feed_dict=feed_dict)

        # print training progress every 50 epochs
        if epoch % 50 == 0:
            print('cost', current_cost)
            print('acc', sess.run(accuracy, feed_dict=feed_dict))
            summary, _ = sess.run([merged, optimizer], feed_dict=feed_dict)
            file_writer.add_summary(summary, epoch)

        epoch += 1

        # cap training at 100,000 epochs
        if epoch == 100001:
            train = False

        # monitor how many epochs go by without improvement in model performance
        if np.less(current_cost, best_cost - 0.0001):
            best_cost = current_cost
            lr_wait = 0
            stop_wait = 0

        else:
            lr_wait += 1
            stop_wait += 1

            # decrease learning rate if over 50 epochs go by with no improvement, but no lower than 0.00001
            if lr_wait > 50:
                if lr > 0.00001:
                    lr = max(lr * 0.9, 0)
                    print(f'reducing learning rate to {lr}')
                    lr_wait = 0

            # if over 200 epochs go by without any change, stop model training early to avoid
            # overfitting
            if stop_wait >= 200:
                stopped_epoch = epoch
                print(f'early stopping on epoch {epoch}')
                #print('output', pd.DataFrame(sess.run(output, feed_dict=feed_dict)), index=profiles.index)
                train = False

    # save model training information
    saver.save(sess, os.path.join('.', f'{dataset}_{endpoint}_{assay_file}_{fingerprints}'))

    file_writer.close()

    # save edge weights for each layer of the kDNN
    for weight, counter in zip(weights[:-1], range(len(split_data) - 1)):
        w = pd.DataFrame(sess.run(weight), index=split_data[counter].columns, columns=split_data[counter + 1].columns)
        w.to_csv(os.path.join(directory, f'{dataset}_{endpoint}_{assay_file}_{fingerprints}_{weight.name[:-2]}.csv'))

    # save output edge weights
    output_W = pd.DataFrame(sess.run(weights[-1]), index=split_data[-2].columns)
    output_W.to_csv(os.path.join(directory, f'{dataset}_{endpoint}_{assay_file}_{fingerprints}_output_W.csv'))
