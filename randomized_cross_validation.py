import os

import numpy as np
import pandas as pd
import tensorflow as tf

from arch_functions import create_architecture, layer_divide
from generate_input import cache_data, load_data

from argparse import ArgumentParser
from datetime import datetime

from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold

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
true_profiles, true = load_data(dataset, directory, assay_file, endpoint, fingerprints, identifier=identifier)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for i in range(100):
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')

    # define number of chemicals and fragments being used for training
    num_chemicals = true_profiles.shape[0]
    num_fragments = true_profiles.shape[1] - sum(num_nodes)
    assays = list(true_profiles.columns[num_fragments:])

    # shuffle in vivo data
    in_vivo = true.sample(frac=1)
    in_vivo.index = true_profiles.index
    # print(type(in_vivo))

    # shuffles data within assays
    randomized_profiles = pd.concat([pd.Series(true_profiles[assay].sample(frac=1).values, name=assay, index=true_profiles.index)
                                     for assay in assays], axis=1)
    randomized_profiles.index = true_profiles.index
    randomized_profiles = pd.concat([true_profiles.iloc[:, :num_fragments], randomized_profiles], axis=1)

    cache_data(randomized_profiles, in_vivo, directory, dataset + now, endpoint, assay_file, fingerprints)

    # set up cross-validation
    cv = KFold(shuffle=True, n_splits=42, random_state=0)
    predictions = []

    # do n iterations for n chemicals of the training procedure
    for train_idx, val_idx in cv.split(randomized_profiles, in_vivo):
        # define number of chemicals and fragments being used for training
        num_chemicals = randomized_profiles.iloc[train_idx].shape[0]
        num_fragments = randomized_profiles.shape[1] - sum(num_nodes)

        # divide data into layers for the kDNN model
        split_data = layer_divide(randomized_profiles.iloc[train_idx], num_fragments, num_nodes)
        split_data.append(in_vivo.iloc[train_idx].values.reshape((randomized_profiles.iloc[train_idx].shape[0], 1)))
        feed_dict, output, weights = create_architecture(num_chemicals, split_data)
        y_true = list(feed_dict.items())[-1][0]

        # define training parameters (cost function, metric function, overfitting protection
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
        root_logdir = 'tf_logs'
        logdir = f'{root_logdir}/run-{now}'

        # initialize variables used to monitor and save model training
        lr_wait = 0
        stop_wait = 0
        best_cost = np.inf
        lr = 0.01

        init = tf.global_variables_initializer()

        train = True
        epoch = 0

        with tf.Session() as sess:
            sess.run(init)

            while train:
                # optimize in single epoch
                feed_dict[lr_placeholder] = lr
                sess.run(optimizer, feed_dict=feed_dict)
                current_cost = sess.run(cost, feed_dict=feed_dict)

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
                            lr_wait = 0

                    # if over 200 epochs go by without any change, stop model training early to avoid
                    # overfitting
                    if stop_wait >= 200:
                        stopped_epoch = epoch
                        train = False

            paths = []
            outputs = []

            # make test prediction for left out compound
            for cpd in randomized_profiles.iloc[val_idx].index:
                test_split_data = [data.loc[cpd].astype('float32').values.reshape(1, data.shape[1]) for data in
                                   layer_divide(randomized_profiles.iloc[val_idx], num_fragments, num_nodes)]
                counter = 1
                current_output = None
                path = []

                while counter < len(test_split_data):
                    prev_idx = counter - 1

                    if current_output is None:
                        current_output = tf.multiply(tf.matmul(test_split_data[prev_idx], weights[prev_idx]),
                                                     test_split_data[counter])

                    else:
                        current_output = tf.multiply(tf.matmul(current_output, weights[prev_idx]), test_split_data[counter])

                    layer_assays = list(split_data[prev_idx + 1].columns)
                    node_vals = current_output.eval()
                    counter += 1
                    path.append(layer_assays[np.argmax(node_vals)])

                output = tf.nn.sigmoid(tf.matmul(current_output, weights[-1]))
                paths.append(path)
                outputs.append(output.eval()[0][0])

            path_df = pd.DataFrame(paths, columns=[f'Assay {n}' for n in range(1, 6)], index=randomized_profiles.iloc[val_idx].index)
            outputs = pd.Series(outputs, index=randomized_profiles.iloc[val_idx].index)
            final_preds = pd.concat([path_df, outputs], axis=1)
            predictions.append(final_preds)

        sess.close()
        del sess

    tf.reset_default_graph()

    # save cross-validation predictions
    pd.concat(predictions).to_csv('data', f'{dataset + now}_{assay_file}_loocv_predictions.csv')

    fpr, tpr, thresholds = roc_curve(true, pd.concat(predictions).iloc[:, -1])
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    print(i, roc_auc)

# calculate statistics
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
std_tpr = np.std(tprs, axis=0)
pd.Series(aucs).to_csv('aucs_100_randomizations.csv')

# save data for plotting purposes
for_plotting = pd.concat([pd.Series(mean_fpr), pd.Series(mean_tpr)])
for_plotting.to_csv('data', f'mean_roc_100_randomizations.csv')
