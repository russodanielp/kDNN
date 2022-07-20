import os

import pandas as pd
import tensorflow as tf

from arch_functions import layer_divide
from generate_input import get_assay_list, load_qsar_dataset

from argparse import ArgumentParser

# define input parameters
parser = ArgumentParser(description='Get step by step layer output for predictions using weights from training')
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
parser.add_argument('-ip', '--identifier_predict', metavar='ip', type=str,
                    help='Unique identifier for compound to predict')
parser.add_argument('-nn', '--num_nodes', metavar='nn', type=str,
                    help='Csv string with the number of nodes in each hidden layer, in order')
parser.add_argument('-ts', '--training_set', metavar='ts', type=str,
                    help='Name of training set used for model (excluding .sdf)')

# collect user inputs into variables
args = parser.parse_args()
assay_file = args.assay_file
dataset = args.dataset
endpoint = args.endpoint
directory = 'data'
fingerprints = args.fingerprints
identifier = args.identifier
identifier_predict = args.identifier_predict
num_nodes = [int(num) for num in args.num_nodes.split(',')]
training_set = args.training_set

cache_name = f'{training_set}_{endpoint}_{assay_file}_{fingerprints}'

# load assay names and input data (chemical and biological)
assays = get_assay_list(os.path.join(directory, f'{assay_file}.csv'))
fragments, profiles = load_qsar_dataset(dataset, directory, assays, features=fingerprints,
                                        identifier=identifier_predict)

all_input_data = pd.concat([fragments, profiles], axis=1)

# define number of chemicals in the test set and fragments used for training
num_chemicals = all_input_data.shape[0]
num_fragments = all_input_data.shape[1] - sum(num_nodes)

# load weights learned during model training
weights = [pd.read_csv(os.path.join(directory, f'{cache_name}_{kind}_W.csv'), header=0, index_col=0).astype(
        'float64') for kind
        in ['chemical', 'layer_1_output', 'layer_2_output', 'layer_3_output', 'layer_4_output', 'output']]
preds = []

# predict each compound
for cpd in fragments.index:
    # divide up the compound's data into layers for input to the k-DNN
    split_data = [data.loc[cpd].astype('float64').values.reshape(1, data.shape[1]) for data in
                  layer_divide(all_input_data, num_fragments, num_nodes)]

    counter = 1
    current_output = None

    with tf.Session() as sess:
        while counter < len(split_data):
            prev_idx = counter - 1

            if current_output is None:
                current_output = tf.multiply(tf.matmul(split_data[prev_idx], weights[prev_idx]), split_data[counter])

            else:
                current_output = tf.multiply(tf.matmul(current_output, weights[prev_idx]), split_data[counter])

            counter += 1

        output = tf.nn.sigmoid(tf.matmul(current_output, weights[-1]))
        print(cpd, output.eval())
        preds.append(output.eval()[0][0])

# write all predictions for the dataset to a file
pred_file = pd.DataFrame(preds, index=fragments.index)
pred_file.columns = ['Probability']
pred_file['Class'] = [int(prob > 0.5) for prob in pred_file.Probability]
pred_file.to_csv(f'{dataset}_predictions_more_nodes.csv')
