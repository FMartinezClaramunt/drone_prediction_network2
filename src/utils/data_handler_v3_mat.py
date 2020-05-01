"""
Data handler using TensorFlow input pipelines that directly uses the datasets in .mat format
"""

import os
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from scipy.linalg import hankel
from utils.model_utils import parse_mat_dataset_names, parse_input_types

class DataHandler():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.past_horizon = args.past_horizon
        self.prediction_horizon = args.prediction_horizon
        self.input_types = parse_input_types(args.input_types)
        
        self.datasets_training = parse_mat_dataset_names(args.datasets_training)
        if args.datasets_validation.lower() == "none":
            self.datasets_validation = []
        else:
            self.datasets_validation = parse_mat_dataset_names(args.datasets_validation)    
        if args.datasets_testing.lower() == "none":
            self.datasets_testing = []
        else:
            self.datasets_testing = parse_mat_dataset_names(args.datasets_testing)
        
        
        pass
    
    def getSampleInputBatch(self):
        """
        Get sample input batch full of zeros so that it can be used when calling the Keras model before loading the weights, either for testing or for a warm start
        """
        pass
        
    def getPlottingData(self, dataset_idx):
        
        pass
        # return {
        #     'trajectories': trajectories,
        #     'predictions': predictions,
        #     'goals': goals
        # }
        
        
# StackOverflow example
def read_mat(filepath):   
    def _read_mat(filepath):
        matfile = loadmat(filepath)
        data0 = matfile['data0']
        data1 = matfile['data1']
        data2 = matfile['data2']
        shape0 = matfile['data0'].shape
        return data0, data1, data2, np.asarray(shape0)

    output = tf.py_func(_read_mat, [filepath], [tf.double, tf.uint16, tf.double, tf.int64])
    shape = output[3]
    data0 = tf.reshape(output[0], shape)
    data1 = tf.reshape(output[1], shape)
    data2 = tf.reshape(output[2], shape)
    return data0, data1, data2

dataset = tf.data.Dataset.list_files('*.mat')
dataset = dataset.map(read_mat, num_parallel_calls=16)
dataset = dataset.repeat(100)
dataset = dataset.batch(8)
dataset = dataset.prefetch(8)
iterator = dataset.make_initializable_iterator()
sess = tf.Session()
sess.run(iterator.initializer)
values = sess.run(iterator.get_next())