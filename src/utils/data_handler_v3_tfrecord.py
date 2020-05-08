"""
Data handler using TensorFlow input pipelines with .tfrecord files as input
"""
import os
import copy
from pathlib import Path
from glob import glob
import numpy as np
import tensorflow as tf
import pickle as pkl
from scipy.io import loadmat
from scipy.linalg import hankel
#import sklearn.preprocessing as skp
from utils.model_utils import parse_dataset_names, parse_input_types

class DataHandler():
    def __init__(self, args):
        # Data parameters
        self.dt = args.dt
        self.train = args.train
        self.batch_size = args.batch_size
        self.past_horizon = args.past_horizon
        self.prediction_horizon = args.prediction_horizon
        self.separate_goals = args.separate_goals

        # TODO: implement the processing of different data types, as of now it considers velocity query input, full state others input, no obstacles input and velocity target
        self.data_types = {
            "query_input_type": args.query_input_type.lower(),
            "others_input_type": args.others_input_type.lower(),
            "obstacle_input_type": args.obstacle_input_type.lower(),
            "target_type": args.target_type.lower()
        }
        self.common_params = {
            "past_horizon": self.past_horizon,
            "prediction_horizon": self.prediction_horizon,
            "separate_goals": self.separate_goals,
            "data_types": self.data_types
        }
        self.data_key_list = self.getKeyList()
        
        # Data directories
        self.raw_data_dir = args.raw_data_dir
        self.tfrecord_data_dir = args.tfrecord_data_dir
        if not os.path.isdir(self.tfrecord_data_dir):
            Path(self.tfrecord_data_dir).mkdir(parents=True, exist_ok=True)
        
        # Training datasets
        self.datasets_training = parse_dataset_names(args.datasets_training)
        self.tfrecords_training = self.getTFRecords(self.datasets_training)

        self.scaler = self.getScaler() # Get scaler based on the data for the first quadrotor of the first training dataset
        self.n_quadrotors = self.getNQuadrotors() # Get total number of quadrotors 

        # Get dataset object for the training data
        self.tfdataset_training = tf.data.TFRecordDataset(self.tfrecords_training).apply(self.dataset_setup)
        
        # Validation datasets
        if args.datasets_validation.lower() == "none":
            self.datasets_validation = []
            self.tfdataset_validation = None
        else:
            self.datasets_validation = parse_dataset_names(args.datasets_validation)
            self.tfrecords_validation = self.getTFRecords(self.datasets_validation)
            self.tfdataset_validation = tf.data.TFRecordDataset(self.tfrecords_validation).apply(lambda x: self.dataset_setup(x, shuffle=False))
        
        # Test datasets
        if args.datasets_testing.lower() == "none":
            self.datasets_testing = []
        else:
            self.datasets_testing = parse_dataset_names(args.datasets_testing)
            self.tfrecords_testing = self.getTFRecords(self.datasets_testing)
            self.tfdataset_testing = tf.data.TFRecordDataset(self.tfrecords_testing).apply(lambda x: self.dataset_setup(x, shuffle=False))
    
    def getSampleInputBatch(self): # TODO: Probably remove, find replacement
        """
        Get sample input batch so that it can be used when calling the Keras model before loading the weights, either for testing or for a warm start
        """
        if self.train:
            sample_batch = next(iter(self.tfdataset_training))
        else:
            raw_dataset_path = os.path.join(self.raw_data_dir, self.datasets_testing[0] + '.mat')
            data = loadmat(raw_dataset_path)
            sample_batch = self.preprocess_data(data, 0)
        
        return sample_batch
        
    def getPlottingData(self, trained_model, test_dataset_idx = 0, quads_to_plot = -1):
        """
        Takes converts data from the test set into data ready for plotting
        """
        raw_dataset_path = os.path.join(self.raw_data_dir, self.datasets_testing[test_dataset_idx] + '.mat')
        data = loadmat(raw_dataset_path)
        
        goal_array = data['log_quad_goal'] # [goal pose (4), timesteps, quadrotors] 
        state_array = data['log_quad_state_real'] # [states (9), timesteps, quadrotors] 
        logsize = int(data['logsize'])
        n_quads = goal_array.shape[2]

        # Find last time step which can be used for training
        final_timestep = find_last_usable_step(goal_array, logsize, n_quads)

        trajs = np.swapaxes(state_array[0:3, 0:final_timestep, :], 0, 1)
        goals = np.swapaxes(goal_array[0:3, 0:final_timestep, :], 0, 1)

        position_predictions = np.zeros((trajs.shape[0]-self.prediction_horizon-self.past_horizon+1,\
                                        self.prediction_horizon+1,\
                                        3,\
                                        n_quads))

        if quads_to_plot == -1: # Plot all quads
            quads_to_plot = [idx for idx in range(n_quads)]
        elif type(quads_to_plot) == int:
            quads_to_plot = [quads_to_plot]

        for quad_idx in quads_to_plot:
            data_dict = self.preprocess_data(data, quad_idx, relative=True)
            scaled_data_dict = self.scaler.transform(data_dict)
            scaled_velocity_predictions = trained_model.predict(scaled_data_dict)
            scaled_data_dict["target"] = scaled_velocity_predictions
            unscaled_data = self.scaler.inverse_transform(scaled_data_dict)
            unscaled_velocity_predictions = unscaled_data["target"]
            
            position_predictions[:, 0, :, quad_idx] = trajs[self.past_horizon:-self.prediction_horizon+1, :, quad_idx]
            for timestep in range(1, self.prediction_horizon+1):
                position_predictions[:, timestep, :, quad_idx] = position_predictions[:, timestep-1, :, quad_idx] \
                                                                + unscaled_velocity_predictions[:, timestep-1, :] * self.dt
        
        return {
            'trajectories': trajs,
            'predictions': position_predictions,
            'goals': goals
        }
        
    def preprocess_data(self, data_, query_quad_idx, relative = True, separate_goals = False):
        processed_data_dict = {}
        
        data = copy.deepcopy(data_)
        
        # Extract data from datafile
        goal_array = data['log_quad_goal'] # [goal pose (4), timesteps, quadrotors] 
        state_array = data['log_quad_state_real'] # [states (9), timesteps, quadrotors] 
        logsize = int(data['logsize'])
        n_quads = goal_array.shape[2]

        # Find last time step which can be used for training
        final_timestep = find_last_usable_step(goal_array, logsize, n_quads)
        past_timesteps_idxs = [idx for idx in range(0, final_timestep - self.prediction_horizon)]
        future_timesteps_idxs = [idx for idx in range(self.past_horizon, final_timestep)]
        
        other_quad_idxs = [idx for idx in range(n_quads) if idx != query_quad_idx]
                
        # Add first element of the list of inputs, which corresponds to the query agent's data
        query_input_data = state_array[3:6,\
                                    past_timesteps_idxs,\
                                    query_quad_idx]
        query_input = expand_sequence(query_input_data, self.past_horizon)
        processed_data_dict["query_input"] = query_input
        
        # Add second element to the list of inputs, which is the list of other agent's data
        others_input_data = state_array[0:6,\
                                        past_timesteps_idxs,\
                                        :]

        if relative:
            query_agent_curr_pos = state_array[0:3, past_timesteps_idxs, query_quad_idx:query_quad_idx+1]
            others_input_data[0:3, :, :] = others_input_data[0:3, :, :] - query_agent_curr_pos # Relative positions to the query agent

        others_input_list = []
        for quad_idx in other_quad_idxs:
            other_quad_sequence = expand_sequence(others_input_data[:,:,quad_idx], self.past_horizon)
            others_input_list.append(other_quad_sequence)
        others_input = np.stack(others_input_list, axis=-1) # Stack along last dimension
        processed_data_dict["others_input"] = others_input

        # Slice state_array to build target_data
        target_data = state_array[3:6,\
                                future_timesteps_idxs,\
                                query_quad_idx:query_quad_idx + 1]

        # Expand target feature sequences
        target = expand_sequence(target_data, self.prediction_horizon)
        processed_data_dict["target"] = target
        
        if separate_goals: # Separate trajectories by goals
            goal_data = goal_array[0:3,\
                                    0:final_timestep,\
                                    query_quad_idx]
            goal_data_hankel = expand_sequence(goal_data, self.past_horizon + self.prediction_horizon)
            same_goals_idxs = np.nonzero(np.all(goal_data_hankel == goal_data_hankel[:, 0:1, :], axis=(1,2)))[0] # Compare all goals to the first one in the sequence and get the indexes of the dataset samples in which it is true for all elements of the sequence

            for key in processed_data_dict.keys():
                processed_data_dict[key] = processed_data_dict[key][same_goals_idxs,]
        
        return processed_data_dict
    
    def makeTFRecords(self, dataset_name):
        """
        Makes TFRecords out of a raw dataset 
        """
        raw_dataset_path = os.path.join(self.raw_data_dir, dataset_name + '.mat')
        tfrecord_dataset_list = []
        
        data = loadmat(raw_dataset_path)
        n_quadrotors = data['log_quad_state_real'].shape[2]
        
        for quad_idx in range(n_quadrotors):
            print(f"Preprocessing data for quad %d out of %d" % (quad_idx+1, n_quadrotors))
            
            data_dict = self.preprocess_data(data, quad_idx, separate_goals=self.separate_goals)
            n_samples = data_dict["target"].shape[0]
            
            tfrecord_dataset_name = f"%s_quad%02d.tfrecord" % (dataset_name, quad_idx)
            tfrecord_dataset_path = os.path.join(self.tfrecord_data_dir, tfrecord_dataset_name)
            
            writer = tf.io.TFRecordWriter(tfrecord_dataset_path)
            for sample in range(n_samples):
                feature = {}
                for key in self.data_key_list:
                    feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=data_dict[key][sample].flatten()))
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                serialized = example.SerializeToString()
                writer.write(serialized)
            writer.close()
            
            tfrecord_dataset_list.append(tfrecord_dataset_path)
        
        return tfrecord_dataset_list
    
    def getTFRecords(self, dataset_names):
        """
        Gets list of TFRecords for a list of datasets
        """
        all_tfrecord_dataset_list = []
        
        # For each dataset
        for dataset_name in dataset_names:
            raw_dataset_path = os.path.join(self.raw_data_dir, dataset_name + ".mat")
            tfrecord_dataset_root_path = os.path.join(self.tfrecord_data_dir, dataset_name)

            # Check if TFRecords exist with the same parameters as the ones needed
            premade_records = False
            if os.path.isfile(tfrecord_dataset_root_path + ".pkl"):
                tfrecord_params = pkl.load( open( tfrecord_dataset_root_path + ".pkl", "rb" ) )
                if tfrecord_params == self.common_params:
                    print(f"Data for dataset '%s' has already been preprocessed" % dataset_name)
                    tfrecord_dataset_list = glob(tfrecord_dataset_root_path + "*.tfrecord")
                    premade_records = True

            if not premade_records:
                print(f"Preprocessing dataset '%s'"%dataset_name)
                tfrecord_dataset_list = self.makeTFRecords(dataset_name)
                pkl.dump( self.common_params, open( tfrecord_dataset_root_path + ".pkl", "wb" ) )

            for dataset in tfrecord_dataset_list:
                all_tfrecord_dataset_list.append(dataset)
        
        return all_tfrecord_dataset_list
    
    def getKeyList(self):
        data_key_list = []
        for key in self.data_types.keys():
            if self.data_types[key] != "none":
                # data_key_list.append(key.split("_")[0]) # Split string by "_"
                data_key_list.append(key[0:-5]) # Remove "_type" from the end of the string (last five letters)
        
        return data_key_list
    
    def getNQuadrotors(self):
        dataset = os.path.join(self.raw_data_dir, self.datasets_training[0] + '.mat')
        data = loadmat(dataset)
        n_quadrotors = data['log_quad_state_real'].shape[2]
        return n_quadrotors
    
    def getScaler(self, feature_range = (-1, 1)):
        dataset = os.path.join(self.raw_data_dir, self.datasets_training[0] + '.mat')
        data = loadmat(dataset)
        data_dict = self.preprocess_data(data, 0)
        
        new_scaler = scaler()
        new_scaler.fit(data_dict)
        
        return new_scaler
                
    def scaleData(self, data):
        data_scaled = data
        for key in self.data_key_list:
            if key != "obstacle_input":
                if len(data[key].shape) == 3:
                    for timestep in range(data[key].shape[1]):
                        data_scaled[key][:, timestep, :] = scaler[key].transform( data[key][:, timestep, :] )
                else:
                    for quad_idx in data[key].shape[3]:
                        for timestep in range(data[key].shape[1]):
                            data_scaled[key][:, timestep, :, quad_idx] = scaler[key].transform( data[key][:, timestep, :, quad_idx] )
                        
        return data_scaled
    
    def unscaleData(self, data_scaled):
        data = data_scaled
        for key in self.data_key_list:
            if key != "obstacle_input":
                if len(data[key].shape) == 3:
                    for timestep in range(data[key].shape[1]):
                        data[key][:, timestep, :] = scaler[key].inverse_transform( data_scaled[key][:, timestep, :] )
                else:
                    for quad_idx in data[key].shape[3]:
                        for timestep in range(data[key].shape[1]):
                            data[key][:, timestep, :, quad_idx] = scaler[key].inverse_transform( data_scaled[key][:, timestep, :, quad_idx] )

        return data
    
    def _parse_function(self, example_proto):
        keys_to_features = {}
        for key in self.data_key_list:
            if key == "target":
                shape = (self.prediction_horizon, 3)
            elif key == "query_input":
                shape = (self.past_horizon, 3)
            elif key == "others_input":
                shape = (self.past_horizon, 6, self.n_quadrotors-1)

            keys_to_features[key] = tf.io.FixedLenFeature(shape, tf.float32)

        parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
        return parsed_features
    
    def dataset_setup(self, dataset_in, shuffle = True):
        dataset_out = dataset_in.map(self._parse_function)
        if shuffle:
            dataset_out = dataset_out.shuffle(buffer_size = int(1e7))
        dataset_out = dataset_out.map(self.scaler.transform)
        dataset_out = dataset_out.batch(self.batch_size, drop_remainder=True) # Drop remainder so that all batches have the same size
        return dataset_out


def find_last_usable_step(goal_array, logsize, n_quads): 
    zero_index = [] # Time step at which all goal states are zero (simulation stops)
    for i in range(100, goal_array.shape[1]): # The 100 is a manual fix, datasets should always be much bigger than this
        for j in range(n_quads): 
            if np.array_equiv(goal_array[:,i,j], [0, 0, 0, 0]):
                zero_index = i
                break
            else:
                zero_index = []
        if zero_index != []:
            break
    if zero_index == []:
        zero_index = goal_array.shape[1]
    final_timestep = min(zero_index-1, logsize) # Final time step
    return final_timestep

def expand_sequence(sequence_array, horizon): 
    # Sequence array has shape [features, time steps]
    # Expanded sequence will have shape [time steps, horizon, features]
    expanded_sequence = np.zeros((sequence_array.shape[1]-horizon+1,\
                                horizon,\
                                sequence_array.shape[0]))
    
    for i in range(sequence_array.shape[0]): # For each feature
        sequence = sequence_array[i, :]
        expanded_sequence[:, :, i] = hankel(sequence[0:horizon],\
                                            sequence[horizon-1:]).transpose()
    
    return expanded_sequence

def reduce_sequence(hankel_matrix):
    aux = []
    if len(hankel_matrix.shape) == 4:
        for quad_idx in range(hankel_matrix.shape[3]):
            for feature_idx in range(hankel_matrix.shape[2]):
                aux.append( np.concatenate([hankel_matrix[0, :, feature_idx, quad_idx], hankel_matrix[1:, -1, feature_idx, quad_idx]], axis = 0) )
        reduced_sequence = np.stack(aux, axis = 1)
        
    else:
        for feature_idx in range(hankel_matrix.shape[2]):
            aux.append( np.concatenate([hankel_matrix[0, :, feature_idx], hankel_matrix[1:, -1, feature_idx]], axis = 0) )
        reduced_sequence = np.stack(aux, axis = 1)
    
    return reduced_sequence

class scaler():
    def __init__(self, feature_range = (-1 ,1)):
        self.feature_range = feature_range
    
    def fit(self, data):
        self.mins = {}
        self.maxs = {}
        for key in data.keys():    
            if len(data[key].shape) == 3:
                self.mins[key] = data[key].min(axis=0).min(axis=0)
                self.maxs[key] = data[key].max(axis=0).max(axis=0)
            elif len(data[key].shape) == 4:
                self.mins[key] = data[key].min(axis=3).min(axis=0).min(axis=0)
                self.maxs[key] = data[key].max(axis=3).max(axis=0).max(axis=0)
            else:
                raise Exception("Data does not have the expected dimensions")
        
        
    def transform(self, data):
        scaled_data = {}
        for key in data.keys():
            # X_std = (data[key] - self.mins[key])/(self.maxs[key] - self.mins[key])
            # scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
            
            if tf.is_tensor(data[key]):
                # unscaled_data = data[key]#.numpy()
                
                if len(data[key].shape) == 2:
                    X_std = (data[key] - self.mins[key][np.newaxis,:])/(self.maxs[key][np.newaxis,:] - self.mins[key][np.newaxis,:])
                    scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
                elif len(data[key].shape) == 3:
                    # actual_range = self.maxs[key] - self.mins[key]
                    X_std = (data[key] - self.mins[key][np.newaxis, :, np.newaxis])/(self.maxs[key][np.newaxis, :, np.newaxis] - self.mins[key][np.newaxis, :, np.newaxis])
                    scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
                else:
                    raise Exception("Data does not have the expected dimensions")
                
            else:
                # unscaled_data = data[key]
                if len(data[key].shape) == 3:
                    X_std = (data[key] - self.mins[key][np.newaxis, np.newaxis, :])/(self.maxs[key][np.newaxis, np.newaxis, :] - self.mins[key][np.newaxis, np.newaxis, :])
                    scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
                elif len(data[key].shape) == 4:
                    # actual_range = self.maxs[key] - self.mins[key]
                    X_std = (data[key] - self.mins[key][np.newaxis, np.newaxis, :, np.newaxis])/(self.maxs[key][np.newaxis, np.newaxis, :, np.newaxis] - self.mins[key][np.newaxis, np.newaxis, :, np.newaxis])
                    scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
                else:
                    raise Exception("Data does not have the expected dimensions")
                
        return scaled_data
    
    def inverse_transform(self, data):
        unscaled_data = {}
        for key in data.keys():
            if len(data[key].shape) == 3:
                X_std = (data[key] - self.feature_range[0])/(self.feature_range[1] - self.feature_range[0])
                unscaled_data[key] = X_std * (self.maxs[key][np.newaxis, np.newaxis, :] - self.mins[key][np.newaxis, np.newaxis, :]) + self.mins[key][np.newaxis, np.newaxis, :]
            elif len(data[key].shape) == 4:
                X_std = (data[key] - self.feature_range[0])/(self.feature_range[1] - self.feature_range[0])
                unscaled_data[key] = X_std * (self.maxs[key][np.newaxis, np.newaxis, :, np.newaxis] - self.mins[key][np.newaxis, np.newaxis, :, np.newaxis]) + self.mins[key][np.newaxis, np.newaxis, :, np.newaxis]
                
        return unscaled_data
        
