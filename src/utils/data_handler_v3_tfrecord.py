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
from scipy.io import loadmat, savemat
from scipy.linalg import hankel
from utils.model_utils import parse_dataset_names, parse_input_types

WORKSPACE_LIMITS = np.array([5, 5, 2.4])

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class DataHandler():
    def __init__(self, args):
        # Data parameters
        self.dt = args.dt
        self.train = args.train
        self.batch_size = args.batch_size
        self.past_horizon = args.past_horizon
        self.prediction_horizon = args.prediction_horizon
        self.test_prediction_horizon_list = []
        for prediction_horizon in args.test_prediction_horizons.split(" "):
            self.test_prediction_horizon_list.append(int(prediction_horizon))
        self.separate_goals = args.separate_goals
        self.separate_obstacles = args.separate_obstacles
        self.remove_stuck_quadrotors = args.remove_stuck_quadrotors
        self.export_plotting_data = args.export_plotting_data
        
        # For model evaluation
        self.model_name = args.model_name
        self.model_number = args.model_number

        self.data_types = {
            "query_input_type": args.query_input_type.lower(),
            "others_input_type": args.others_input_type.lower(),
            "obstacles_input_type": args.obstacles_input_type.lower(),
            "target_type": args.target_type.lower()
        }
        self.common_params = {
            "past_horizon": self.past_horizon,
            "prediction_horizon": self.prediction_horizon,
            "separate_goals": self.separate_goals,
            "separate_obstacles": self.separate_obstacles,
            "data_types": self.data_types
        }
        self.data_key_list = self.getKeyList()
        
        # Data directories
        self.raw_data_dir = args.raw_data_dir
        self.tfrecord_data_dir = args.tfrecord_data_dir
        if not os.path.isdir(self.tfrecord_data_dir):
            Path(self.tfrecord_data_dir).mkdir(parents=True, exist_ok=True)
        
        # Training datasets
        print(f"{bcolors.OKBLUE}{bcolors.BOLD}Processing the training datasets{bcolors.ENDC}")
        self.datasets_training = parse_dataset_names(args.datasets_training)
        self.tfrecords_training = self.getTFRecords(self.datasets_training, self.common_params)

        if "scaler" in args:
            self.scaler = args.scaler
        else:
            self.scaler = self.getScaler() # Get scaler based on the data for the first quadrotor of the first training dataset
            args.scaler = self.scaler
        
        self.n_quadrotors, self.n_obstacles = self.getNObjects() # Get total number of quadrotors 
        if self.n_quadrotors == 1:
            self.data_types["others_input_type"] = "none"
        
        quadrotor_size = np.array([0.3, 0.3, 0.4])
        obstacle_size = np.array([0.4, 0.4, 0.9])
        self.quadrotor_sizes = quadrotor_size[:, np.newaxis] * np.ones((1, self.n_quadrotors))
        self.obstacle_sizes = obstacle_size[:,np.newaxis] * np.ones((1, self.n_obstacles))

        # Get dataset object for the training data
        self.tfdataset_training = tf.data.TFRecordDataset(self.tfrecords_training).apply(self.dataset_setup)
        
        # Validation datasets
        if args.datasets_validation.lower() == "none":
            print(f"{bcolors.FAIL}No validation datasets have been specified{bcolors.ENDC}")
            self.datasets_validation = []
            self.tfdataset_validation = None
        else:
            print(f"{bcolors.OKBLUE}{bcolors.BOLD}Processing the validation datasets{bcolors.ENDC}")
            self.datasets_validation = parse_dataset_names(args.datasets_validation)
            self.tfrecords_validation = self.getTFRecords(self.datasets_validation, self.common_params)
            self.tfdataset_validation = tf.data.TFRecordDataset(self.tfrecords_validation).apply(self.dataset_setup)
        
        # Test datasets
        if args.datasets_testing.lower() == "none":
            print(f"{bcolors.FAIL}No testing datasets have been specified{bcolors.ENDC}")
            self.datasets_testing = []
            self.tfdataset_testing= None
            self.tfdataset_fde_testing = None
        else:
            print(f"{bcolors.OKBLUE}{bcolors.BOLD}Processing the testing datasets{bcolors.ENDC}")
            self.datasets_testing = parse_dataset_names(args.datasets_testing)
            self.tfrecords_testing = self.getTFRecords(self.datasets_testing, self.common_params)
            self.tfdataset_testing = tf.data.TFRecordDataset(self.tfrecords_testing).apply(self.dataset_setup)
            
            if len(self.test_prediction_horizon_list) == 4:
                print(f"{bcolors.OKBLUE}{bcolors.BOLD}Processing the datasets to test FDEs for different prediction horizons{bcolors.ENDC}")
                params = copy.deepcopy(self.common_params)
                params["prediction_horizon"] = self.test_prediction_horizon_list[-1]
                self.tfrecords_fde_testing = self.getTFRecords(self.datasets_testing, params)

                self.tfdataset_fde_testing = tf.data.TFRecordDataset(self.tfrecords_fde_testing).apply(lambda x: self.dataset_setup(x, params = params))
            else:
                self.datasets_fde_testing = []
                self.tfdataset_fde_testing = None

                if len(self.test_prediction_horizon_list) > 1:
                    print(f"{bcolors.FAIL}There should be 4 different prediction horizons to test{bcolors.ENDC}")
                else:
                    print(f"{bcolors.FAIL}[WARNING] No FDE testing will be performed{bcolors.ENDC}")
                    
            # self.tfdataset_testing = tf.data.TFRecordDataset(self.tfrecords_testing).apply(lambda x: self.dataset_setup(x, shuffle=False))
    
    def getSampleInputBatch(self, dataset_type = "train"): 
        """
        Get sample input batch so that it can be used when calling the Keras model before loading the weights, either for testing or for a warm start
        """
        if "train" in dataset_type:
            sample_batch = next(iter(self.tfdataset_training))
        elif "val" in dataset_type:
            sample_batch = next(iter(self.tfdataset_validation))
        else:
            sample_batch = next(iter(self.tfdataset_testing))

        return sample_batch
        
    def getPlottingData(self, trained_model, test_dataset_idx = 0, quads_to_plot = -1, params = None):
        """
        Converts data from the test set into data ready for plotting
        """
        
        if params == None:
            params = copy.deepcopy(self.common_params)
            params["separate_goals"] = False
            params["separate_obstacles"] = False
            
        prediction_horizon = params["prediction_horizon"]
        
        raw_dataset_path = os.path.join(self.raw_data_dir, self.datasets_testing[test_dataset_idx] + '.mat')
        data = loadmat(raw_dataset_path)
        
        goal_array = data['log_quad_goal'] # [goal pose (4), timesteps, quadrotors] 
        state_array = data['log_quad_state_real'] # [states (9), timesteps, quadrotors] 
        logsize = int(data['logsize'])
        
        if len(goal_array.shape) == 3:
            n_quadrotors = goal_array.shape[2]
        else:
            n_quadrotors = 1
            goal_array = goal_array[:, :, np.newaxis]
            state_array = state_array[:, :, np.newaxis]

        # Find last time step which can be used for training
        final_timestep = find_last_usable_step(goal_array, logsize)

        trajs = np.swapaxes(state_array[0:3, 0:final_timestep, :], 0, 1)
        vels = np.swapaxes(state_array[3:6, 0:final_timestep, :], 0, 1)
        goals = np.swapaxes(goal_array[0:3, 0:final_timestep, :], 0, 1)

        if 'log_obs_state_est' in data.keys():
            obstacle_array = data['log_obs_state_est'] # [states (6), timesteps, obstacles] 
            n_obstacles = obstacle_array.shape[2]
            obs_trajs = np.swapaxes(obstacle_array[0:3, 0:final_timestep, :], 0, 1)
        else:
            n_obstacles = 0
            obs_trajs = None

        position_predictions = np.zeros((trajs.shape[0]-prediction_horizon-self.past_horizon+1,\
                                        prediction_horizon+1,\
                                        3,\
                                        n_quadrotors))

        if self.export_plotting_data and "log_quad_path" in data:
            mpc_trajectory = np.zeros((trajs.shape[0]-prediction_horizon-self.past_horizon+1,\
                        prediction_horizon+1,\
                        3,\
                        n_quadrotors))
            mpc_trajectory[:, 1:] = np.swapaxes(data["log_quad_path"][0:3, :, self.past_horizon:final_timestep, :], 0, 2) # 2:final_timestep+2
            
            cvm_trajectory = np.zeros((trajs.shape[0]-prediction_horizon-self.past_horizon+1,\
                        prediction_horizon+1,\
                        3,\
                        n_quadrotors))
        else:
            mpc_trajectory = None
            cvm_trajectory = None

        if quads_to_plot == -1: # Plot all quads
            quads_to_plot = [idx for idx in range(n_quadrotors)]
        elif type(quads_to_plot) == int:
            quads_to_plot = [quads_to_plot]

        for quad_idx in quads_to_plot:
            data_dict = self.preprocess_data(data, quad_idx, params=params, relative=True, remove_stuck_quadrotors = False)
            scaled_data_dict = self.scaler.transform(data_dict)
            scaled_velocity_predictions = trained_model.predict(scaled_data_dict)
            scaled_data_dict["target"] = scaled_velocity_predictions
            unscaled_data = self.scaler.inverse_transform(scaled_data_dict)
            unscaled_velocity_predictions = unscaled_data["target"]
            
            position_predictions[:, 0, :, quad_idx] = trajs[self.past_horizon:-prediction_horizon+1, :, quad_idx]
            for timestep in range(1, prediction_horizon+1):
                position_predictions[:, timestep, :, quad_idx] = position_predictions[:, timestep-1, :, quad_idx] \
                                                                + unscaled_velocity_predictions[:, timestep-1, :] * self.dt
                                                                
                                                                
                if cvm_trajectory is not None:                                                
                    cvm_trajectory[:, timestep, :, quad_idx] = cvm_trajectory[:, timestep-1, :, quad_idx] \
                                                                + vels[self.past_horizon:-prediction_horizon+1,:,:] * self.dt
        
        return {
            'trajectories': trajs,
            'obstacle_trajectories': obs_trajs,
            'predictions': position_predictions,
            'mpc_trajectory': mpc_trajectory,
            'cvm_trajectory': cvm_trajectory,
            'goals': goals
        }
        
    def savePlottingData(self, filename, data, max_samples = None):
        if max_samples == None:
            n_samples = data['predictions'].shape[0]
        else:
            n_samples = min(data['predictions'].shape[0], max_samples)
        n_quadrotors = data['trajectories'].shape[-1]
        n_obstacles = data['obstacle_trajectories'].shape[-1]
        prediction_horizon = data['predictions'].shape[1] - 1
        past_horizon = data['trajectories'].shape[0] - n_samples - prediction_horizon + 1
        
        data_to_save = {
            "quadrotor_positions": data['trajectories'][past_horizon-1:past_horizon-1+n_samples, :, :],
            "obstacle_positions": data['obstacle_trajectories'][past_horizon-1:past_horizon-1+n_samples, :, :],
            "quadrotor_past_trajectories": np.zeros((n_samples, past_horizon, 3, n_quadrotors)),
            "quadrotor_future_trajectories": np.zeros((n_samples, prediction_horizon + 1, 3, n_quadrotors)),
            "quadrotor_predicted_trajectories": data['predictions'],
            "quadrotor_mpc_trajectories": data['mpc_trajectory'],
            "quadrotor_cvm_trajectories": data['cvm_trajectory'],
            "goals": data['goals'][past_horizon-1:past_horizon-1+n_samples, :, :],
            "quadrotor_sizes": self.quadrotor_sizes,
            "obstacle_sizes": self.obstacle_sizes
        }
        
        for iteration in range(n_samples):
            current_idx = iteration + past_horizon
            
            future_idx = current_idx + prediction_horizon
            data_to_save["quadrotor_past_trajectories"][iteration] = data['trajectories'][iteration:current_idx, :, :]
            data_to_save["quadrotor_future_trajectories"][iteration] = data['trajectories'][current_idx-1:future_idx, :, :]
            
        savemat(filename, data_to_save)
    
    def preprocess_data(self, data_, query_quad_idx, relative = True, params=None, remove_stuck_quadrotors=None):
        if params is None:
            params = self.common_params
        
        if remove_stuck_quadrotors is None:
            remove_stuck_quadrotors = self.remove_stuck_quadrotors
        
        past_horizon = params["past_horizon"]
        prediction_horizon = params["prediction_horizon"]
        separate_goals = params["separate_goals"]
        separate_obstacles = params["separate_obstacles"]
        
        data = copy.deepcopy(data_)
        processed_data_dict = {}
        
        # Extract data from datafile
        goal_array = data['log_quad_goal'] # [goal pose (4), timesteps, quadrotors] 
        state_array = data['log_quad_state_real'] # [states (9), timesteps, quadrotors] 
        
        if 'log_obs_state_est' in data.keys() and self.data_types['obstacles_input_type'] != "none":
            obstacle_array = data['log_obs_state_est'] # [states (6), timesteps, obstacles] 
            n_obstacles = obstacle_array.shape[2]
        else:
            n_obstacles = 0
        
        if 'log_obs_size' in data.keys() and self.data_types['obstacles_input_type'] != "none":
            obs_size_array = data['log_obs_size']
        else:
            obs_size_array = None
        
        logsize = int(data['logsize'])
        if len(goal_array.shape) == 2:
            goal_array = goal_array[:,:,np.newaxis]
            state_array = state_array[:,:,np.newaxis]
        
        n_quadrotors = goal_array.shape[2]
        
        # Find last time step which can be used for training
        final_timestep = find_last_usable_step(goal_array, logsize)
        
        if remove_stuck_quadrotors: 
            stuck_quadrotor_step = np.argmax(np.any(np.abs(state_array[0:3, :, :]) > 1.2*WORKSPACE_LIMITS[:,np.newaxis,np.newaxis], axis = (0,2)))
            if stuck_quadrotor_step > 0:
                final_timestep = min(final_timestep, stuck_quadrotor_step)
        
        assert final_timestep > prediction_horizon + past_horizon - 1
        
        past_timesteps_idxs = [idx for idx in range(0, final_timestep - prediction_horizon)]
        future_timesteps_idxs = [idx for idx in range(past_horizon, final_timestep)]
        
        # Add first element of the list of inputs, which corresponds to the query agent's data
        query_input_data = state_array[3:6,\
                                    past_timesteps_idxs,\
                                    query_quad_idx]
        processed_data_dict["query_input"] = expand_sequence(query_input_data, past_horizon)
        
        # Add second element to the list of inputs, which is the list of other agent's data
        others_input_data = state_array[0:6,\
                                        past_timesteps_idxs,\
                                        :]

        # Store query agent's current position and velocity (they may also be used for the obstacle input)
        query_agent_curr_pos = state_array[0:3, past_timesteps_idxs, query_quad_idx:query_quad_idx+1]
        query_agent_curr_vel = state_array[3:6, past_timesteps_idxs, query_quad_idx:query_quad_idx+1]

        if relative and "pos" in self.data_types['others_input_type']: # Relative postition for other agents
            others_input_data[0:3, :, :] = others_input_data[0:3, :, :] - query_agent_curr_pos # Relative positions to the query agent    

        if "relvel" in self.data_types['others_input_type']: # Relative velocity for other agents
            others_input_data[3:6, :, :] = others_input_data[3:6, :, :] - query_agent_curr_vel # Relative positions to the query agent
            
        if n_obstacles > 0:
            if self.data_types['obstacles_input_type'] == "static": # Static obstacles
                # Only positions
                obstacles_input_data = obstacle_array[0:3,\
                                                    past_timesteps_idxs,\
                                                    :]
            elif "dynamic" in self.data_types['obstacles_input_type']: # Dynamic obstacles
                # Positions and speeds
                obstacles_input_data = obstacle_array[:,\
                                                    past_timesteps_idxs,\
                                                    :]
            else:
                raise Exception(f"{bcolors.FAIL}Invalid obstacles input type{bcolors.ENDC}")
            
            obstacles_input_data[0:3, :, :] = obstacles_input_data[0:3, :, :] - query_agent_curr_pos # Relative positions to the query agent
            
            if "dynamic" in self.data_types['obstacles_input_type'] and "relvel" in self.data_types["obstacles_input_type"]:
                obstacles_input_data[3:6,] = obstacles_input_data[3:6,] - query_agent_curr_vel # Relative positions to the query agent
            
            
            if obs_size_array is not None:
                obs_size_data = obs_size_array[:, past_timesteps_idxs, :]

                if "radii" in self.data_types['obstacles_input_type']:
                    obstacles_input_data = np.concatenate([obstacles_input_data,\
                                                            obs_size_data], axis = 0)
                    
                elif "points" in self.data_types['obstacles_input_type']:
                    # Increase in positions along all 3D axes in both directions
                    deltas = np.zeros((obstacles_input_data.shape[0], obstacles_input_data.shape[1], obstacles_input_data.shape[2], 6))
                    for i in range(6):
                        deltas[int(i/2),:,:,i] = obs_size_data[int(i/2),:,:] * (-1)**i
                    
                    # if "points6" in self.data_types['obstacles_input_type']:
                    #     # Stack array six times along axis = 2
                    #     new_obstacles_input_data = np.tile(obstacles_input_data, [1, 1, 6])
                        
                    #     for i in range(6):
                    #         new_obstacles_input_data[:, :, n_obstacles*i:n_obstacles*(i+1)] += deltas[:,:,:,i]
                    
                    # elif "points3" in self.data_types['obstacles_input_type']:
                    #     # Stack array three times along axis = 2
                    #     new_obstacles_input_data = np.tile(obstacles_input_data, [1, 1, 3])
                        
                    # else:
                    #     raise Exception("Unimplemented obstacles input type")
                    
                    obstacles_input_data_6points = np.tile(obstacles_input_data[:,:,:,np.newaxis], [1, 1, 1, 6]) + deltas
                    
                    if  "points6stack" in self.data_types['obstacles_input_type']:
                        # obstacles_input_data = np.swapaxes(obstacles_input_data_6points, 0, 1)
                        obstacles_input_data = obstacles_input_data_6points
                    
                    elif "points6concat" in self.data_types['obstacles_input_type']:
                        obstacles_input_data = np.reshape(obstacles_input_data_6points, [obstacles_input_data_6points.shape[0],\
                                                                                        obstacles_input_data_6points.shape[1],\
                                                                                        obstacles_input_data_6points.shape[2]*obstacles_input_data_6points.shape[3]])
                                        
                    # elif "points3" in self.data_types['obstacles_input_type']:
                    #     obstacles_input_data[0:3,] > 0
                    else:
                        raise Exception(f"{bcolors.FAIL}Invalid obstacles input type{bcolors.ENDC}")
                        

        if self.data_types["others_input_type"] != "none":
            others_input_list = []
            other_quad_idxs = [idx for idx in range(n_quadrotors) if idx != query_quad_idx]
            for quad_idx in other_quad_idxs:
                other_quad_sequence = expand_sequence(others_input_data[:, :, quad_idx], past_horizon)
                others_input_list.append(other_quad_sequence)
                
            processed_data_dict["others_input"] = np.stack(others_input_list, axis=-1) # Stack along last dimension

        if n_obstacles > 0:            
            if "stack" in  self.data_types["obstacles_input_type"]:
                processed_data_dict["obstacles_input"] = np.zeros((obstacles_input_data.shape[1]-past_horizon+1,\
                                            past_horizon,\
                                            obstacles_input_data.shape[0],\
                                            obstacles_input_data.shape[2],\
                                            obstacles_input_data.shape[3]))

                for obs_idx in range(obstacles_input_data.shape[2]):
                    for point_idx in range(obstacles_input_data.shape[3]):
                        processed_data_dict["obstacles_input"][:,:,:,obs_idx,point_idx] = expand_sequence(obstacles_input_data[:, :, obs_idx,point_idx], past_horizon)

                # processed_data_dict["obstacles_input"] = obstacles_input
                processed_data_dict["obstacles_input"] = processed_data_dict["obstacles_input"][:, -1, :, :, :] # Only use position at current step

            else:
                processed_data_dict["obstacles_input"] = np.zeros((obstacles_input_data.shape[1]-past_horizon+1,\
                                            past_horizon,\
                                            obstacles_input_data.shape[0],\
                                            obstacles_input_data.shape[2]))
                
                for obs_idx in range(obstacles_input_data.shape[2]):
                    processed_data_dict["obstacles_input"][:,:,:,obs_idx] = expand_sequence(obstacles_input_data[:, :, obs_idx], past_horizon)
                
                # processed_data_dict["obstacles_input"] = obstacles_input
                processed_data_dict["obstacles_input"] = processed_data_dict["obstacles_input"][:, -1, :, :] # Only use position at current step

        # Slice state_array to build target_data
        target_data = state_array[3:6,\
                                future_timesteps_idxs,\
                                query_quad_idx:query_quad_idx + 1]

        # Expand target feature sequences
        processed_data_dict["target"] = expand_sequence(target_data, prediction_horizon)
        
        keep_idxs = [True] * processed_data_dict["target"].shape[0]
        final_timestep - (past_horizon + prediction_horizon) + 1
        if separate_goals: # Separate trajectories by goals
            goal_data = goal_array[0:3,\
                                    0:final_timestep,\
                                    query_quad_idx]
            goal_data_hankel = expand_sequence(goal_data, past_horizon + prediction_horizon)
            
            # Compare all goals to the first one in the sequence and get the indexes of the dataset samples in which it is true for all elements of the sequence
            # same_goals_idxs = np.nonzero(np.all(goal_data_hankel == goal_data_hankel[:, 0:1, :], axis=(1,2)))[0]
            same_goals_idxs = np.all(goal_data_hankel == goal_data_hankel[:, 0:1, :], axis=(1,2)) 
            keep_idxs = np.logical_and(keep_idxs, same_goals_idxs)

        # TODO: Check if separate_obstacles works as expected
        if separate_obstacles and n_obstacles > 0: # Remove data samples with obstacles that have been reset in position (discontinuous velocity)
            absolute_obstacle_velocity_hankel = expand_sequence(obstacle_array[3:6, past_timesteps_idxs, :], past_horizon + prediction_horizon)
            continuous_obstacles_idxs = np.all(absolute_obstacle_velocity_hankel == absolute_obstacle_velocity_hankel[:, 0:1, :], axis=(1,2))
            keep_idxs = np.logical_and(keep_idxs, continuous_obstacles_idxs)
            
        for key in processed_data_dict.keys():
            processed_data_dict[key] = processed_data_dict[key][keep_idxs,]
                
        return processed_data_dict
    
    def makeTFRecords(self, dataset_name, root_name, params):
        """
        Makes TFRecords out of a raw dataset 
        """
        
        raw_dataset_path = os.path.join(self.raw_data_dir, dataset_name + '.mat')
        tfrecord_dataset_list = []
        
        data = loadmat(raw_dataset_path)
        if len(data['log_quad_state_real'].shape) == 2:
            data['log_quad_state_real'] = data['log_quad_state_real'][:,:,np.newaxis]
            data['log_quad_goal'] = data['log_quad_goal'][:,:,np.newaxis]
        
        n_quadrotors = data['log_quad_state_real'].shape[2]
        
        for quad_idx in range(n_quadrotors):
            print(f"Preprocessing data for quad %d out of %d" % (quad_idx+1, n_quadrotors))
            
            # data_dict = self.preprocess_data(data, quad_idx, separate_goals=params["separate_goals"], separate_obstacles=params["separate_obstacles"])
            data_dict = self.preprocess_data(data, quad_idx, params=params)
            n_samples = data_dict["target"].shape[0]
            
            tfrecord_dataset_name = f"%s_quad%02d.tfrecord" % (root_name, quad_idx) 
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
    
    def getTFRecords(self, dataset_names, params):
        """
        Gets list of TFRecords for a list of datasets
        """
        all_tfrecord_dataset_list = []

        # For each dataset
        for dataset_name in dataset_names:
            raw_dataset_path = os.path.join(self.raw_data_dir, dataset_name + ".mat")
            tfrecord_dataset_root_path = os.path.join(self.tfrecord_data_dir, dataset_name)

            # Check if TFRecords exist with the same parameters as the ones needed
            params_ID = -1
            premade_records = False
            for params_ID, params_file in enumerate(sorted(glob(tfrecord_dataset_root_path + "*.pkl"))):
                tfrecord_params = pkl.load( open( params_file, "rb" ) )
                if tfrecord_params == params:
                    print(f"{bcolors.OKGREEN}Data for dataset '%s' has already been preprocessed (ID = %04d){bcolors.ENDC}" % (dataset_name, params_ID))
                    premade_records = True
                    # params_ID = idx
                    tfrecord_dataset_root_path_with_index = tfrecord_dataset_root_path + f"_ID%04d" % params_ID
                    tfrecord_dataset_list = glob(tfrecord_dataset_root_path_with_index + "*.tfrecord")
                    break

            if not premade_records:
                params_ID += 1

                print(f"{bcolors.WARNING}Preprocessing dataset '%s' with ID %04d{bcolors.ENDC}" % (dataset_name, params_ID))

                tfrecord_dataset_root_path_with_index = tfrecord_dataset_root_path + f"_ID%04d" % params_ID
                tfrecord_dataset_list = self.makeTFRecords(dataset_name, tfrecord_dataset_root_path_with_index, params)

                pkl.dump( params, open( tfrecord_dataset_root_path_with_index + ".pkl", "wb" ) )

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
    
    def getNObjects(self):
        dataset = os.path.join(self.raw_data_dir, self.datasets_training[0] + '.mat')
        data = loadmat(dataset)
        if len(data['log_quad_state_real'].shape) == 2:
            n_quadrotors = 1
        else:
            n_quadrotors = data['log_quad_state_real'].shape[2]
            
        if 'log_obs_state_est' in data.keys():
            n_obstacles = data['log_obs_state_est'].shape[2]
        else:
            n_obstacles = 0
            
        return n_quadrotors, n_obstacles
    
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
            if len(data[key].shape) == 3:
                for timestep in range(data[key].shape[1]):
                    data[key][:, timestep, :] = scaler[key].inverse_transform( data_scaled[key][:, timestep, :] )
            else:
                for quad_idx in data[key].shape[3]:
                    for timestep in range(data[key].shape[1]):
                        data[key][:, timestep, :, quad_idx] = scaler[key].inverse_transform( data_scaled[key][:, timestep, :, quad_idx] )

        return data
    
    def _parse_function(self, example_proto, params):
        keys_to_features = {}
        for key in self.data_key_list:
            if key == "target":
                shape = (params["prediction_horizon"], 3)
            elif key == "query_input":
                shape = (params["past_horizon"], 3)
            elif key == "others_input":
                shape = (params["past_horizon"], 6, self.n_quadrotors-1)
            elif key == "obstacles_input":
                # shape = (self.past_horizon, 6, self.n_obstacles)
                if self.data_types['obstacles_input_type'] == "static":
                    shape = (3, self.n_obstacles)
                elif self.data_types['obstacles_input_type'] == "dynamic"\
                    or self.data_types['obstacles_input_type'] == "dynamic_relvel":
                    shape = (6, self.n_obstacles)
                elif "dynamic_radii" in self.data_types['obstacles_input_type']:
                    shape = (9, self.n_obstacles)
                elif "dynamic_points6concat" in self.data_types['obstacles_input_type']:
                    shape = (6, self.n_obstacles*6)
                elif "dynamic_points6stack" in self.data_types['obstacles_input_type']:
                    shape = (6, self.n_obstacles, 6)
                else:
                    raise Exception(f"{bcolors.FAIL}Invalid obstacles_input parameter{bcolors.FAIL}")
            keys_to_features[key] = tf.io.FixedLenFeature(shape, tf.float32)

        parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
        return parsed_features
    
    def dataset_setup(self, dataset_in, params = None, shuffle = True):
        if params is None:
            params = self.common_params
            
        dataset_out = dataset_in.map(lambda x: self._parse_function(x, params=params))
        if shuffle:
            dataset_out = dataset_out.shuffle(buffer_size = int(1e6))
        dataset_out = dataset_out.map(self.scaler.transform)
        dataset_out = dataset_out.batch(self.batch_size, drop_remainder=True) # Drop remainder so that all batches have the same size
        return dataset_out


def find_last_usable_step(goal_array, logsize, intial_ignored_steps = 100): 
    if len(goal_array) == 3:
        zero_index = np.argmax(np.all(goal_array[:, intial_ignored_steps:, :] == 0, axis=(0,2)))
    else:
        zero_index = np.argmax(np.all(goal_array[:, intial_ignored_steps:] == 0, axis=0))
    
    if zero_index == 0:
        zero_index = float('inf')

    zero_index += intial_ignored_steps - 1

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
            if key == "target" or key == "query_input":
                self.mins[key] = data[key].min(axis=0).min(axis=0)
                self.maxs[key] = data[key].max(axis=0).max(axis=0)
            elif key == "obstacles_input":
                self.mins[key] = data[key].min(axis=2).min(axis=0)
                self.maxs[key] = data[key].max(axis=2).max(axis=0)
            elif key == "others_input":
                self.mins[key] = data[key].min(axis=3).min(axis=0).min(axis=0)
                self.maxs[key] = data[key].max(axis=3).max(axis=0).max(axis=0)
            else:
                raise Exception(f"{bcolors.FAIL}Data does not have the expected dimensions{bcolors.ENDC}")
            
            
    def transform(self, data):
        scaled_data = {}
        for key in data.keys():
            if tf.is_tensor(data[key]):
                if key == "target" or key == "query_input":
                    X_std = (data[key] - self.mins[key][np.newaxis,:])/(self.maxs[key][np.newaxis,:] - self.mins[key][np.newaxis,:])
                    scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
                elif key == "obstacles_input":
                    X_std = (data[key] - self.mins[key][:, np.newaxis])/(self.maxs[key][:, np.newaxis] - self.mins[key][:, np.newaxis])
                    scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
                elif key == "others_input":
                    X_std = (data[key] - self.mins[key][np.newaxis, :, np.newaxis])/(self.maxs[key][np.newaxis, :, np.newaxis] - self.mins[key][np.newaxis, :, np.newaxis])
                    scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
                else:
                    raise Exception(f"{bcolors.FAIL}Data does not have the expected dimensions{bcolors.ENDC}")
                
            else:
                if key == "target" or key == "query_input":
                    X_std = (data[key] - self.mins[key][np.newaxis, np.newaxis, :])/(self.maxs[key][np.newaxis, np.newaxis, :] - self.mins[key][np.newaxis, np.newaxis, :])
                    scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
                elif key == "obstacles_input":
                    X_std = (data[key] - self.mins[key][np.newaxis, :, np.newaxis])/(self.maxs[key][np.newaxis, :, np.newaxis] - self.mins[key][np.newaxis, :, np.newaxis])
                    scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
                elif key == "others_input":
                    X_std = (data[key] - self.mins[key][np.newaxis, np.newaxis, :, np.newaxis])/(self.maxs[key][np.newaxis, np.newaxis, :, np.newaxis] - self.mins[key][np.newaxis, np.newaxis, :, np.newaxis])
                    scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
                else:
                    raise Exception(f"{bcolors.FAIL}Data does not have the expected dimensions{bcolors.ENDC}")

        return scaled_data

    def inverse_transform(self, data):
        unscaled_data = {}
        for key in data.keys():
            if key == "target" or key == "query_input":
                X_std = (data[key] - self.feature_range[0])/(self.feature_range[1] - self.feature_range[0])
                unscaled_data[key] = X_std * (self.maxs[key][np.newaxis, np.newaxis, :] - self.mins[key][np.newaxis, np.newaxis, :]) + self.mins[key][np.newaxis, np.newaxis, :]
            elif key == "obstacles_input":
                X_std = (data[key] - self.feature_range[0])/(self.feature_range[1] - self.feature_range[0])
                unscaled_data[key] = X_std * (self.maxs[key][np.newaxis, :, np.newaxis] - self.mins[key][np.newaxis, :, np.newaxis]) + self.mins[key][np.newaxis, :, np.newaxis]
            elif key == "others_input":
                X_std = (data[key] - self.feature_range[0])/(self.feature_range[1] - self.feature_range[0])
                unscaled_data[key] = X_std * (self.maxs[key][np.newaxis, np.newaxis, :, np.newaxis] - self.mins[key][np.newaxis, np.newaxis, :, np.newaxis]) + self.mins[key][np.newaxis, np.newaxis, :, np.newaxis]
            else:
                raise Exception(f"{bcolors.FAIL}Data does not have the expected dimensions{bcolors.ENDC}")
            
        return unscaled_data
    
    
    """def fit(self, data):
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
            raise Exception(f"{bcolors.FAIL}Data does not have the expected dimensions{bcolors.ENDC}")
    
    def transform(self, data):
        scaled_data = {}
        for key in data.keys():
            # X_std = (data[key] - self.mins[key])/(self.maxs[key] - self.mins[key])
            # scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
            
            if tf.is_tensor(data[key]):
                if len(data[key].shape) == 2:
                    X_std = (data[key] - self.mins[key][np.newaxis,:])/(self.maxs[key][np.newaxis,:] - self.mins[key][np.newaxis,:])
                    scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
                elif len(data[key].shape) == 3:
                    # actual_range = self.maxs[key] - self.mins[key]
                    X_std = (data[key] - self.mins[key][np.newaxis, :, np.newaxis])/(self.maxs[key][np.newaxis, :, np.newaxis] - self.mins[key][np.newaxis, :, np.newaxis])
                    scaled_data[key] = X_std * (self.feature_range[1]-self.feature_range[0]) + self.feature_range[0]
                else:
                    raise Exception(f"{bcolors.FAIL}Data does not have the expected dimensions{bcolors.ENDC}")
                
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
                    raise Exception(f"{bcolors.FAIL}Data does not have the expected dimensions{bcolors.ENDC}")
                
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
                
        return unscaled_data"""
        

def obstacles_data_to_input(obstacles_input_data, past_horizon):
    obstacles_input_list = []
    for obs_idx in range(obstacles_input_data.shape[2]):
        obstacle_sequence = expand_sequence(obstacles_input_data[:, :, obs_idx], past_horizon)
        obstacles_input_list.append(obstacle_sequence)
    return np.stack(obstacles_input_list, axis=-1) # Stack along last dimension
