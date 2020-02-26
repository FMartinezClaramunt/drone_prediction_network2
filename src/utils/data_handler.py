"""
Module to process the data coming from the .mat files.
It can deal produce a multitude of input and training data structures, depending on the network shape that wants to be used.
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.linalg import hankel
import sklearn.preprocessing as skp

def get_dataset(args):
    folder_path = args['folder_path']
    train_datasets = args['train_datasets']
    validation_datasets = args['validation_datasets']
    test_datasets = args['test_datasets']
    past_steps = args['past_steps']
    future_steps = args['future_steps']
    X_type = args['X_type'] # [pos, vel, full, vel_pos, vel_full]
    Y_type = args['Y_type'] # [pos, vel]
    split = args['split']
    
    train_data_paths = []
    for dataset in train_datasets:
        train_data_paths.append(os.path.join(folder_path, dataset + '.mat'))
        
    val_data_paths = []
    for dataset in validation_datasets:
        val_data_paths.append(os.path.join(folder_path, dataset + '.mat'))

    test_data_paths = []
    for dataset in test_datasets:
        test_data_paths.append(os.path.join(folder_path, dataset + '.mat'))

    X_train, Y_train = prepare_data(train_data_paths, past_steps, future_steps, X_type = X_type, Y_type = Y_type, split = split)
    X_val, Y_val = prepare_data(val_data_paths, past_steps, future_steps, X_type = X_type, Y_type = Y_type, split = split)
    X_test, Y_test = prepare_data(test_data_paths, past_steps, future_steps, X_type = X_type, Y_type = Y_type, split = split)

    if not split:
        # Add number of input and output features to the args dictionary, to use when constructing the model 
        args.update({"input_features": X_train.shape[2], "output_features": Y_train.shape[2]})
    else:
        # Add number of input and output features to the args dictionary, to use when constructing the model 
        args.update({"input1_features": X_train[0].shape[2], "input2_features": X_train[1].shape[2], "output_features": Y_train.shape[2]})

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, args

def get_scaler(data, feature_range = (-1, 1)):
    if type(data) == list:
        scaler = [] # List of scalers
        for i in range(len(data)):
            scaler.append(skp.MinMaxScaler(feature_range=feature_range))
            scaler[i].fit(np.reshape(data[i], [data[i].shape[0]*data[i].shape[1], data[i].shape[2]])) # Samples and horizon are stacked into the first dimension
    else:
        scaler = skp.MinMaxScaler(feature_range=feature_range)
        scaler.fit(np.reshape(data, [data.shape[0]*data.shape[1], data.shape[2]])) # Samples and horizon are stacked into the first dimension

    return scaler

def scale_data(data, scaler):

    if type(data) == list:
        data_scaled = []
        for idx in range(len(data)):
            data_scaled.append( np.zeros_like(data[idx]) )
            for i in range(data[idx].shape[1]):
                data_scaled[idx][:,i,:] = scaler[idx].transform(data[idx][:,i,:])
    else:    
        data_scaled = np.zeros_like(data)
        for i in range(data.shape[1]):
            data_scaled[:,i,:] = scaler.transform(data[:,i,:])
    
    return data_scaled

def unscale_output(Y_scaled, scaler):

    Y = np.zeros_like(Y_scaled)

    for i in range(Y_scaled.shape[1]):
        Y[:,i,:] = scaler.inverse_transform(Y_scaled[:, i, :])

    return Y

def prepare_data(data_paths, past_steps, future_steps, shuffle = True, X_type = 'full', Y_type = 'pos', split = False):
    if Y_type == 'pos' and X_type == 'vel':
        raise Exception("Invalid X_type/Y_type combination")
    
    # Define lists that will store inputs and targets
    X_list = []
    Y_list = []

    for data_path in data_paths:
        # Load mat file
        data = loadmat(data_path)

        # Extract data from datafile
        goal_array = data['log_quad_goal']
        state_array = data['log_quad_state_real']
        logsize = int(data['logsize'])
        n_quads = goal_array.shape[2]

        # Find last time step which can be used for training
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
        final_timestep = min(zero_index-1, logsize) # Final time step

        # Define data to be used as input
        if X_type == 'pos': # Positions state as input
            input_data = state_array[0:3, 0:final_timestep-future_steps, :]
            n_states_q0 = 3
        elif X_type == 'vel': # Velocities as input
            input_data = state_array[3:6, 0:final_timestep-future_steps, :]
            n_states_q0 = 3
        elif X_type == 'full' or X_type == 'vel_pos' or X_type == 'vel_full': # Full state as input
            input_data = state_array[0:6, 0:final_timestep-future_steps, :]
            n_states_q0 = 6
        else:
            raise Exception("Invalid X_type argument")
            
        # Define data to be used as target
        if split: # Target changes depending on whether we are going to split the input data or not (in the first case only on quadrotor's trajectory is predicted, in the second one the trajectory of all quadrotors is)
            idx_quad_end = 1 # Necessary so that the array doesn't lose the last dimension
        else:
            idx_quad_end = None
        
        if Y_type == 'pos':
            idx_state_init = 0
            idx_state_end = 3
        elif Y_type == 'vel':
            idx_state_init = 3
            idx_state_end = 6
        else:
            raise Exception("Invalid Y_type argument") 
        
        target_data = state_array[idx_state_init:idx_state_end, past_steps:final_timestep, 0:idx_quad_end]
        
        
        # Reshape into 2D arrays (all features ✕ all timesteps)
        input_shape = input_data.shape
        input_array2D = np.reshape(np.moveaxis(input_data,-1,0),\
                                    (input_shape[0]*input_shape[2], input_shape[1]))

        target_shape = target_data.shape
        target_array2D = np.reshape(np.moveaxis(target_data,-1,0),\
                                    (target_shape[0]*target_shape[2], target_shape[1]))

        # Placeholder input and target 3D arrays (samples ✕ horizon timesteps ✕ all features)
        train_input = np.zeros((input_array2D.shape[1]-past_steps+1,\
                                past_steps,\
                                input_array2D.shape[0]))

        train_target = np.zeros((target_array2D.shape[1]-future_steps+1,\
                                future_steps,\
                                target_array2D.shape[0]))

        # Fill in the arrays
        for i in range(train_input.shape[2]): # For all features
            sequence = input_array2D[i, :] # Feature sequence
            train_input[:, :, i] = hankel(sequence[0:past_steps],\
                                            sequence[past_steps-1:]).transpose()
        X_list.append(train_input)
            
        for i in range(train_target.shape[2]): # For all features
            sequence = target_array2D[i, :] # Feature sequence
            train_target[:, :, i] = hankel(sequence[0:future_steps],\
                                            sequence[future_steps-1:]).transpose()
        Y_list.append(train_target)

    if len(X_list) > 1: # If there is more than one dataset
        X = np.vstack(X_list)
        Y = np.vstack(Y_list)
    else:
        X = X_list[0]
        Y = Y_list[0]

    if shuffle:
        idxs = np.random.permutation(len(X)) # Indexes to shuffle arays
        X = X[idxs]
        Y = Y[idxs]

    if split: # For multi input networks
        if Y_type == 'pos': # Compute relative positions for the future trajectory of quadrotor 0
            Y = Y - X[:, -1:, 0:3]
        
        X_ = [] # X will be a list of len = 2
        input2 = [] # To store the second input
        
        if X_type == 'full':
            input1 = X[:,:,0:6]
            input1[:,:,0:3] = input1[:,:,0:3] - input1[:,-1:,0:3] # Position history relative to the last position
            X_.append( input1 )
        elif X_type == 'pos' or X_type == 'vel':
            X_.append(X[:, :, 0:n_states_q0]) # Data for the first quadrotor
        elif X_type == 'vel_pos' or X_type == 'vel_full':
            X_.append(X[:, :, 3:6]) # Data for the first quadrotor

        input2 = X[:, :, n_states_q0:] # Every quadrotor besides the first one
        if X_type == 'pos' or X_type == 'full' or X_type == 'vel_full' or X_type == 'vel_pos': # If positions are used as inputs
            for i in range(0, input2.shape[2], n_states_q0): 
                input2[:, :, i:i+3] = input2[:, :, i:i+3] - X[:, :, 0:3] # Convert to relative positions w.r.t. quadrotor 0
        
        input2 = X[:, :, n_states_q0:] # Every quadrotor besides the first one
        if X_type == 'pos' or X_type == 'full' or X_type == 'vel_pos' or X_type == 'vel_full':
            for i in range(0, input2.shape[2], n_states_q0): 
                input2[:, :, i:i+3] = input2[:, :, i:i+3] - X[:, :, 0:3] # Convert to relative positions w.r.t. quadrotor 0
            if X_type == 'vel_pos':
                idxs = [True, True, True, False, False, False]*(n_quads-1)
                input2 = input2[:, :, idxs]

        X_.append( input2 )  
            
        X = X_

    return X, Y        


