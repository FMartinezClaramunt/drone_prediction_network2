"""
Module to process the data coming from the .mat files. Some functionalities have been removed from the original version to make development more agile.
The module assumes that all datasets have the same number of quadrotors

It only produces one set of input/target data shapes:
- Input data is a list with:
    - Element 0: Data array for the query agent.
    - Element 1: List of data arrays for the other agents.
- Target data is an array with the future positions/velocities for the query agent.
Furthermore, all quadrotors in each of the datasets have been considered as query agents in order to generate more data for training.

NOTE: X_types 'pos', 'vel' and 'full' have not been implemented. Only 'vel_pos' and 'vel_full' are available.
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.linalg import hankel
import sklearn.preprocessing as skp

def prepare_data(data_paths, args, shuffle = True, relative = True):
    past_steps = args['past_steps']
    future_steps = args['future_steps']
    X_type = args['X_type']
    Y_type = args['Y_type']
    
    # Define lists that will store inputs and targets
    query_agent_input = []
    other_agents_inputs = []
    targets = []

    for data_path in data_paths:
        # Load mat file
        data = loadmat(data_path)

        # Extract data from datafile
        goal_array = data['log_quad_goal'] # [goal pose (4), timesteps, quadrotors] 
        state_array = data['log_quad_state_real'] # [states (9), timesteps, quadrotors] 
        logsize = int(data['logsize'])
        n_quads = goal_array.shape[2]

        # Find last time step which can be used for training
        final_timestep = find_last_usable_step(goal_array, logsize, n_quads)

        # Define states to be used as inputs
        if X_type == 'vel_pos':
            idx_X1_state_init = 3
            idx_X1_state_end = 6
            idx_X2_state_init = 0
            idx_X2_state_end = 3
        elif X_type == 'vel_full':
            idx_X1_state_init = 3
            idx_X1_state_end = 6
            idx_X2_state_init = 0
            idx_X2_state_end = 6
        elif X_type == 'vel': 
            idx_X1_state_init = 3
            idx_X1_state_end = 6
            idx_X2_state_init = 3
            idx_X2_state_end = 6
        elif X_type == 'pos': 
            idx_X1_state_init = 0
            idx_X1_state_end = 3
            idx_X2_state_init = 0
            idx_X2_state_end = 3
        elif X_type == 'full': 
            idx_X1_state_init = 0
            idx_X1_state_end = 6
            idx_X2_state_init = 0
            idx_X2_state_end = 6
        else:
            raise Exception("Invalid X_type argument")
            
        if Y_type == 'pos':
            idx_Y_state_init = 0
            idx_Y_state_end = 3
        elif Y_type == 'vel':
            idx_Y_state_init = 3
            idx_Y_state_end = 6
        else:
            raise Exception("Invalid Y_type argument") 
        
        for query_quad_idx in range(n_quads):
            other_quad_idxs = [idx for idx in range(n_quads) if idx != query_quad_idx]
            
            # Add first element of the list of inputs, which corresponds to the query agent's data
            input1_data = state_array[idx_X1_state_init:idx_X1_state_end,\
                                      0:final_timestep - future_steps,\
                                      query_quad_idx]
            query_quad_sequence = expand_sequence(input1_data, past_steps)
            query_agent_input.append(query_quad_sequence)
            
            # Add second element to the list of inputs, which is the list of other agent's data
            input2_data = state_array[idx_X2_state_init:idx_X2_state_end,\
                                      0:final_timestep - future_steps,\
                                      :]
            
            if relative:
                query_agent_curr_pos = state_array[0:3, 0:final_timestep - future_steps, query_quad_idx:query_quad_idx+1]
                input2_data[0:3, :, :] = input2_data[0:3, :, :] - query_agent_curr_pos # Relative positions to the query agent
            
            input2 = []
            for quad_idx in other_quad_idxs:
                other_quad_sequence = expand_sequence(input2_data[:,:,quad_idx], past_steps)
                input2.append(other_quad_sequence)
            other_agents_inputs.append(input2)

            # Slice state_array to build target_data
            target_data = state_array[idx_Y_state_init:idx_Y_state_end,\
                                      past_steps:final_timestep,\
                                      query_quad_idx:query_quad_idx + 1]
            
            # Expand target feature sequences
            target = expand_sequence(target_data, future_steps)
            
            # Compute relative positions for the future trajectory of query quadrotor
            if Y_type == 'pos':
                query_agent_prev_pos = state_array[0:3,\
                                              past_steps-1:final_timestep-future_steps,\
                                              query_quad_idx].transpose() # [time steps, position coordinates]
                target = target - np.expand_dims(query_agent_prev_pos, axis = 1)
            
            # Add target to master list of targets
            targets.append(target)
    
    # Stack all the data for the query agent through the first axis (samples)
    X = [ np.vstack(query_agent_input) ]
    
    n_experiments = len(other_agents_inputs)
    n_other_agents = len(other_agents_inputs[0])
    
    X2_list = [] # To unpack the information of the other quadrotors
    for __ in range(n_other_agents): # Number of other quadrotors
        X2_list.append([])
    
    for exp_idx in range(n_experiments): # For each of the experiments
        for other_quad_idx in range(n_other_agents): # We assume that all experiments have the same number of quadrotors
            X2_list[other_quad_idx].append(other_agents_inputs[exp_idx][other_quad_idx])
        
    for other_quad_idx in range(n_other_agents):
        X.append(np.vstack(X2_list[other_quad_idx]))    
    
    Y = np.vstack(targets)
    
    if shuffle:
        idxs = np.random.permutation(Y.shape[0]) # Indexes to shuffle arays
        for i in range(len(X)):
            X[i] = X[i][idxs]
        Y = Y[idxs]

    return X, Y        

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
    for feature_idx in range(hankel_matrix.shape[2]):
        aux.append( np.concatenate([hankel_matrix[0, :, feature_idx], hankel_matrix[1:, -1, feature_idx]], axis = 0) )
    return np.stack(aux, axis = 1)

def get_scaler(data, feature_range = (-1, 1)):
    if type(data) == list: # For input data
        scaler = [] # List of scalers
        scaler.append( skp.MinMaxScaler(feature_range=feature_range) )
        scaler.append( skp.MinMaxScaler(feature_range=feature_range) )

        scaler[0].fit(reduce_sequence(data[0])) 
        # scaler[1].fit(reduce_sequence( np.vstack(data[1])))
        # TODO: Data structure fix
        scaler[1].fit(reduce_sequence( np.vstack(data[1:])))
    else: # For target data
        scaler = skp.MinMaxScaler(feature_range=feature_range)
        scaler.fit(reduce_sequence(data))
        
    return scaler

def scale_data(data, scaler):
    if type(data) == list: # For input data
        data_scaled = []
        data_scaled.append( np.zeros_like(data[0]) )
        for horizon_step in range(data[0].shape[1]):
            data_scaled[0][:, horizon_step, :] = scaler[0].transform( data[0][:, horizon_step, :] )
        
        # aux = []
        # for other_quad_idx in range(len(data[1])):
        #     aux.append( np.zeros_like(data[1][other_quad_idx]) )
        #     for horizon_step in range(data[1][other_quad_idx].shape[1]):
        #         aux[other_quad_idx][:, horizon_step, :] = scaler[1].transform( data[1][other_quad_idx][:, horizon_step, :] )
        # data_scaled.append(aux)
        
        # TODO: Data structure fix
        for other_quad in range(1, len(data)):
            data_scaled.append( np.zeros_like(data[other_quad]) )
            for horizon_step in range(data[other_quad].shape[1]):
                data_scaled[other_quad][:, horizon_step, :] = scaler[1].transform( data[other_quad][:, horizon_step, :] )
            
    else: # For target data
        data_scaled = np.zeros_like(data)
        for horizon_step in range(data.shape[1]):
            data_scaled[:, horizon_step, :] = scaler.transform(data[:, horizon_step, :])
    
    return data_scaled

def unscale_output(Y_scaled, scaler):
    Y = np.zeros_like(Y_scaled)

    for horizon_step in range(Y_scaled.shape[1]):
        Y[:, horizon_step, :] = scaler.inverse_transform(Y_scaled[:, horizon_step, :])

    return Y

def get_batch(X, Y, batch_size, index):
    X_batch = []
    
    # TODO: Data structure fix
    # X_batch.append(X[0][index:index+batch_size, :, :]) # Query agent state
    # aux = []
    # for other_quad_idx in range(len(X[1])):
    #     aux.append( X[1][other_quad_idx][index:index+batch_size, :, :] ) # Other agents' states
    #     X_batch.append( X[1][other_quad_idx][index:index+batch_size, :, :] ) # Other agents' states
    # X_batch.append(aux)
    
    for quad_idx in range(len(X)):
        X_batch.append( X[quad_idx][index:index+batch_size, :, :] ) # Other agents' states
    
    Y_batch = Y[index:index+batch_size, :, :]
    
    return X_batch, Y_batch
    