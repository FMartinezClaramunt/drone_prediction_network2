import os, sys
import pickle as pkl
import numpy as np
from utils.model_utils import model_selector
from utils.data_handler_v3_tfrecord import find_last_usable_step, expand_sequence
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import copy

# dataset_name = "testSwap200_centralized"
# dataset_name = "randomCentralized_noObs"
# dataset_name = "randomCentralized_staticObs"
# dataset_name = "randomCentralized_dynObs"
# dataset_name = "randomCentralized_noObs4"
# dataset_name = "randomCentralized_dynObs4"
# dataset_name = "centralized_20drones"
# dataset_name = "cent20_log_20210103_122943" # same as centralized_20drones
# dataset_name = "cent20_10obs_log_20210106_233859"

# datasets = ["randomCentralized_noObs", "randomCentralized_dynObs", "randomCentralized_noObs4", "randomCentralized_dynObs4", "cent20_large_log_20210107_212200", "cent20_10obs_large_log_20210107_182450"]
datasets = ["cent20_large_log_20210107_212200", "cent20_10obs_large_log_20210107_182450"]
prefix = "results20_"

iRNN_model_name = "dynamicEllipsoidObstaclesRNN_commonInputMaxPooling_alt"
iRNN_model_number = 508

sRNN_model_name = "simple_RNN"
sRNN_model_number = 0

prediction_horizon = 20
past_horizon = 10
dt = 0.05

root_dir = os.path.dirname(sys.path[0])

for dataset_name in datasets:
    ############ Load iRNN model ############
    iRNN_model_dir = os.path.join(root_dir, "trained_models", iRNN_model_name, str(iRNN_model_number))
    iRNN_parameters_path = os.path.join(iRNN_model_dir, "model_parameters.pkl")
    iRNN_checkpoint_path = os.path.join(iRNN_model_dir, "model_checkpoint.h5")

    assert os.path.isfile(iRNN_checkpoint_path)    
    args = pkl.load( open( iRNN_parameters_path, "rb" ) )
    args.prediction_horizon = prediction_horizon
    args.past_horizon = past_horizon

    assert "scaler" in args
    scaler = args.scaler

    model_iRNN = model_selector(args)

    ############ Load sRNN model ############
    sRNN_model_dir = os.path.join(root_dir, "trained_models", sRNN_model_name, str(sRNN_model_number))
    sRNN_parameters_path = os.path.join(sRNN_model_dir, "model_parameters.pkl")
    sRNN_checkpoint_path = os.path.join(sRNN_model_dir, "model_checkpoint.h5")

    assert os.path.isfile(sRNN_checkpoint_path)    
    args = pkl.load( open( sRNN_parameters_path, "rb" ) )
    args.prediction_horizon = prediction_horizon
    args.past_horizon = past_horizon

    assert "scaler" in args
    scaler = args.scaler

    model_sRNN = model_selector(args)
    
    ############ Load data ############
    print("Evaluating dataset: ", dataset_name)
    
    data_master_dir = os.path.join(root_dir, "data", "")
    raw_data_dir = os.path.join(data_master_dir, "Raw", "")
    dataset_path = os.path.join(raw_data_dir, dataset_name + ".mat")

    data = loadmat(dataset_path)

    goal_array = copy.copy(data['log_quad_goal']) # [goal pose (4), samples, quadrotors] 
    state_array = copy.copy(data['log_quad_state_real']) # [states (9), samples, quadrotors] 
    obs_state_array = copy.copy(data["log_obs_state_est"]) # [states (6), samples, obstacles] 
    mpc_plan = copy.copy(data["log_quad_path"]) # [pos(3), timesteps, horizon, samples, quadrotors]
    logsize = int(data['logsize'])

    del data

    # final_timestep = find_last_usable_step(goal_array, logsize)
    # all_close_to_goal = np.all(np.linalg.norm(goal_array[0:3] - state_array[0:3], axis = 0) < 0.2, axis = -1)
    # all_stopped = np.all(np.linalg.norm(state_array[3:6], axis = 0) < 0.2, axis = -1)
    # all_reached = all_close_to_goal & all_stopped
    # all_reached_idxs = np.where(all_reached)[0]
    # final_timestep = min(20000, all_reached_idxs[np.where(all_reached_idxs[10:] - all_reached_idxs[0:-10] == 10)[0][0]])
    final_timestep = 20000

    past_timesteps_idxs = [idx for idx in range(0, final_timestep - prediction_horizon)]
    future_timesteps_idxs = [idx for idx in range(past_horizon, final_timestep)]

    state_array = state_array[:, 0:final_timestep]
    mpc_plan = mpc_plan[:, :, 2:final_timestep+2]

    if len(goal_array.shape) == 2:
        n_robots = 1
    else:
        n_robots = goal_array.shape[2]

    del goal_array

    if obs_state_array.size == 0:
        n_obstacles = 0
        obs_state_array = 0
    else:
        n_obstacles = obs_state_array.shape[2]
        obs_state_array = obs_state_array[:, 0:final_timestep]

    predictions_per_robot = final_timestep - prediction_horizon - past_horizon + 1
    n_predictions = n_robots * predictions_per_robot

    input_data = {}
    dummy_input_data = {}
    input_data["query_input"] = np.zeros((n_predictions, past_horizon, 3))
    dummy_input_data["query_input"] = np.zeros((16, past_horizon, 3))

    if n_robots > 1:
        input_data["others_input"] = np.zeros((n_predictions, past_horizon, 6, n_robots-1))
        dummy_input_data["others_input"] = np.zeros((16, past_horizon, 6, n_robots-1))
    else:
        args.others_input_type = "none"
    if n_obstacles > 0:
        input_data["obstacles_input"] = np.zeros((n_predictions, 6, n_obstacles))
        dummy_input_data["obstacles_input"] = np.zeros((16, 6, n_obstacles))
    else:
        args.obstacles_input_type = "none"

    model_iRNN.call(dummy_input_data)
    model_iRNN.load_weights(iRNN_checkpoint_path)

    model_sRNN.call(dummy_input_data)
    model_sRNN.load_weights(sRNN_checkpoint_path)

    del dummy_input_data

    if model_iRNN.stateful:
        for i in range(len(model_iRNN.layers)):
            model_iRNN.layers[i].stateful = False
            
    if model_sRNN.stateful:
        for i in range(len(model_sRNN.layers)):
            model_sRNN.layers[i].stateful = False

    pos_target = np.zeros((n_predictions, prediction_horizon, 3))
    pos_prediction_iRNN = np.zeros((n_predictions, prediction_horizon+1, 3))
    pos_prediction_sRNN = np.zeros((n_predictions, prediction_horizon+1, 3))
    pos_prediction_CV = np.zeros((n_predictions, prediction_horizon+1, 3))
    pos_prediction_MPC = np.zeros((n_predictions, prediction_horizon, 3))

    for query_quad_idx in range(n_robots):
        other_quad_idxs = [idx for idx in range(n_robots) if idx != query_quad_idx]
        sample_idxs = [idx for idx in range(query_quad_idx*predictions_per_robot,(query_quad_idx+1)*predictions_per_robot)]
        
        input_data["query_input"][sample_idxs] = expand_sequence(state_array[3:6, past_timesteps_idxs, query_quad_idx], past_horizon)
        
        pos_target[sample_idxs] = expand_sequence(state_array[0:3, future_timesteps_idxs, query_quad_idx], prediction_horizon)
        
        pos_prediction_iRNN[sample_idxs, 0, :] = np.transpose(state_array[0:3, past_timesteps_idxs[past_horizon-1:], query_quad_idx])
        pos_prediction_sRNN[sample_idxs, 0, :] = np.transpose(state_array[0:3, past_timesteps_idxs[past_horizon-1:], query_quad_idx])
        pos_prediction_CV[sample_idxs, 0, :] = np.transpose(state_array[0:3, past_timesteps_idxs[past_horizon-1:], query_quad_idx])
        pos_prediction_MPC[sample_idxs, :, :] = np.moveaxis( mpc_plan[:, :, past_timesteps_idxs[past_horizon-1:], query_quad_idx], [0,2], [2,0])
        
        if n_robots > 1:
            for idx in range(n_robots-1):
                input_data["others_input"][sample_idxs, :, :, idx] = expand_sequence( state_array[0:6, past_timesteps_idxs, other_quad_idxs[idx]] - state_array[0:6, past_timesteps_idxs, query_quad_idx], past_horizon )
        
        if n_obstacles > 0:
            input_data["obstacles_input"][sample_idxs] = np.moveaxis( obs_state_array[:, past_timesteps_idxs[past_horizon-1:], :] - state_array[0:6, past_timesteps_idxs[past_horizon-1:], query_quad_idx:query_quad_idx+1], [0, 1, 2], [1, 0, 2])

    del state_array
    del obs_state_array
    del mpc_plan

    print("Obtaining predictions for iRNN")
    scaled_data = scaler.transform(input_data)
    scaled_data["target"] = model_iRNN.predict(scaled_data)
    vel_prediction_iRNN = scaler.inverse_transform({"target": scaled_data["target"]})["target"]

    print("Obtaining predictions for sRNN")
    scaled_data["target"] = model_sRNN.predict(scaled_data)
    vel_prediction_sRNN = scaler.inverse_transform({"target": scaled_data["target"]})["target"]

    del scaled_data, model_iRNN, model_sRNN

    for step in range(1, prediction_horizon+1):
        pos_prediction_iRNN[:, step, :] = pos_prediction_iRNN[:, step-1, :] + dt * vel_prediction_iRNN[:, step-1, :]
        pos_prediction_sRNN[:, step, :] = pos_prediction_sRNN[:, step-1, :] + dt * vel_prediction_sRNN[:, step-1, :]
        pos_prediction_CV[:, step, :] = pos_prediction_CV[:, step-1, :] + dt * input_data["query_input"][:, -1, :]

    pos_error_iRNN = np.linalg.norm(pos_target - pos_prediction_iRNN[:, 1:, :], axis = 2)
    pos_error_sRNN = np.linalg.norm(pos_target - pos_prediction_sRNN[:, 1:, :], axis = 2)
    pos_error_CV = np.linalg.norm(pos_target - pos_prediction_CV[:, 1:, :], axis = 2)
    pos_error_MPC = np.linalg.norm(pos_target - pos_prediction_MPC, axis = 2) # The errors are a bit too high for this case

    all_means_iRNN = np.mean(pos_error_iRNN, axis = 0)
    all_means_sRNN = np.mean(pos_error_sRNN, axis = 0)
    all_means_CV = np.mean(pos_error_CV, axis = 0)
    all_means_MPC = np.mean(pos_error_MPC, axis = 0)
    all_stds_iRNN = np.std(pos_error_iRNN, axis = 0)
    all_stds_sRNN = np.std(pos_error_sRNN, axis = 0)
    all_stds_CV = np.std(pos_error_CV, axis = 0)
    all_stds_MPC = np.std(pos_error_MPC, axis = 0)

    # Retrieve values for 5, 10, 15 and 20
    print(f"\nMeans CV:\t\t", all_means_CV[4::5], f"\nStandard deviations CV:\t", all_stds_CV[4::5])
    print(f"\nMeans MPC:\t\t", all_means_MPC[4::5], f"\nStandard deviations MPC:", all_stds_MPC[4::5])
    print(f"\nMeans sRNN:\t\t", all_means_sRNN[4::5], f"\nStandard deviations RNN:", all_stds_sRNN[4::5])
    print(f"\nMeans iRNN:\t\t", all_means_iRNN[4::5], f"\nStandard deviations RNN:", all_stds_iRNN[4::5])


    data = {
        "CVM":{
            "mean": all_means_CV,
            "std": all_stds_CV,
        },
        "MPC":{
            "mean": all_means_MPC,
            "std": all_stds_MPC,
        },
        "sRNN":{
            "mean": all_means_sRNN,
            "std": all_stds_sRNN,
        },
        "iRNN":{
            "mean": all_means_iRNN,
            "std": all_stds_iRNN,
        }
    }
    pkl.dump(data, open( prefix + dataset_name + ".pkl", "wb" ))
    savemat(open( prefix + dataset_name + ".mat", "wb" ), mdict=data )

print("Done")


