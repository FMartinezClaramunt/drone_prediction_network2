import sys, os, copy
import numpy as np
import pickle as pkl
from scipy.io import loadmat, savemat
from utils.data_handler_v3_tfrecord import find_last_usable_step, expand_sequence
from utils.model_utils import model_selector
from pathlib import Path

WORKSPACE_LIMITS = np.array([5, 5, 2.4])

dataset_name = "testSwap200_centralized"

model_name = "dynamicEllipsoidObstaclesRNN_commonInputMaxPooling_alt"
model_number = 508
prediction_horizon = 20
past_horizon = 20
dt = 0.05
n_frames = 2000

def main():
    ############ Load model ############
    root_dir= os.path.dirname(sys.path[0])
    model_dir = os.path.join(root_dir, "trained_models", model_name, str(model_number))
    parameters_path = os.path.join(model_dir, "model_parameters.pkl")
    checkpoint_path = os.path.join(model_dir, "model_checkpoint.h5")

    assert os.path.isfile(checkpoint_path)    
    args = pkl.load( open( parameters_path, "rb" ) )
    args.prediction_horizon = prediction_horizon
    args.past_horizon = past_horizon

    assert "scaler" in args
    scaler = args.scaler

    model = model_selector(args)

    ############ Load data ############
    data_master_dir = os.path.join(root_dir, "data", "")
    raw_data_dir = os.path.join(data_master_dir, "Raw", "")
    dataset_path = os.path.join(raw_data_dir, dataset_name + ".mat")
    recording_dir = os.path.join(model_dir, "Recordings", "")
    export_plotting_data_path = os.path.join(recording_dir, dataset_name + ".mat")

    data = loadmat(dataset_path)

    n_robots = data['log_quad_goal'].shape[2]
    n_obstacles = data['log_obs_state_est'].shape[2]
    
    dummy_input_data = {}
    dummy_input_data["query_input"] = np.zeros((16, past_horizon, 3))
    
    if n_robots > 1:
        dummy_input_data["others_input"] = np.zeros((16, past_horizon, 6, n_robots-1))
    
    if n_obstacles > 0:
        dummy_input_data["obstacles_input"] = np.zeros((16, 6, n_obstacles))

    model.call(dummy_input_data)
    model.load_weights(checkpoint_path)

    if model.stateful:
        for i in range(len(model.layers)):
            model.layers[i].stateful = False
    
    plotting_data = getPlottingData(data, model, scaler, prediction_horizon, past_horizon, n_frames)
    savePlottingData(export_plotting_data_path, plotting_data, max_samples = n_frames)


def preprocess_data(data_, prediction_horizon, past_horizon, final_timestep, query_quad_idx, relative = True):
    separate_goals = False
    separate_obstacles = False
    
    data = copy.deepcopy(data_)
    processed_data_dict = {}
    
    # Extract data from datafile
    goal_array = data['log_quad_goal'] # [goal pose (4), timesteps, quadrotors] 
    state_array = data['log_quad_state_real'] # [states (9), timesteps, quadrotors] 
    
    obstacle_array = data['log_obs_state_est'] # [states (6), timesteps, obstacles] 
    n_obstacles = obstacle_array.shape[2]
        
    logsize = int(data['logsize'])
    if len(goal_array.shape) == 2:
        goal_array = goal_array[:,:,np.newaxis]
        state_array = state_array[:,:,np.newaxis]
    
    n_quadrotors = goal_array.shape[2]
    
    # Find last time step which can be used for training    
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

    others_input_data[0:3, :, :] = others_input_data[0:3, :, :] - query_agent_curr_pos # Relative positions to the query agent    
    others_input_data[3:6, :, :] = others_input_data[3:6, :, :] - query_agent_curr_vel # Relative positions to the query agent
        
    if n_obstacles > 0:
        # Positions and speeds
        obstacles_input_data = obstacle_array[:,\
                                            past_timesteps_idxs,\
                                            :]
        obstacles_input_data[3:6,] = obstacles_input_data[3:6,] - query_agent_curr_vel # Relative positions to the query agent

    if n_quadrotors > 1:
        others_input_list = []
        other_quad_idxs = [idx for idx in range(n_quadrotors) if idx != query_quad_idx]
        for quad_idx in other_quad_idxs:
            other_quad_sequence = expand_sequence(others_input_data[:, :, quad_idx], past_horizon)
            others_input_list.append(other_quad_sequence)
            
        processed_data_dict["others_input"] = np.stack(others_input_list, axis=-1) # Stack along last dimension

    if n_obstacles > 0:            
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
    # final_timestep - (past_horizon + prediction_horizon) + 1
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

# def getPlottingData(args, trained_model, test_dataset_idx = 0, quads_to_plot = -1, params = None):
def getPlottingData(data, trained_model, scaler, prediction_horizon, past_horizon, n_frames, quads_to_plot = -1, params = None):
    """
    Converts data from the test set into data ready for plotting
    """
    
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
    final_timestep = find_last_usable_step(goal_array, min(logsize, n_frames+prediction_horizon+past_horizon-1))

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

    position_predictions = np.zeros((trajs.shape[0]-prediction_horizon-past_horizon+1,\
                                    prediction_horizon+1,\
                                    3,\
                                    n_quadrotors))

    mpc_trajectory = np.zeros((trajs.shape[0]-prediction_horizon-past_horizon+1,\
                prediction_horizon+1,\
                3,\
                n_quadrotors))
    mpc_trajectory[:, 0] = trajs[past_horizon:final_timestep-prediction_horizon+1]
    mpc_trajectory[:, 1:] = np.swapaxes(data["log_quad_path"][0:3, :, past_horizon+2:final_timestep-prediction_horizon+3, :], 0, 2) # 2:final_timestep+2
    
    cvm_trajectory = np.zeros((trajs.shape[0]-prediction_horizon-past_horizon+1,\
                prediction_horizon+1,\
                3,\
                n_quadrotors))


    if quads_to_plot == -1: # Plot all quads
        quads_to_plot = [idx for idx in range(n_quadrotors)]
    elif type(quads_to_plot) == int:
        quads_to_plot = [quads_to_plot]

    for quad_idx in quads_to_plot:
        data_dict = preprocess_data(data, prediction_horizon, past_horizon, final_timestep, quad_idx, relative=True)
        scaled_data_dict = scaler.transform(data_dict)
        scaled_velocity_predictions = trained_model.predict(scaled_data_dict)
        scaled_data_dict["target"] = scaled_velocity_predictions
        unscaled_data = scaler.inverse_transform(scaled_data_dict)
        unscaled_velocity_predictions = unscaled_data["target"]
        
        position_predictions[:, 0, :, quad_idx] = trajs[past_horizon:-prediction_horizon+1, :, quad_idx]
        for timestep in range(1, prediction_horizon+1):
            position_predictions[:, timestep, :, quad_idx] = position_predictions[:, timestep-1, :, quad_idx] \
                                                            + unscaled_velocity_predictions[:, timestep-1, :] * dt
                                                            
                                                            
            if cvm_trajectory is not None:                                                
                cvm_trajectory[:, timestep, :, quad_idx] = cvm_trajectory[:, timestep-1, :, quad_idx] \
                                                            + vels[past_horizon:-prediction_horizon+1,:,quad_idx] * dt
    
    return {
        'trajectories': trajs,
        'obstacle_trajectories': obs_trajs,
        'predictions': position_predictions,
        'mpc_trajectory': mpc_trajectory,
        'cvm_trajectory': cvm_trajectory,
        'goals': goals
    }
    
def savePlottingData(filename, data, max_samples = None):
    if max_samples == None:
        n_samples = data['predictions'].shape[0]
    else:
        n_samples = min(data['predictions'].shape[0], max_samples)
    n_quadrotors = data['trajectories'].shape[-1]
    n_obstacles = data['obstacle_trajectories'].shape[-1]
    prediction_horizon = data['predictions'].shape[1] - 1
    past_horizon = data['trajectories'].shape[0] - n_samples - prediction_horizon + 1
    
    quadrotor_size = np.array([0.3, 0.3, 0.4])
    obstacle_size = np.array([0.4, 0.4, 0.9])
    quadrotor_sizes = quadrotor_size[:, np.newaxis] * np.ones((1, n_quadrotors))
    obstacle_sizes = obstacle_size[:,np.newaxis] * np.ones((1, n_obstacles))
    
    data_to_save = {
        "quadrotor_positions": data['trajectories'][past_horizon-1:past_horizon-1+n_samples, :, :],
        "obstacle_positions": data['obstacle_trajectories'][past_horizon-1:past_horizon-1+n_samples, :, :],
        "quadrotor_past_trajectories": np.zeros((n_samples, past_horizon, 3, n_quadrotors)),
        "quadrotor_future_trajectories": np.zeros((n_samples, prediction_horizon + 1, 3, n_quadrotors)),
        "quadrotor_predicted_trajectories": data['predictions'],
        "quadrotor_mpc_trajectories": data['mpc_trajectory'],
        "quadrotor_cvm_trajectories": data['cvm_trajectory'],
        "goals": data['goals'][past_horizon-1:past_horizon-1+n_samples, :, :],
        "quadrotor_sizes": quadrotor_sizes,
        "obstacle_sizes": obstacle_sizes
    }
    
    for iteration in range(n_samples):
        current_idx = iteration + past_horizon

        future_idx = current_idx + prediction_horizon
        data_to_save["quadrotor_past_trajectories"][iteration] = data['trajectories'][iteration:current_idx, :, :]
        data_to_save["quadrotor_future_trajectories"][iteration] = data['trajectories'][current_idx-1:future_idx, :, :]

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    savemat(filename, data_to_save)
    
    
    
if __name__ == "__main__":
    main()