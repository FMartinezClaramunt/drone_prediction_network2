import os, sys
import pickle as pkl
import numpy as np
from utils.model_utils import model_selector
from utils.data_handler_v3_tfrecord import find_last_usable_step, expand_sequence
from scipy.io import loadmat
import matplotlib.pyplot as plt
import copy

# dataset_name = "testSwap200_centralized"
# dataset_name = "randomCentralized_noObs"
# dataset_name = "randomCentralized_staticObs"
# dataset_name = "randomCentralized_dynObs"
# dataset_name = "randomCentralized_noObs4"
# dataset_name = "randomCentralized_dynObs4"
dataset_name = "centralized_20drones"

model_name = "dynamicEllipsoidObstaclesRNN_commonInputMaxPooling_alt"
model_number = 508
prediction_horizon = 20
past_horizon = 20
dt = 0.05

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

model.call(dummy_input_data)
model.load_weights(checkpoint_path)

del dummy_input_data

if model.stateful:
    for i in range(len(model.layers)):
        model.layers[i].stateful = False
        

pos_target = np.zeros((n_predictions, prediction_horizon, 3))
pos_prediction_RNN = np.zeros((n_predictions, prediction_horizon+1, 3))
pos_prediction_CV = np.zeros((n_predictions, prediction_horizon+1, 3))
pos_prediction_MPC = np.zeros((n_predictions, prediction_horizon, 3))

for query_quad_idx in range(n_robots):
    other_quad_idxs = [idx for idx in range(n_robots) if idx != query_quad_idx]
    sample_idxs = [idx for idx in range(query_quad_idx*predictions_per_robot,(query_quad_idx+1)*predictions_per_robot)]
    
    input_data["query_input"][sample_idxs] = expand_sequence(state_array[3:6, past_timesteps_idxs, query_quad_idx], past_horizon)
    
    pos_target[sample_idxs] = expand_sequence(state_array[0:3, future_timesteps_idxs, query_quad_idx], prediction_horizon)
    
    pos_prediction_RNN[sample_idxs, 0, :] = np.transpose(state_array[0:3, past_timesteps_idxs[past_horizon-1:], query_quad_idx])
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

scaled_data = scaler.transform(input_data)
scaled_data["target"] = model.predict(scaled_data)
del scaled_data["query_input"]
del scaled_data["others_input"]
if "obstacles_input" in scaled_data:
    del scaled_data["obstacles_input"]
# vel_prediction = scaler.inverse_transform(scaled_data)["target"]
vel_prediction = scaler.inverse_transform({"target": scaled_data["target"]})["target"]

del scaled_data

for step in range(1, prediction_horizon+1):
    pos_prediction_RNN[:, step, :] = pos_prediction_RNN[:, step-1, :] + dt * vel_prediction[:, step-1, :]
    pos_prediction_CV[:, step, :] = pos_prediction_CV[:, step-1, :] + dt * input_data["query_input"][:, -1, :]

pos_error_RNN = np.linalg.norm(pos_target - pos_prediction_RNN[:, 1:, :], axis = 2)
pos_error_CV = np.linalg.norm(pos_target - pos_prediction_CV[:, 1:, :], axis = 2)
pos_error_MPC = np.linalg.norm(pos_target - pos_prediction_MPC, axis = 2) # The errors are a bit too high for this case

all_means_RNN = np.mean(pos_error_RNN, axis = 0)
all_means_CV = np.mean(pos_error_CV, axis = 0)
all_means_MPC = np.mean(pos_error_MPC, axis = 0)
all_stds_RNN = np.std(pos_error_RNN, axis = 0)
all_stds_CV = np.std(pos_error_CV, axis = 0)
all_stds_MPC = np.std(pos_error_MPC, axis = 0)

# Retrieve values for 5, 10, 15 and 20
print(f"\nMeans CV:\t\t", all_means_CV[4::5], f"\nStandard deviations CV:\t", all_stds_CV[4::5])
print(f"\nMeans MPC:\t\t", all_means_MPC[4::5], f"\nStandard deviations MPC:", all_stds_MPC[4::5])
print(f"\nMeans RNN:\t\t", all_means_RNN[4::5], f"\nStandard deviations RNN:", all_stds_RNN[4::5])


"""fig, ax = plt.subplots(figsize=(8, 6))

y_CV = np.array([0] + list(all_means_CV))
delta_CV = 0.3*np.array([0] + list(all_stds_CV))
ax.plot(y_CV, label="CVM", color = 'b')
ax.fill_between(np.arange(0,21), y_CV-delta_CV, y_CV+delta_CV, color='b', alpha=0.3)

y_MPC = np.array([0] + list(all_means_MPC))
delta_MPC = 0.3*np.array([0] + list(all_stds_MPC))
ax.plot(y_MPC, label="MPC", color='r')
ax.fill_between(np.arange(0,21), y_MPC-delta_MPC, y_MPC+delta_MPC, color='r', alpha=0.3)

y_RNN = np.array([0] + list(all_means_RNN))
delta_RNN = 0.3*np.array([0] + list(all_means_RNN))
ax.plot(y_RNN, label="RNN", color='g')
ax.fill_between(np.arange(0,21), y_RNN-delta_RNN, y_RNN+delta_RNN, color='g', alpha=0.3)


plt.xlabel('Future timestep')
plt.ylabel('Displacement error [m]')
plt.xticks(np.arange(0, 21.0, 2.0))
ax.set_xlim(-0.1, 20.1)
ax.set_ylim(-0.02, 0.7)
plt.legend(loc="upper left")
plt.grid()

plt.savefig("evaluation_" + dataset_name+'.pdf', bbox_inches='tight')
plt.show()"""

data = {
    "CVM":{
        "mean": all_means_CV,
        "std": all_stds_CV,
    },
    "MPC":{
        "mean": all_means_MPC,
        "std": all_stds_MPC,
    },
    "RNN":{
        "mean": all_means_RNN,
        "std": all_stds_RNN,
    }
}
pkl.dump(data, open( "test_" + dataset_name + ".pkl", "wb" ))

print("Done")











""" To test inference time
from time import time
times = []
n_robs = 6
test_data = {}
test_data["query_input"] = scaled_data["query_input"][0:n_robs]
test_data["others_input"] = scaled_data["others_input"][0:n_robs]

for i in range(100):
    t1 = time()
    model.predict(test_data)
    times.append(time()-t1)
print(np.mean(times))
"""