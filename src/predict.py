import os, sys
import pickle as pkl
import numpy as np
# from utils.data_handler_v3_tfrecord import expand_sequence
from utils.model_utils import model_selector

model_name = "dynamicEllipsoidObstaclesRNN_commonInputMaxPooling_alt"
model_number = 508

class trajectory_predictor():
    def __init__(self, n_robots, n_obstacles, past_horizon=10, prediction_horizon=20, dt=0.05, model_name = model_name, model_number = model_number):
        self.n_robots = n_robots
        self.n_obstacles = n_obstacles
        self.past_horizon = past_horizon
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        
        root_dir= os.path.dirname(sys.path[0])
        model_dir = os.path.join(root_dir, "trained_models", model_name, str(model_number))
        parameters_path = os.path.join(model_dir, "model_parameters.pkl")
        checkpoint_path = os.path.join(model_dir, "model_checkpoint.h5")
        
        assert os.path.isfile(checkpoint_path)    
        args = pkl.load( open( parameters_path, "rb" ) )
        args.prediction_horizon = self.prediction_horizon
        
        assert "scaler" in args
        self.scaler = args.scaler
        
        self.input_data = {}
        self.input_data["query_input"] = np.zeros((self.n_robots, self.past_horizon, 3))
        if self.n_robots > 1:
            self.input_data["others_input"] = np.zeros((self.n_robots, self.past_horizon, 6, self.n_robots-1))
        else:
            args.others_input_type = "none"
        if self.n_obstacles > 0:
            self.input_data["obstacles_input"] = np.zeros((self.n_robots, 6, self.n_obstacles))
        else:
            args.obstacles_input_type = "none"

        self.model = model_selector(args)
        self.model.call(self.input_data)
        self.model.load_weights(checkpoint_path)
        
        if self.model.stateful:
            for i in range(len(self.model.layers)):
                self.model.layers[i].stateful = False
        
    def predict(self, robot_data, obstacle_data):
        
        for query_quad_idx in range(self.n_robots):
            other_quad_idxs = [idx for idx in range(self.n_robots) if idx != query_quad_idx]
            
            self.input_data["query_input"][query_quad_idx] = np.transpose( robot_data[3:6, : , query_quad_idx] )
            
            if self.n_robots > 1:
                self.input_data["others_input"][query_quad_idx] = np.moveaxis( robot_data[0:6, : , other_quad_idxs] - robot_data[0:6, :, query_quad_idx:query_quad_idx+1], 0, 1)
            
            if self.n_obstacles > 0:
                self.input_data["obstacles_input"][query_quad_idx] = obstacle_data - robot_data[0:6, -1, query_quad_idx:query_quad_idx+1]
        
        scaled_data = self.scaler.transform(self.input_data)
        
        scaled_data["target"] = self.model.predict(scaled_data)
        vel_prediction = self.scaler.inverse_transform(scaled_data)["target"]
        
        pos_prediction = np.zeros((self.n_robots, self.prediction_horizon+1, 3))
        pos_prediction[:, 0, :] = np.transpose(robot_data[0:3, -1 , :])
        for step in range(1, self.prediction_horizon+1):
            pos_prediction[:, step, :] = pos_prediction[:, step-1, :] + self.dt * vel_prediction[:, step-1, :]
        
        return np.swapaxes(pos_prediction[:, 1:, :], 0, -1)
        
        
        
        



