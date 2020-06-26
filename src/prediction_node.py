#!/usr/bin/env python
import time
import rospy
import sys
print('Python %s on %s' % (sys.version, sys.platform))
import os
import numpy as np

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
from drone_prediction_network2.srv import MultiTrajPrediction, MultiTrajPredictionResponse

# Import model
from predict import trajectory_predictor

n_robots = 4
n_obstacles = 2
prediction_network = trajectory_predictor(n_robots, n_obstacles)


def query_network(req):
	
	robot_data = req.robotTrajHistory.data
	robot_data = np.reshape(robot_data, (6, prediction_network.past_horizon, prediction_network.n_robots), order="F")
	obstacle_data = req.obsState.data
	obstacle_data = np.reshape(obstacle_data, (6, prediction_network.n_obstacles), order="F")
	predicted_trajectories = prediction_network.predict(robot_data, obstacle_data)

	resp = MultiTrajPredictionResponse()
	resp.robotTrajPrediction.data = np.reshape(predicted_trajectories, (3*prediction_network.prediction_horizon*prediction_network.n_robots, 1), order="F")
	return resp


if __name__ == '__main__':
	rospy.init_node('drone_prediction_server')

	s = rospy.Service('/drone_trajectory_prediction', MultiTrajPrediction, query_network)
	print("Ready to provide drone predictions.")
	rospy.spin()
