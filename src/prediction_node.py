#!/usr/bin/env python
import time
import rospy
import sys;
print('Python %s on %s' % (sys.version, sys.platform))
import os

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
from drone_prediction_network2.srv import MultiTrajPrediction, MultiTrajPredictionResponse

# Import model
from src.predict import trajectory_predictor

n_robots = 1
n_obstacles = 0
prediction_network = trajectory_predictor(n_robots, n_obstacles)


def query_network(req):
	resp = MultiTrajPredictionResponse()
	robot_data = req.robotTrajHistory.data
	obstacle_data = req.obsState.data
	predicted_trajectories = prediction_network.predict(robot_data, obstacle_data)

	resp.data = predicted_trajectories
	return resp


if __name__ == '__main__':
	rospy.init_node('Drone_Prediction_node')

	s = rospy.Service('/predict', MultiTrajPrediction, query_network)
	print("Ready to provide drone predictions.")
	rospy.spin()
