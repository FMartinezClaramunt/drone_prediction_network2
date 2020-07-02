import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.io import loadmat

# root_name = "visualization/test_centralized_performance"
root_name = "visualization/test_RNNpred_performance"
data_path = root_name + ".mat"
recording_path = root_name + ".mp4"

class plot_simulation_log():
    def __init__(self, data_path, recording_path, figsize=(8,6)):
        """
        Predictions has shape (number of samples, prediction horizon + 1, number of coordinates (3), number of quadrotors) [4]
        Trajectories has shape (number of samples + past_horizon + prediction_horizon - 1, number of coordinates (3), number of quadrotors) [3]
        Obstacle trajectories has shape (number of samples + past_horizon + prediction_horizon - 1, number of coordinates (3), number of obstacles) [3]
        Goals has shape (number of samples + past_horizon + prediction_horizon - 1, number of coordinates (3), number of quadrotors) [3]
        
        The reason for the missmatch in the number of samples is that we need extra steps at the beginning so that we can feed them into the network, and extra steps at the end so that we can compare the prediction vs the trajectory up until the end.
        """
        data = loadmat(data_path)
        
        self.nframes = 500
        self.plot_ellipsoids = False
        self.plot_goals = True
        
        self.quadrotor_radius = [0.3, 0.3, 0.4]
        self.obstacle_radius = [0.4, 0.4, 0.9]
        
        self.dt = 0.05

        self.n_quadrotors = data['log_quad_state_real'].shape[2]
        self.n_obstacles = data['log_obs_state_est'].shape[2]
        self.prediction_horizon = data["log_quad_pred_path"].shape[1]
        self.past_horizon = 10
        
        self.trajectories = np.moveaxis(data["log_quad_state_real"][0:3, :, :], [0, 1], [1, 0])
        self.obstacle_trajectories = np.moveaxis(data["log_obs_state_est"][0:3, :, :], [0, 1], [1, 0])
        self.predictions = np.moveaxis(data["log_quad_pred_path"], [0, 2], [2, 0]) # 3 x N x logsize x n
        self.mpc_paths = np.moveaxis(data["log_quad_mpc_plan"], [0, 2], [2, 0])
        self.goals = np.moveaxis(data["log_quad_goal"][0:3, :, :], [0, 1], [1, 0])
        
        self.mins = [5, 5, 3]
        self.maxs = [-5, -5, 0]
        
        self.recording_path = recording_path
        
        self.fig = plt.figure(figsize=figsize)
        self.ax = p3.Axes3D(self.fig)

        # self.stream = self.data_stream()
        self.ani = FuncAnimation(self.fig, self.animate, frames = self.nframes, init_func=self.setup_plot, interval=self.dt, blit=False, repeat=False)
        self.save_recording()
        
    def setup_plot(self):
        plt.cla()

        # Axis setup
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim3d([self.mins[0], self.maxs[0]])
        self.ax.set_ylim3d([self.mins[1], self.maxs[1]])
        self.ax.set_zlim3d([self.mins[2], self.maxs[2]])
        self.ax.set_proj_type('persp')

        quadrotors = [ self.ax.plot3D([], [], [], 'o', color = 'b')[0] for i in range(self.n_quadrotors) ]
        if self.n_obstacles > 0:
            obstacles = [ self.ax.plot3D([], [], [], 'o', color = 'k')[0] for i in range(self.n_obstacles) ]
        else:
            obstacles = None
        past_trajectories = [ self.ax.plot3D( [], [], [], color = 'b' )[0] for i in range(self.n_quadrotors) ]
        future_trajectories = [ self.ax.plot3D( [], [], [], color = 'g' )[0] for i in range(self.n_quadrotors) ]
        predicted_trajectories = [ self.ax.plot3D( [], [], [], color = 'r' )[0] for i in range(self.n_quadrotors) ]
        if self.goals == [] or not self.plot_goals:
            goals = None
            if self.n_obstacles > 0:
                self.ax.legend(handles = [quadrotors[0], obstacles[0], past_trajectories[0], future_trajectories[0], predicted_trajectories[0]], labels = ["Quadrotor", "Obstacle", "Past trajectory", "Future trajectory", "Predicted trajectory"])
            else:
                self.ax.legend(handles = [quadrotors[0], past_trajectories[0], future_trajectories[0], predicted_trajectories[0]], labels = ["Quadrotor", "Past trajectory", "Future trajectory", "Predicted trajectory"])
        else:
            goals = [ self.ax.plot3D([], [], [], 'o', color = 'c', zorder=0)[0] for i in range(self.n_quadrotors) ]
            if self.n_obstacles > 0:
                self.ax.legend(handles = [quadrotors[0], obstacles[0], past_trajectories[0], future_trajectories[0], predicted_trajectories[0], goals[0]], labels = ["Quadrotor", "Obstacle", "Past trajectory", "Future trajectory", "Predicted trajectory", "Goal"])
            else:
                self.ax.legend(handles = [quadrotors[0], past_trajectories[0], future_trajectories[0], predicted_trajectories[0], goals[0]], labels = ["Quadrotor", "Past trajectory", "Future trajectory", "Predicted trajectory", "Goal"])

        self.plot_objects = {
            'quadrotors': quadrotors,
            'obstacles': obstacles,
            'past_trajectories': past_trajectories,
            'future_trajectories': future_trajectories,
            'predicted_trajectories': predicted_trajectories,
            'goals': goals,
        }

    def animate(self, iteration):
        current_idx = iteration + self.past_horizon
        future_idx = current_idx + self.prediction_horizon

        for quad in range(self.n_quadrotors):
            self.plot_objects['quadrotors'][quad].set_data( self.trajectories[current_idx-1, 0, quad], self.trajectories[current_idx-1, 1, quad] )
            self.plot_objects['quadrotors'][quad].set_3d_properties( self.trajectories[current_idx-1, 2, quad] )

            self.plot_objects['past_trajectories'][quad].set_data( self.trajectories[iteration:current_idx, 0, quad], self.trajectories[iteration:current_idx, 1, quad] )
            self.plot_objects['past_trajectories'][quad].set_3d_properties( self.trajectories[iteration:current_idx, 2, quad] )

            if self.plot_objects['goals'] is not None and self.plot_goals:
                self.plot_objects['goals'][quad].set_data( self.goals[current_idx-1, 0, quad], self.goals[current_idx-1, 1, quad])
                self.plot_objects['goals'][quad].set_3d_properties( self.goals[current_idx-1, 2, quad])

        for obs in range(self.n_obstacles):
            self.plot_objects['obstacles'][obs].set_data( self.obstacle_trajectories[current_idx-1, 0, obs], self.obstacle_trajectories[current_idx-1, 1, obs] )
            self.plot_objects['obstacles'][obs].set_3d_properties( self.obstacle_trajectories[current_idx-1, 2, obs] )

        for quad in range(self.n_quadrotors):
            self.plot_objects['future_trajectories'][quad].set_data( self.trajectories[current_idx-1:future_idx, 0, quad], self.trajectories[current_idx-1:future_idx, 1, quad] )
            self.plot_objects['future_trajectories'][quad].set_3d_properties( self.trajectories[current_idx-1:future_idx, 2, quad] )

            self.plot_objects['predicted_trajectories'][quad].set_data( self.predictions[current_idx-1, :, 0, quad], self.predictions[current_idx-1, :, 1, quad] )
            self.plot_objects['predicted_trajectories'][quad].set_3d_properties( self.predictions[current_idx-1, :, 2, quad] )

        return self.plot_objects

    def save_recording(self):
        self.writer = FFMpegWriter(fps=int(1/self.dt), metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264']) # Define animation writer
        video_dir = os.path.dirname(self.recording_path) # Video directory
        Path(video_dir).mkdir(parents=True, exist_ok=True) # Create video directory if it doesn't exist
        self.ani.save(self.recording_path, writer=self.writer) # Save animation
            
obj = plot_simulation_log(data_path, recording_path)           
            
            
            
            
            
