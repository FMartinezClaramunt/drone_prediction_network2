import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3

class plot_predictions():
    def __init__(self, data, args, recording_path, figsize=(8,6)):
        """
        Predictions has shape (number of samples, prediction horizon + 1, number of coordinates (3), number of quadrotors)
        Trajectories has shape (number of samples + past_horizon + prediction_horizon - 1, number of coordinates (3), number of quadrotors)
        Goals has shape (number of samples + past_horizon + prediction_horizon - 1, number of coordinates (3), number of quadrotors)
        
        The reason for the missmatch in the number of samples is that we need extra steps at the beginning so that we can feed them into the network, and extra steps at the end so that we can compare the prediction vs the trajectory up until the end.
        """
        self.nframes = min(args.n_frames, data['predictions'].shape[0])
        self.n_total_quadrotors = data['trajectories'].shape[2]
        if len(data['predictions'].shape) == 3:
            self.n_predicted_quadrotors = 1
        else:
            self.n_predicted_quadrotors = data['predictions'].shape[3]
        self.trajectories = data['trajectories']
        self.predictions = data['predictions']
        self.goals = data['goals']
        
        self.mins = self.trajectories.min(axis=-1).min(axis=0)-0.2
        self.maxs = self.trajectories.max(axis=-1).max(axis=0)+0.2
        
        self.dt = args.dt
        self.past_horizon = args.past_horizon
        self.prediction_horizon = args.prediction_horizon
        
        self.recording_path = recording_path
        
        self.fig = plt.figure(figsize=figsize)
        self.ax = p3.Axes3D(self.fig)

        # self.stream = self.data_stream()
        self.ani = FuncAnimation(self.fig, self.animate, frames = self.nframes, init_func=self.setup_plot, interval=self.dt, blit=False, repeat=False)
        
        if args.record:
            self.save_recording()
            print("Saving animation recording")

        if args.display:
            print("Displaying animation")
            plt.show()
        
    def setup_plot(self):
        plt.cla()

        # Axis setup
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        # self.ax.set_xlim3d([-5, 5])
        # self.ax.set_ylim3d([-5, 5])
        # self.ax.set_zlim3d([0, 3])
        self.ax.set_xlim3d([self.mins[0], self.maxs[0]])
        self.ax.set_ylim3d([self.mins[1], self.maxs[1]])
        self.ax.set_zlim3d([self.mins[2], self.maxs[2]])
        self.ax.set_proj_type('persp')
        
        quadrotors = [ self.ax.plot3D([], [], [], 'o', color = 'b')[0] for i in range(self.n_total_quadrotors) ]
        past_trajectories = [ self.ax.plot3D( [], [], [], color = 'b' )[0] for i in range(self.n_total_quadrotors) ]
        future_trajectories = [ self.ax.plot3D( [], [], [], color = 'g' )[0] for i in range(self.n_predicted_quadrotors) ]
        predicted_trajectories = [ self.ax.plot3D( [], [], [], color = 'r' )[0] for i in range(self.n_predicted_quadrotors) ]
        if self.goals == []:
            goals = None
            self.ax.legend(handles = [quadrotors[0], past_trajectories[0], future_trajectories[0], predicted_trajectories[0]], labels = ["Quadrotor", "Past trajectory", "Future trajectory", "Predicted trajectory"])
        else:
            goals = [ self.ax.plot3D([], [], [], 'o', color = 'c', zorder=0)[0] for i in range(self.n_total_quadrotors) ]
            self.ax.legend(handles = [quadrotors[0], past_trajectories[0], future_trajectories[0], predicted_trajectories[0], goals[0]], labels = ["Quadrotor", "Past trajectory", "Future trajectory", "Predicted trajectory", "Goal"])
        
        self.plot_objects = {
            'quadrotors': quadrotors,
            'past_trajectories': past_trajectories,
            'future_trajectories': future_trajectories,
            'predicted_trajectories': predicted_trajectories,
            'goals': goals,
        }
        
    def animate(self, iteration):
        current_idx = iteration + self.past_horizon
        future_idx = current_idx + self.prediction_horizon
        
        for quad in range(self.n_total_quadrotors):
            self.plot_objects['quadrotors'][quad].set_data( self.trajectories[current_idx-1, 0, quad], self.trajectories[current_idx-1, 1, quad] )
            self.plot_objects['quadrotors'][quad].set_3d_properties( self.trajectories[current_idx-1, 2, quad] )
            
            self.plot_objects['past_trajectories'][quad].set_data( self.trajectories[iteration:current_idx, 0, quad], self.trajectories[iteration:current_idx, 1, quad] )
            self.plot_objects['past_trajectories'][quad].set_3d_properties( self.trajectories[iteration:current_idx, 2, quad] )
            
            if not self.plot_objects['goals'] == None:
                self.plot_objects['goals'][quad].set_data( self.goals[current_idx-1, 0, quad], self.goals[current_idx-1, 1, quad])
                self.plot_objects['goals'][quad].set_3d_properties( self.goals[current_idx-1, 2, quad])

        for quad in range(self.n_predicted_quadrotors):
            self.plot_objects['future_trajectories'][quad].set_data( self.trajectories[current_idx-1:future_idx, 0, quad], self.trajectories[current_idx-1:future_idx, 1, quad] )
            self.plot_objects['future_trajectories'][quad].set_3d_properties( self.trajectories[current_idx-1:future_idx, 2, quad] )
            
            self.plot_objects['predicted_trajectories'][quad].set_data( self.predictions[iteration, :, 0, quad], self.predictions[iteration, :, 1, quad] )
            self.plot_objects['predicted_trajectories'][quad].set_3d_properties( self.predictions[iteration, :, 2, quad] )

        return self.plot_objects
    
    def save_recording(self):
        self.writer = FFMpegWriter(fps=int(1/self.dt), metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264']) # Define animation writer
        video_dir = os.path.dirname(self.recording_path) # Video directory
        Path(video_dir).mkdir(parents=True, exist_ok=True) # Create video directory if it doesn't exist
        self.ani.save(self.recording_path, writer=self.writer) # Save animatio