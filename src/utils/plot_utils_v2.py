import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from utils.data_handler_v2 import prepare_data, scale_data, unscale_output

class plot_scenario(object): 
    def __init__(self, model, args, save = False, display = True, figsize = (8,6), nframes = 10000, dt = 0.05): # 0.05 is the sampling period for the CCMPC_MAV simulator
        self.dt = dt
        
        folder_path = args['folder_path']
        test_datasets = [ args['test_datasets'][0] ]
        # test_datasets = args['test_datasets']

        test_data_paths = []
        for dataset in test_datasets:
            test_data_paths.append(os.path.join(folder_path, dataset + '.mat'))
        test_args = args.copy()
        test_args['X_type'] = 'pos'
        
        X_plot_data, __ = prepare_data(test_data_paths, test_args, shuffle = False, relative = False)
        self.past_trajectories = np.concatenate([X_plot_data[0], np.block(X_plot_data[1])], axis = 2)
        
        X_test, Y_test = prepare_data(test_data_paths, args, shuffle = False)
        
        # Scaling/unscaling if necessary
        if args['scale_data']:
            input_scaler = args['input_scaler']
            output_scaler = args['output_scaler']
            X_test = scale_data(X_test, input_scaler)      
            if args['kerasAPI'] == 'functional':
                X_test = [X_test[0], X_test[1][0], X_test[1][1], X_test[1][2] ]
            Y_predicted = model.predict(X_test)
            Y_predicted = unscale_output(Y_predicted, output_scaler)
        else:
            if args['kerasAPI'] == 'functional':
                X_test = [X_test[0], X_test[1][0], X_test[1][1], X_test[1][2] ]
            Y_predicted = model.predict(X_test)
        
        # Convert predicted data to absolute position trajectories
        self.pred_trajectories = np.zeros([Y_predicted.shape[0], Y_predicted.shape[1]+1, Y_predicted.shape[2]])
        self.pred_trajectories[:, 0, :] = self.past_trajectories[:, -1, 0:3]
        self.future_trajectories = np.zeros([Y_predicted.shape[0], Y_predicted.shape[1]+1, Y_predicted.shape[2]])
        self.future_trajectories[:, 0, :] = self.past_trajectories[:, -1, 0:3]
        
        for i in range(1, self.pred_trajectories.shape[1]): # For each timestep in the prediction horizon
            self.pred_trajectories[:, i, :] = self.pred_trajectories[:, i-1, :] + Y_predicted[:,i-1,:] * self.dt
            self.future_trajectories[:, i, :] = self.future_trajectories[:, i-1, :] + Y_test[:,i-1,:] * self.dt

        self.fig = plt.figure(figsize=figsize)
        self.ax = p3.Axes3D(self.fig)

        self.stream = self.data_stream()

        self.nframes = min(nframes, self.past_trajectories.shape[0] - 1) # -1 to account for the initialization frame

        self.ani = animation.FuncAnimation(self.fig, self.animate, frames = self.nframes, init_func=self.setup_plot, interval=50, blit=False, repeat=False)

        if save:
            self.save_animation(self.ani)

        if display:
            plt.show()


    def setup_plot(self):
        plt.cla()

        # Axis setup
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # TODO: Set up dimensions automatically
        self.ax.set_xlim3d([-5, 5])
        self.ax.set_ylim3d([-5, 5])
        self.ax.set_zlim3d([0, 3])

        # self.ax.set_title('Quadrotor animation')
        self.ax.set_proj_type('persp')
        # self.ax.view_init(25, 10) # Viewing angle

        # Plot initial data
        data = next(self.stream)
        past_traj = data[0]
        future_traj = data[1]
        pred_traj = data[2]
        n_total_quads = past_traj.shape[1] // 3
        n_pred_quads = pred_traj.shape[1] // 3

        # Quadrotor positions
        self.quadrotors = [ self.ax.plot3D(past_traj[-1, 3*i:3*i+1], past_traj[-1, 3*i+1:3*i+2], past_traj[-1, 3*i+2:3*i+3], 'o', color = 'b')[0] for i in range(n_total_quads) ]

        # Past trajectories
        self.past_lines = [ self.ax.plot3D( past_traj[:, 3*i], past_traj[:, 3*i+1], past_traj[:, 3*i+2], color = 'b' )[0] for i in range(n_total_quads) ]

        # Future trajectories
        self.future_lines = [ self.ax.plot3D( future_traj[:, 3*i], future_traj[:, 3*i+1], future_traj[:, 3*i+2], color = 'g' )[0] for i in range(n_pred_quads) ]

        # Predicted trajectories
        self.pred_lines =  [ self.ax.plot3D( pred_traj[:, 3*i], pred_traj[:, 3*i+1], pred_traj[:, 3*i+2], color = 'r' )[0] for i in range(n_pred_quads) ]

        self.ax.legend(handles = [self.quadrotors[0], self.past_lines[0], self.future_lines[0], self.pred_lines[0]], labels = ["Quadrotors", "Past trajectory", "Future trajectory", "Predicted trajectory"])
        # self.ax.legend()

        return self.quadrotors, self.past_lines, self.future_lines, self.pred_lines,

    def animate(self, iteration):
        data = next(self.stream)
        past_traj = data[0]
        future_traj = data[1]
        pred_traj = data[2]
        n_total_quads = past_traj.shape[1] //3
        n_pred_quads = pred_traj.shape[1] // 3

        # Quadrotor positions
        for i in range(n_total_quads):
            # self.quadrotors[i]._offset3d = ( past_traj[-1, 3*i:3*i+1], past_traj[-1, 3*i+1:3*i+2], past_traj[-1, 3*i+2:3*i+3] )
            self.quadrotors[i].set_data( past_traj[-1, 3*i:3*i+1], past_traj[-1, 3*i+1:3*i+2] )
            self.quadrotors[i].set_3d_properties( past_traj[-1, 3*i+2:3*i+3] )

            self.past_lines[i].set_data( past_traj[:, 3*i], past_traj[:, 3*i+1] )
            self.past_lines[i].set_3d_properties( past_traj[:, 3*i+2] )
        
        for i in range(n_pred_quads):
            self.future_lines[i].set_data( future_traj[:, 3*i], future_traj[:, 3*i+1] )
            self.future_lines[i].set_3d_properties( future_traj[:, 3*i+2] )

            self.pred_lines[i].set_data( pred_traj[:, 3*i], pred_traj[:, 3*i+1] )
            self.pred_lines[i].set_3d_properties( pred_traj[:, 3*i+2] )

        # if iteration == self.nframes-1:
        # 	self.save_animation(self.ani)
        # 	plt.close()

        return self.quadrotors, self.past_lines, self.future_lines, self.pred_lines,

    def data_stream(self):
        """ Generator for the data to be plotted """
        while True:
            for past_traj, future_traj, pred_traj in zip(self.past_trajectories, self.future_trajectories, self.pred_trajectories):
                data = []
                data.append(past_traj)
                data.append(future_traj)
                data.append(pred_traj)

                yield data

    def save_animation(self, ani):
        self.writer =  animation.writers['ffmpeg']
        self.writer = self.writer(fps=20, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        video_name = './Videos/video.mp4'
        self.ani.save(video_name, writer=self.writer)
