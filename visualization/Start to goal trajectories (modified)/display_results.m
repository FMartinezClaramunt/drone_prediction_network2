for i=1:length(results.min_dist)
    disp(["Experiment " + num2str(i)])
    
    disp(["Minimum distance"])
    results.min_dist{i}
    
    disp(["Num of collisions"])
    results.collisions{i}
    
    disp(["Trajectory length"])
    results.traj_length_stats{i}
    
    disp(["Trajectory time"])
    results.traj_time_stats{i}
    
    disp(["Velocity"])
    results.velocity_stats{i}
end