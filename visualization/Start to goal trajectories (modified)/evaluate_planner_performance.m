function [min_dist, traj_length_stats, traj_time_stats, velocity_stats, comp_time_stats, goal_change_idxs, trajectory_lengths, trajectory_times] = evaluate_planner_performance(data)
% IN
% data - Simulation log data or string with the path to the data
% OUT
% min_dist
% traj_length
% traj_time
% avg_comp_time
distance_threshold = 0.15;
speed_threshold = 0.05;

if isstring(data)
    data = load(data);
end

dt = data.model.dt;

%% Determine relevant simulation steps
goals = data.log_quad_goal(1:3, :, :);
positions = data.log_quad_state_real(1:3, :, :);
velocities = data.log_quad_state_real(4:6, :, :);
n_quadrotors = size(goals, 3);

all_close_to_goal = all(vecnorm(goals-positions, 2, 1) < distance_threshold, 3);
all_stopped = all(vecnorm(velocities, 2, 1) < speed_threshold, 3);
all_reached = all_close_to_goal & all_stopped;
all_reached_idxs = find(all_reached);
last_idx = all_reached_idxs(find(all_reached_idxs(11:end) - all_reached_idxs(1:end-10) == 10, 1))-1; % All reached for 10 steps
span = 1:last_idx;

goals = goals(:, span, :);
positions = positions(:, span, :);
velocities = velocities(:, span, :);


%% Compute minimum distance between quadrotors
min_dist = Inf;
for i = 1:n_quadrotors
    for j = i+1:n_quadrotors
        new_min_dist = min(vecnorm(positions(:, :, i) - positions(:, :, j)), [], 2);
        if min_dist > new_min_dist
            min_dist = new_min_dist;
        end
    end
end    

%% Compute trajectory length and time statistics
trajectory_lengths = zeros(0, n_quadrotors);

% Split trajectories
goal_change_idxs = [0 find(any(abs(goals(:, 2:end, :) - goals(:, 1:end-1, :)) > 0, [1,3])) span(end)];
trajectory_times = (goal_change_idxs(2:end) - goal_change_idxs(1:end-1)) * dt;
traj_time_stats.min = min(trajectory_times);
traj_time_stats.max = max(trajectory_times);
traj_time_stats.avg = mean(trajectory_times);
traj_time_stats.std = std(trajectory_times);

% Compute length of each trajectory of each quadrotor
for i = 1:length(goal_change_idxs)-1
    trajectory = positions(:, goal_change_idxs(i)+1:goal_change_idxs(i+1), :);
    trajectory_lengths(i, :) = sum(vecnorm(trajectory(:,2:end,:) - trajectory(:,1:end-1,:)), 2);
end

traj_length_stats.min = min(trajectory_lengths, [], 'all');
traj_length_stats.max = max(trajectory_lengths, [], 'all');
traj_length_stats.avg = mean(trajectory_lengths, 'all');
traj_length_stats.std = std(trajectory_lengths, 0, 'all');

%% Compute velocity statistics
velocity_stats.avg = mean(vecnorm(velocities), 'all');
velocity_stats.std = std(vecnorm(velocities), 0, 'all');


%% Compute computation time statistics
comp_time_stats.avg = mean(data.log_time(span));
comp_time_stats.std = std(data.log_time(span));

