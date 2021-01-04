function results = evaluate_scenario_performance(data, n_scenarios, goals_per_scenario)
if nargin < 2
    n_scenarios = 4;
end

if nargin < 3
    goals_per_scenario = 25;
end

if isstring(data)
    data = load(data);
end

dt = data.model.dt;

distance_threshold = 0.15;
speed_threshold = 0.05;

results.min_dist = {};
results.collisions = {};
results.traj_length_stats = {};
results.traj_time_stats = {};
results.velocity_stats = {};
% results.comp_time_stats = {};
results.goal_change_idxs  = {};
results.trajectory_lengths  = {};
results.trajectory_times = {};

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

%% Split trajectories
results.goal_change_idxs = [1 find(any(abs(goals(:, 2:end, :) - goals(:, 1:end-1, :)) > 0, [1,3]))+1,  span(end)];

limit = min(goals_per_scenario * 3 * n_scenarios, length(results.goal_change_idxs));
scenario_change_idxs = union(1 : goals_per_scenario * 3 : limit, limit);

%% Compute minimum distance between quadrotors
for scenario_idx = 1:n_scenarios
    min_dist = Inf;
    scenario_velocities = [];
%     init_idx = results.goal_change_idxs(scenario_change_idxs(scenario_idx));
%     end_idx = results.goal_change_idxs(scenario_change_idxs(scenario_idx+1)-1);
    
    traj_idxs = get_custom_range(scenario_change_idxs(scenario_idx), scenario_change_idxs(scenario_idx+1)-1);
    
    results.trajectory_times{scenario_idx} = [];
    results.trajectory_lengths{scenario_idx} = [];
    
    counter = 0;
    n_collisions = 0;
    for traj_idx = traj_idxs
        counter = counter + 1;
        steps = results.goal_change_idxs(traj_idx):results.goal_change_idxs(traj_idx+1)-1;
        results.trajectory_times{scenario_idx}(end+1) = (steps(end) - steps(1)) * dt;
        
        trajectories = positions(:, steps, :);
        scenario_velocities(:, end+1:end+length(steps), :) = velocities(:,steps,:);
        
        collision = false;
        
        for quad_i = 1:n_quadrotors
            for quad_j = quad_i+1:n_quadrotors
                new_min_dist = min(vecnorm(trajectories(:, :, quad_i) - trajectories(:, :, quad_j)), [], 2);
                if min_dist > new_min_dist
                    min_dist = new_min_dist;
                end
                
                if new_min_dist < 0.6
                    collision = true;
                end
            end
        end
        
        if collision
            n_collisions = n_collisions + 1;
        end
        
        results.trajectory_lengths{scenario_idx}(counter, :) = sum(vecnorm(trajectories(:,2:end,:) - trajectories(:,1:end-1,:)), 2);
    end
    
    results.min_dist{scenario_idx} = min_dist;
    results.collisions{scenario_idx} = n_collisions;
    
    results.traj_time_stats{scenario_idx}.min = min(results.trajectory_times{scenario_idx});
    results.traj_time_stats{scenario_idx}.max = max(results.trajectory_times{scenario_idx});
    results.traj_time_stats{scenario_idx}.avg = mean(results.trajectory_times{scenario_idx});
    results.traj_time_stats{scenario_idx}.std = std(results.trajectory_times{scenario_idx});
    
    results.traj_length_stats{scenario_idx}.min = min(results.trajectory_lengths{scenario_idx}, [], 'all');
    results.traj_length_stats{scenario_idx}.max = max(results.trajectory_lengths{scenario_idx}, [], 'all');
    results.traj_length_stats{scenario_idx}.avg = mean(results.trajectory_lengths{scenario_idx}, 'all');
    results.traj_length_stats{scenario_idx}.std = std(results.trajectory_lengths{scenario_idx}, 0, 'all');
    
    results.velocity_stats{scenario_idx}.avg = mean(vecnorm(scenario_velocities), 'all');
    results.velocity_stats{scenario_idx}.std = std(vecnorm(scenario_velocities), 0, 'all');
end



end

function range = get_custom_range(init_idx, end_idx)
    range1 = init_idx:3:end_idx;
    range2 = init_idx+1:3:end_idx;
    range = union(range1, range2, 'sorted');
end
