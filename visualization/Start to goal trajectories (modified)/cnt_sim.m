clear
clc

%% set up
dt = 0.05;
N  = 20;
rQuad   = 0.3;
nQuad   = 6;
scn_6_0
clc

%% read data
% logged data
dataset_name_root = "testSwap100";
dataset_names = {char(dataset_name_root + "_centralized"), ...
    char(dataset_name_root + "_distributed"), ... 
    char(dataset_name_root + "_constVel"), ...
    char(dataset_name_root + "_RNN")};

for dataset_idx = 1:length(dataset_names)
%     close all
    dataset = string(dataset_names{dataset_idx})
    data = load(dataset + ".mat");
    [min_dist, traj_length_stats, traj_time_stats, velocity_stats, comp_time_stats, goal_change_idxs, trajectory_lengths, trajectory_times] = evaluate_planner_performance(data);
    % Display stats
    min_dist, traj_length_stats, traj_time_stats, velocity_stats, comp_time_stats
    
    % QuadGoals = data.log_quad_goal(1:3, :, :);
    % QuadPos = data.log_quad_state_real(1:3, :, :);
    % QuadVel = data.log_quad_state_real(4:6, :, :);
    % 
    % all_close_to_goal = all(vecnorm(QuadGoals-QuadPos, 2, 1) < 0.2, 3);
    % all_stopped = all(vecnorm(QuadVel, 2, 1) < 0.2, 3);
    % all_reached = all_close_to_goal & all_stopped;
    % all_reached_idxs = find(all_reached);
    % ind_min = 1;
    % ind_max = all_reached_idxs(1);

    [~, traj_idx_max] = max(trajectory_times);
    [~, traj_idx_min] = min(trajectory_times);

    span1 = goal_change_idxs(traj_idx_max)+1:goal_change_idxs(traj_idx_max+1);
    span2 = goal_change_idxs(traj_idx_min)+1:goal_change_idxs(traj_idx_min+1);

    % quad position
    QuadPos = {data.log_quad_state_real(1:3,span1,:), data.log_quad_state_real(1:3,span2,:)};


    for idx = 1:2 % Best and worst trajectories
        if idx == 1
            traj_type = 'worst';
        else
            traj_type = 'best';
        end
        
        %% plot trajectories
        color_map = ['r','b','g','k','c','m']; 
        % % 3D
        % figure;
        % hold all;
        % grid on;
        % box on;
        % axis([-3.0 3.0 -3.0 3.0 0 3.0])
        % xlabel('x [m]')
        % ylabel('y [m]')
        % zlabel('z [m]')
        % for k = 1 : nQuad
        %     plot3(QuadPos(1,:,k), QuadPos(2,:,k), QuadPos(3,:,k),...
        %         '-','color',color_map(k));
        % end
        % xy-plane
        figure;
        hold all;
        grid on;
        box on;
        axis([-3.5 3.5 -3.5 3.5])
        xlabel('x [m]')
        ylabel('y [m]')
        % quad
        diameter = 0.6;
        sacle   = 0.24;
        proprad = 0.15*diameter;
        for k = 1 : nQuad
            plot(QuadPos{idx}(1,:,k), QuadPos{idx}(2,:,k),...
                '-','color',color_map(k), 'linewidth', 2);
            % quad
            quadEluer = [0;0;0];
            R = rotateEuler(quadEluer);
            pos = [QuadPos{idx}(1:2,1,k);0];
            rotCenter(1,:) = pos + R * sacle*diameter * [-1,1,0]';
            rotCenter(2,:) = pos + R * sacle*diameter * [1,1,0]';
            rotCenter(3,:) = pos + R * sacle*diameter * [1,-1,0]';
            rotCenter(4,:) = pos + R * sacle*diameter * [-1,-1,0]';
            line([rotCenter(1,1),rotCenter(3,1)],[rotCenter(1,2),rotCenter(3,2)],...
                [rotCenter(1,3),rotCenter(3,3)], 'color', color_map(k), 'linewidth', 2);
            line([rotCenter(2,1),rotCenter(4,1)],[rotCenter(2,2),rotCenter(4,2)],...
                [rotCenter(2,3),rotCenter(4,3)], 'color', color_map(k), 'linewidth', 2);
            for i = 1:4
                rotor_points = plotCircle3D(rotCenter(i,:),[0, 0, 1], proprad);
                plot3(rotor_points(1,:), rotor_points(2,:), rotor_points(3,:),'-',...
                    'color',color_map(k),'linewidth',2);
            end
        end
        set(gcf,'color','w');
        export_fig(char(dataset + "_" + traj_type + "_xy" + ".pdf"))
        
        % xz-plane
        figure;
        hold all;
        grid on;
        box on;
        axis([-3.5 3.5 0.4 2.5])
        xlabel('x [m]')
        ylabel('z [m]')
        for k = 1 : nQuad
            plot(QuadPos{idx}(1,:,k), QuadPos{idx}(3,:,k),...
                '-','color',color_map(k), 'linewidth', 2);
            % quad
            plot(QuadPos{idx}(1,1,k), QuadPos{idx}(3,1,k),...
                'o','color',color_map(k), 'MarkerSize', 30, ...
                'MarkerFaceColor', color_map(k), 'MarkerEdgeColor', color_map(k));
        end
        set(gcf,'color','w');
        export_fig(char(dataset + "_" + traj_type + "_xz" + ".pdf"))
    end
end

%% some functions
function points = plotCircle3D(center, normal, radius)
    % Create points for 3D circle
    theta = 0:0.1:2*pi+0.1;
    v = null(normal); % Null space (normal*v = 0);
    points=repmat(center',1,size(theta,2))+radius*(v(:,1)*cos(theta)+v(:,2)*sin(theta));
end

function R = rotateEuler(euler)
    % Evaluate rotations of roll, pitch then yaw (r p y)
    r = euler(1); p = euler(2); y = euler(3);
    R = [cos(y)*cos(p), cos(y)*sin(p)*sin(r) - sin(y)*cos(r), cos(y)*sin(p)*cos(r) + sin(y)*sin(r); ...
        sin(y)*cos(p), sin(y)*sin(p)*sin(r) + cos(y)*cos(r), sin(y)*sin(p)*cos(r) - cos(y)*sin(r); ...
        -sin(p), cos(p)*sin(r), cos(p)*cos(r) ];
end





