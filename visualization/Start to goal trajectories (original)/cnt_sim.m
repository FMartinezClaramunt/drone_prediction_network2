clear
clc

%% set up
dt = 0.05;
N  = 20;
rQuad   = 0.3;
nQuad   = 6;
scn_6_0

%% read data
% logged data
data = load('cnt_sim_6_0_2.mat');
clc
% set valid data span
ind_max = 1;
while ~isnan(data.log_MPC(1,ind_max,1))
    ind_max = ind_max + 1;
end
ind_max = ind_max - 2;
% set valid data span
span = (1:ind_max)';
% quad position
QuadPos = data.log_QuadPos(1:3,span,:);


%% data processing
%% trajectories length
QuadTraLeng = zeros(nQuad, 1);
for k = 1 : nQuad
    QuadTraLeng(k) = tra_length_cal(QuadPos(:,:,k));
end
length_mean = mean(QuadTraLeng)
length_std  = std(QuadTraLeng)
length_max  = max(QuadTraLeng)
length_min  = min(QuadTraLeng)
%% time to reach the goal
QuadTraTime = zeros(nQuad, 1);
for k = 1 : nQuad
    goal = quadEndPos(:,k);
   % determine when reaching the goal
   for j = 1 : ind_max
       pos  = QuadPos(:,j,k);
       vel = [0;0;0];
       if ifReachGoal(pos, vel, goal)
           QuadTraTime(k) = j*dt;
           break;
       end
   end
   if QuadTraTime(k) < dt
       QuadTraTime(k) = ind_max*dt;
   end
end
time_mean = mean(QuadTraTime);
time_std  = std(QuadTraTime);
%% minimum distance
minDis = minDis(QuadPos)



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
axis([-2.8 2.8 -2.8 2.8])
xlabel('x [m]')
ylabel('y [m]')
% quad
diameter = 0.6;
sacle   = 0.24;
proprad = 0.15*diameter;
for k = 1 : nQuad
    plot(QuadPos(1,:,k), QuadPos(2,:,k),...
        '-','color',color_map(k), 'linewidth', 2);
    % quad
    quadEluer = [0;0;0];
    R = rotateEuler(quadEluer);
    pos = [QuadPos(1:2,1,k);0];
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
% xz-plane
figure;
hold all;
grid on;
box on;
axis([-2.5 2.5 0.5 2.4])
xlabel('x [m]')
ylabel('z [m]')
for k = 1 : nQuad
    plot(QuadPos(1,:,k), QuadPos(3,:,k),...
        '-','color',color_map(k), 'linewidth', 2);
    % quad
    plot(QuadPos(1,1,k), QuadPos(3,1,k),...
        'o','color',color_map(k), 'MarkerSize', 30, ...
        'MarkerFaceColor', color_map(k), 'MarkerEdgeColor', color_map(k));
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





