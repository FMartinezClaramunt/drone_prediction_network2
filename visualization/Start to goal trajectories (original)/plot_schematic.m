% plot a schematic of collision avoidance in dynamic environments
clear
clc

%% set up
dt = 0.05;
N  = 20;
aObs = 0.5;
bObs = 0.5;
cObs = 1.2;
rQuad   = 0.3;
nQuad   = 2;
nObs    = 2;
delta_r = 0.03;
delta_o = 0.03;

%% load data
data = load('plot_2_2_log_08092018_143719.mat');
load QuadPosCovPlan.mat
clc
k = 427;
% obstacle position
obsPos_1 = data.log_ObsState(1:3, k, 1);
obsPos_2 = data.log_ObsState(1:3, k, 2);
% drone position and eler state
quadPos_1 = data.log_QuadZk(5:7, k, 1);
quadPos_2 = data.log_QuadZk(5:7, k, 2);
quadEluer_1 = data.log_QuadStatesReal(7:9, k, 1);
quadEluer_2 = data.log_QuadStatesReal(7:9, k, 2);
% drone setpoint
quadSet_1 = data.log_QuadSet(1:3, k, 1);
quadSet_2 = data.log_QuadSet(1:3, k, 2);
% drone MPC plan
quadMPC_1 = data.log_QuadMPCplan(:, :, k, 1);
quadMPC_2 = data.log_QuadMPCplan(:, :, k, 2);
% drone uncerainties
quadStateCov_1 = data.log_QuadStateCov(:, :, k, 1);
quadStateCov_2 = data.log_QuadStateCov(:, :, k, 2);
% drone predicted uncertianties
quadPosCovPlan_1 = QuadPosCovPlan(:, :, 1);
quadPosCovPlan_2 = QuadPosCovPlan(:, :, 2);


%% plot schematic
figure;
hold all;
grid on;
box on;
axis([-3 3 -2.4 2.4 0 2.4]);
xlabel('x [m]')
ylabel('y [m]')
zlabel('z [m]')

%% obstacles
% first
[xEll_1, yEll_1, zEll_1] = ellipsoid(obsPos_1(1), obsPos_1(2), obsPos_1(3),...
    aObs, bObs, cObs);
surface(xEll_1, yEll_1, zEll_1, 'EdgeColor',[0.50 0.20 0.00],...
    'FaceColor',[1.00 0.40 0.00],'FaceAlpha',0.01);
% second
[xEll_2, yEll_2, zEll_2] = ellipsoid(obsPos_2(1), obsPos_2(2), obsPos_2(3),...
    aObs, bObs, cObs);
surface(xEll_2, yEll_2, zEll_2, 'EdgeColor',[0.50 0.20 0.00],...
    'FaceColor',[1.00 0.40 0.00],'FaceAlpha',0.01);

%% quadrotors
diameter = 0.5;
sacle   = 0.24;
proprad = 0.15*diameter;
% first
R_1 = rotateEuler(quadEluer_1);
rotCenter(1,:) = quadPos_1 + R_1 * sacle*diameter * [-1,1,0]';
rotCenter(2,:) = quadPos_1 + R_1 * sacle*diameter * [1,1,0]';
rotCenter(3,:) = quadPos_1 + R_1 * sacle*diameter * [1,-1,0]';
rotCenter(4,:) = quadPos_1 + R_1 * sacle*diameter * [-1,-1,0]';
line([rotCenter(1,1),rotCenter(3,1)],[rotCenter(1,2),rotCenter(3,2)],...
    [rotCenter(1,3),rotCenter(3,3)], 'color', 'r', 'linewidth', 2);
line([rotCenter(2,1),rotCenter(4,1)],[rotCenter(2,2),rotCenter(4,2)],...
    [rotCenter(2,3),rotCenter(4,3)], 'color', 'r', 'linewidth', 2);
for i = 1:4
    rotor_points = plotCircle3D(rotCenter(i,:),[0, 0, 1], proprad);
    plot3(rotor_points(1,:), rotor_points(2,:), rotor_points(3,:),'-',...
        'color','r','linewidth',2);
end
% second
R_2 = rotateEuler(quadEluer_2);
rotCenter(1,:) = quadPos_2 + R_2 * sacle*diameter * [-1,1,0]';
rotCenter(2,:) = quadPos_2 + R_2 * sacle*diameter * [1,1,0]';
rotCenter(3,:) = quadPos_2 + R_2 * sacle*diameter * [1,-1,0]';
rotCenter(4,:) = quadPos_2 + R_2 * sacle*diameter * [-1,-1,0]';
line([rotCenter(1,1),rotCenter(3,1)],[rotCenter(1,2),rotCenter(3,2)],...
    [rotCenter(1,3),rotCenter(3,3)], 'color', 'b', 'linewidth', 2);
line([rotCenter(2,1),rotCenter(4,1)],[rotCenter(2,2),rotCenter(4,2)],...
    [rotCenter(2,3),rotCenter(4,3)], 'color', 'b', 'linewidth', 2);
for i = 1:4
    rotor_points = plotCircle3D(rotCenter(i,:),[0, 0, 1], proprad);
    plot3(rotor_points(1,:), rotor_points(2,:), rotor_points(3,:),'-',...
        'color','b','linewidth',2);
end

%% quadrotor setpoint
% first
plot3(quadSet_1(1), quadSet_1(2), quadSet_1(3), 'o','MarkerSize',10, ...
    'MarkerEdgeColor','r', 'MarkerFaceColor', 'r');
% second
plot3(quadSet_2(1), quadSet_2(2), quadSet_2(3), 's','MarkerSize',10, ...
    'MarkerEdgeColor','b', 'MarkerFaceColor', 'b');
% plot defined goal path
ta = 1.4;
tb = 0.9;
tperiod = 18;
nTemp = 18*10;
set_x = zeros(nTemp,1);
set_y = zeros(nTemp,1);
set_z = zeros(nTemp,1);
for i = 1 : nTemp
    set_x(i) = ta*cos(2*pi*i*0.1/tperiod);
    set_y(i) = tb*sin(4*pi*i*0.1/tperiod);
    set_z(i) = 1.1;
end
plot3(set_x, set_y, set_z, '-m', 'linewidth',2);
plot3(2.3, 0, 2.0, 's','MarkerSize',10, ...
    'MarkerEdgeColor','g', 'MarkerFaceColor', 'g');


%% quadrotor MPC plan
plot3(quadMPC_1(5, :), quadMPC_1(6, :), quadMPC_1(7, :), '-r', 'LineWidth', 2);
plot3(quadMPC_2(5, :), quadMPC_2(6, :), quadMPC_2(7, :), '-b', 'LineWidth', 2);


%% quadrotor predicted uncertainties
ind_Vec = [1; 5; 10; 15; 20];
% first
for i = 1:5
    ind = ind_Vec(i);
    pos = quadMPC_1(5:7, ind);
    cov_Vec = 0.4*quadPosCovPlan_1(:, ind);
    cov = [cov_Vec(1), cov_Vec(4), cov_Vec(5); ... 
           cov_Vec(4), cov_Vec(2), cov_Vec(6); ... 
           cov_Vec(5), cov_Vec(6), cov_Vec(3)]; 
   [ConfiEll_x, ConfiEll_y, ConfiEll_z] = getErrorEllipsePoint(...
       cov, [pos(1),pos(2),pos(3)], 3, 0);
   surface(ConfiEll_x, ConfiEll_y, ConfiEll_z,'EdgeColor',[0.5 0.5 0.5],...
       'FaceColor',[0.45 0.94 0.45],'FaceAlpha',0.01);
end
% second
for i = 1:5
    ind = ind_Vec(i);
    pos = quadMPC_2(5:7, ind);
    cov_Vec = 0.6*quadPosCovPlan_2(:, ind);
    cov = [cov_Vec(1), cov_Vec(4), cov_Vec(5); ... 
           cov_Vec(4), cov_Vec(2), cov_Vec(6); ... 
           cov_Vec(5), cov_Vec(6), cov_Vec(3)]; 
   [ConfiEll_x, ConfiEll_y, ConfiEll_z] = getErrorEllipsePoint(...
       cov, [pos(1),pos(2),pos(3)], 3, 0);
   surface(ConfiEll_x, ConfiEll_y, ConfiEll_z,'EdgeColor',[0.5 0.5 0.5],...
       'FaceColor',[0.45 0.94 0.45],'FaceAlpha',0.01);
end

% second


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







