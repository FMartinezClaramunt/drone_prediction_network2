nQuad = 6;
 
%%% Storage parameters
quadStartPos = zeros(3,nQuad); %m (x,y,z) position of where MPC plan starts
quadEndPos = zeros(3,nQuad); %m (x,y,z) position of where the UAV should go
quadStartVel = zeros(3,nQuad); %m/s Simulated start velocity (deactivated)
 
%%% Quadrotor Definition
% Enter one column entry into parameters per quadrotor
 
quadStartPos(:,1) = [-2; 0.0; 1.5]; 
quadEndPos(:,1) = [2; 0; 1.5]; 
quadStartVel(:,1) = [0; 0; 0]; 

quadStartPos(:,2) = [-1; sqrt(3); 1.5]; 
quadEndPos(:,2) = [1; -sqrt(3); 1.5]; 
quadStartVel(:,2) = [0; 0; 0]; 
 
quadStartPos(:,3) = [1; sqrt(3); 1.5]; 
quadEndPos(:,3) = [-1; -sqrt(3); 1.5]; 
quadStartVel(:,3) = [0; 0; 0]; 

quadStartPos(:,4) = [2; 0.0; 1.5]; 
quadEndPos(:,4) = [-2; 0; 1.5]; 
quadStartVel(:,4) = [0; 0; 0]; 

quadStartPos(:,5) = [1; -sqrt(3); 1.5]; 
quadEndPos(:,5) = [-1; sqrt(3); 1.5]; 
quadStartVel(:,5) = [0; 0; 0]; 
 
quadStartPos(:,6) = [-1; -sqrt(3); 1.5]; 
quadEndPos(:,6) = [1; sqrt(3); 1.5]; 
quadStartVel(:,6) = [0; 0; 0];

 
 
%% Obstacle(s) Setup

% Define number of obstacles
nObs = 0;

%%% Storage parameters
obsStartPos = zeros(3,nObs); %m (x,y,z) Used for simulated obstacles
obsEndPos = zeros(3,nObs); %m (x,y,z) Used for simulated obstacles
obsTimer = zeros(2,nObs); %s (start time, end time) Used for simulated obstacles timing

obsDim = zeros(3,nObs); %m (x,y,z) Obstacle cuboid x,y,z dimensions
obsBuffers = zeros(2,nObs); %m 1st row is enclosing buffer, 2nd row is total buffer for potential field
obsStatic = zeros(1,nObs); %bool 0 or 1, tells the controller if obstacle is static for fast computations (not necessary to set).

%%% Obstacle Definition
% Enter one column entry into parameters per obstacle

% obsStartPos(:,1) = [-1.0; -3; 0.6];
% obsEndPos(:,1) = [-1.0; 3; 0.6]; 
% obsTimer(:,1) = [5; 20];
% obsDim(:,1) = [0.4; 0.4; obsStartPos(3,1)*2];
% obsBuffers(:,1) = [0.1; 0.6];
% obsStatic(:,1) = 0;
% 
% obsStartPos(:,2) = [1.0; 3; 0.6];
% obsEndPos(:,2) = [1.0; -3; 0.6]; 
% obsTimer(:,2) = [5; 20];
% obsDim(:,2) = [0.4; 0.4; obsStartPos(3,1)*2];
% obsBuffers(:,2) = [0.1; 0.6];
% obsStatic(:,2) = 0;

