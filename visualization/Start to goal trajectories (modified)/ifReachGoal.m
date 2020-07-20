function flag = ifReachGoal(pos, vel, goal)

    % determine if the robot reaches the goal
    flag = 0;
    dis_x = abs(pos(1)-goal(1));
    dis_y = abs(pos(2)-goal(2));
    dis_z = abs(pos(3)-goal(3));

    speed = norm(vel);

    epsilon = 0.03;
    epsilon_vel = 0.03;
    
    if dis_x < epsilon && dis_y < epsilon && dis_z < epsilon && speed < epsilon_vel
        flag = 1;
    end
    
    
end