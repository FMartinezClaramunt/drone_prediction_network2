function dis = minDis(QuadPos)
    
    dim = size(QuadPos);
    
    M = dim(2);
    nQuad = dim(3);
    
    % minimum distance between drones
    QuadDis = zeros(M, nQuad);
    QuadDisMin = zeros(1, nQuad);
    for i = 1 : nQuad
        for j = 1 : M
            if i ~= nQuad
                QuadDis(j, i) = norm(QuadPos(:,j,i) - QuadPos(:,j,i+1));
            else
                QuadDis(j, nQuad) = norm(QuadPos(:,j,nQuad) - QuadPos(:,j,1));
            end
        end
        QuadDisMin(1,i) = min(QuadDis(:,i));
    end
    
    dis = min(QuadDisMin);
    
end