function totLength = tra_length_cal(posArray)
    num = size(posArray, 2);
    totLength = 0;
    for i = 1 : num-1
        len = norm(posArray(:,i+1) - posArray(:,i));
        totLength = totLength + len;
    end

end