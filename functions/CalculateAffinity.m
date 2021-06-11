function [affinity] = CalculateAffinity(data)

% set the parameters
sigma = 1;
affinity=zeros(size(data,1));
for i=1:size(data,1)    
    for j=[1:i-1 i+1:size(data,1)]
        dist = norm(data(i,:)-data(j,:))^2; 
        affinity(i,j) = exp(-dist/(2*sigma^2));
    end
end


