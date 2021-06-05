function output = res_score(res, band)
%     # count the points fallen in the band
    count = zeros(3,1);
    for n = 1:3
         count(n) = numel(find(res < n*band & res > -n*band)); % one to three sigma bands
    end 
    
    output = count/numel(res);
    
end