function [H,c] = quad_format(A,b,M,lambda) 

% this function reformats the DRT regression 
% as a quadratic program - this uses either re or im

    H = 2*(A'*A+lambda*M);
    H = (H'+H)/2;
    c = -2*b'*A;
    
end

