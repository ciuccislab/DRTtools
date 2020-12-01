function [H,c] = quad_format_combined(A_re,A_im,b_re,b_im,M,lambda) 

% this function reformats the DRT regression 
% as a quadratic program - this uses both re and im

    H = 2*((A_re'*A_re+A_im'*A_im)+lambda*M);
    H = (H'+H)/2;
    c = -2*(b_im'*A_im+b_re'*A_re);

end

