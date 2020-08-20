function [H,c] = quad_format_combined(A_re,A_im,b_re,b_im,M,lambda) 

% dummy_vec=zeros(size(b_im));
% A_im_2 = [dummy_vec,A_im(:,:)];

H=2*((A_re'*A_re+A_im'*A_im)+lambda*M);
c=-2*(b_im'*A_im+b_re'*A_re);



end

