function [H,c] = quad_format_combined(A_re,A_im,b_re,b_im,M_re,M_im,lambda) 

% dummy_vec=zeros(size(b_im));
% A_im_2 = [dummy_vec,A_im(:,:)];

H=2*(0.5*(A_re'*A_re+A_im'*A_im)+lambda*M_re);
c=-2*0.5*(b_im'*A_im+b_re'*A_re);



end

