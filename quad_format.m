function [H,c] = quad_format(A,b,M,lambda) 

H=2*(A'*A+lambda*M);
c=-2*b'*A;

end

