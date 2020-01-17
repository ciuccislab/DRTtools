function out_M_im = assemble_M_im(freq, epsilon, rbf_type, der_used)

% This is the matrix of all inner products 
% of the RBF elements
std_freq = std(diff(log(1./freq)));
mean_freq = mean(diff(log(1./freq)));
N_freq = numel(freq);
R=zeros(1,N_freq);
C=zeros(N_freq,1);
out_M_im_temp = zeros(N_freq);
out_M_im = zeros(N_freq+2, N_freq+2);

switch der_used

    case '1st-order'
    
        if std_freq/mean_freq<1 && ~strcmp(rbf_type,'piecewise')%(error in frequency difference <1% make sure that the terms are evenly distributed)
            for iter_freq_n = 1: N_freq
                
                    freq_n = freq(iter_freq_n);
                    freq_m = freq(1);
                    C(iter_freq_n, 1) = inner_prod_rbf(freq_n, freq_m, epsilon, rbf_type);
                    
            end  

            for iter_freq_m = 1: N_freq

                    freq_n = freq(1);
                    freq_m = freq(iter_freq_m);
                    R(1, iter_freq_m) = inner_prod_rbf(freq_n, freq_m, epsilon, rbf_type);

            end

            out_M_im_temp = toeplitz(C,R);

        elseif strcmp(rbf_type,'piecewise')
            
            out_L_temp = zeros(N_freq-1, N_freq);

                for iter_freq_n = 1: N_freq-1

                        delta_loc = log((1/freq(iter_freq_n+1))/(1/freq(iter_freq_n)));

                        out_L_temp(iter_freq_n,iter_freq_n) = -1/delta_loc;
                        out_L_temp(iter_freq_n,iter_freq_n+1) = 1/delta_loc;

                end

            out_M_im_temp = out_L_temp'*out_L_temp;
            
        else%if log of tau is not evenly distributed

            for iter_freq_n = 1: N_freq

                for iter_freq_m = 1: N_freq

                    freq_n = freq(iter_freq_n);
                    freq_m = freq(iter_freq_m);
                    out_M_im_temp(iter_freq_n, iter_freq_m) = inner_prod_rbf(freq_n, freq_m, epsilon, rbf_type);

                end
            end
        end

        
    case '2nd-order'
        
        if std_freq/mean_freq<1 && ~strcmp(rbf_type,'piecewise') %(error in frequency difference <1% make sure that the terms are evenly distributed)
            for iter_freq_n = 1: N_freq
                
                    freq_n = freq(iter_freq_n);
                    freq_m = freq(1);
                    C(iter_freq_n, 1) = inner_prod_rbf_2(freq_n, freq_m, epsilon, rbf_type);
                    
            end  

            for iter_freq_m = 1: N_freq

                    freq_n = freq(1);
                    freq_m = freq(iter_freq_m);
                    R(1, iter_freq_m) = inner_prod_rbf_2(freq_n, freq_m, epsilon, rbf_type);

            end

            out_M_im_temp = toeplitz(C,R);

        elseif strcmp(rbf_type,'piecewise')

            out_L_temp = zeros((N_freq-2), N_freq);

                for iter_freq_n = 1: (N_freq-2)

                    delta_loc = log((1/freq(iter_freq_n+1))/(1/freq(iter_freq_n)));

                    out_L_temp(iter_freq_n,iter_freq_n) = 1/delta_loc^2;
                    out_L_temp(iter_freq_n,iter_freq_n+1) = -2/delta_loc^2;
                    out_L_temp(iter_freq_n,iter_freq_n+2) = 1/delta_loc^2;

                end

            out_M_im_temp = out_L_temp'*out_L_temp;

        else %if log of tau is not evenly distributed

            for iter_freq_n = 1: N_freq

                for iter_freq_m = 1: N_freq

                    freq_n = freq(iter_freq_n);
                    freq_m = freq(iter_freq_m);
                    out_M_im_temp(iter_freq_n, iter_freq_m) = inner_prod_rbf_2(freq_n, freq_m, epsilon, rbf_type);

                end
            end
        end
    
end

out_M_im(3:end, 3:end) = out_M_im_temp;

end

