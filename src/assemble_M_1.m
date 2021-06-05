function out_M = assemble_M_1(freq, epsilon, rbf_type)

%   this function assembles the M matrix which contains the 
%   the inner products of 1st-derivative of the discretization rbfs
%   size of M matrix depends on the number of collocation points, i.e. tau vector
%   we assume that the tau vector is the inverse of the freq vector

%   first get number of frequencies
    N_freq = numel(freq);
    
%   define the M output matrix
    out_M = zeros(N_freq+2, N_freq+2);
    out_M_temp = zeros(N_freq);
    
%   compute if the frequencies are sufficienly log spaced
    std_diff_freq = std(diff(log(1./freq)));
    mean_diff_freq = mean(diff(log(1./freq)));

%   if they are, then we apply the toeplitz trick
    toeplitz_trick = std_diff_freq/mean_diff_freq<0.01;

%   if terms are evenly distributed & do not use PWL,
%   then we do compute only N terms
%   else we compute all terms with brute force
    if toeplitz_trick && ~strcmp(rbf_type,'Piecewise linear')
        
        % define vectors R and C
        R = zeros(1,N_freq);
        C = zeros(N_freq,1);
        
        % for clarity the C and R computations are separated
        for iter_freq_n = 1: N_freq
            
            freq_n = freq(iter_freq_n);
            freq_m = freq(1);
            C(iter_freq_n, 1) = inner_prod_rbf_1(freq_n, freq_m, epsilon, rbf_type);
                    
        end  

        for iter_freq_m = 1: N_freq

            freq_n = freq(1);
            freq_m = freq(iter_freq_m);
            R(1, iter_freq_m) = inner_prod_rbf_1(freq_n, freq_m, epsilon, rbf_type);

        end

            out_M_temp = toeplitz(C,R);

    % if piecewise linear discretization
    elseif strcmp(rbf_type,'Piecewise linear') 

            out_L_temp = zeros(N_freq-1, N_freq);

                for iter_freq_n = 1:N_freq-1

                    delta_loc = log((1/freq(iter_freq_n+1))/(1/freq(iter_freq_n)));
                    out_L_temp(iter_freq_n,iter_freq_n) = -1/delta_loc;
                    out_L_temp(iter_freq_n,iter_freq_n+1) = 1/delta_loc;

                end

            out_M_temp = out_L_temp'*out_L_temp;
            
    % compute rbf with brute force
    else 
        for iter_freq_n = 1: N_freq

            for iter_freq_m = 1: N_freq

                freq_n = freq(iter_freq_n);
                freq_m = freq(iter_freq_m);
                out_M_temp(iter_freq_n, iter_freq_m) = inner_prod_rbf_1(freq_n, freq_m, epsilon, rbf_type);

            end
        end
    %             out_L_temp = chol(out_M_temp);
    end

    out_M(3:end, 3:end) = out_M_temp;

end