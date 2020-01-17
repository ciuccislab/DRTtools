function out_A_re = assemble_A_re(freq, epsilon, rbf_type)

% Assemble the A_prime matrix
std_freq = std(diff(log(1./freq)));
mean_freq = mean(diff(log(1./freq)));
N_freq = numel(freq);
R = zeros(1,N_freq);
C = zeros(N_freq,1);
out_A_re = zeros(N_freq, N_freq+2);
out_A_re_temp = zeros(N_freq);

if std_freq/mean_freq<1 && ~strcmp(rbf_type,'piecewise')  %(error in frequency difference <1% make sure that the terms are evenly distributed)
    for iter_freq_n = 1: N_freq
        freq_n = freq(iter_freq_n);
            freq_m = freq(1);
            C(iter_freq_n, 1) = g_i(freq_n, freq_m, epsilon, rbf_type);
    end       
        for iter_freq_m = 1: N_freq

            freq_n = freq(1);
            freq_m = freq(iter_freq_m);
            R(1, iter_freq_m) = g_i(freq_n, freq_m, epsilon, rbf_type);

        end

    out_A_re_temp= toeplitz(C,R);
    
else
    for iter_freq_n = 1: N_freq
    
        for iter_freq_m = 1: N_freq

            freq_n = freq(iter_freq_n);
            freq_m = freq(iter_freq_m);
            
            if strcmp(rbf_type,'piecewise')

                if iter_freq_m == 1
                    
                    freq_m_plus_1 = freq(iter_freq_m+1);
                    out_A_re_temp(iter_freq_n, iter_freq_m) = 0.5/(1+((2*pi*freq_n/freq_m))^2)*log((1/freq_m_plus_1)/(1/freq_m));
                
                elseif iter_freq_m == N_freq
                    
                    freq_m_minus_1 = freq(iter_freq_m-1);
                    out_A_re_temp(iter_freq_n, iter_freq_m) = 0.5/(1+((2*pi*freq_n/freq_m))^2)*log((1/freq_m)/((1/freq_m_minus_1)));
                
                else
                    
                    freq_m_plus_1 = freq(iter_freq_m+1);
                    freq_m_minus_1 = freq(iter_freq_m-1);
                    out_A_re_temp(iter_freq_n, iter_freq_m) = 0.5/(1+((2*pi*freq_n/freq_m))^2)*log((1/freq_m_plus_1)/(1/freq_m_minus_1));
                
                end
            
            else
                out_A_re_temp(iter_freq_n, iter_freq_m) = g_i(freq_n, freq_m, epsilon, rbf_type);
            end
        end
    end
end
    
out_A_re(:, 3:end) = out_A_re_temp;
out_A_re(:,2) = 1;

end