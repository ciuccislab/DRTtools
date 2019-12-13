function out_A_re = assemble_A_re(freq, epsilon, rbf_type, integration_algorithm)

% Assemble the A_prime matrix
std_freq = std(diff(log(1./freq)));
mean_freq = mean(diff(log(1./freq)));
R=zeros(1,numel(freq));
C=zeros(numel(freq),1);
out_A_re = zeros(numel(freq), numel(freq)+2);
out_A_re_temp = zeros(numel(freq));

if std_freq/mean_freq<1  %(error in frequency difference <1% make sure that the terms are evenly distributed)
    for iter_freq_n = 1: numel(freq)
        freq_n = freq(iter_freq_n);
            freq_m = freq(1);
            C(iter_freq_n, 1) = g_i(freq_n, freq_m, epsilon, rbf_type, integration_algorithm);
    end       
        for iter_freq_m = 1: numel(freq)

            freq_n = freq(1);
            freq_m = freq(iter_freq_m);
            R(1, iter_freq_m) = g_i(freq_n, freq_m, epsilon, rbf_type, integration_algorithm);

        end

    out_A_re_temp= toeplitz(C,R);
    
else
    for iter_freq_n = 1: numel(freq)
    
        for iter_freq_m = 1: numel(freq)

            freq_n = freq(iter_freq_n);
            freq_m = freq(iter_freq_m);
            out_A_re_temp(iter_freq_n, iter_freq_m) = g_i(freq_n, freq_m, epsilon, rbf_type, integration_algorithm);

        end
    end

end

out_A_re(:, 3:end) = out_A_re_temp;
out_A_re(:,2) = 1;

end