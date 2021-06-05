function out_A_re = assemble_A_re(freq, epsilon, rbf_type)

%   this function assembles the A_re matrix
%   we will assume that the tau vector is identical to the freq vector

%   first get number of frequencies
    N_freq = numel(freq);

%   the define the A_re output matrix
    out_A_re_temp = zeros(N_freq);
    out_A_re = zeros(N_freq, N_freq+2);

%   we compute if the frequencies are sufficiently log spaced
    std_diff_freq = std(diff(log(1./freq)));
    mean_diff_freq = mean(diff(log(1./freq)));

%   if they are, we apply the toeplitz trick
    toeplitz_trick = std_diff_freq/mean_diff_freq<0.01;

%   if terms are evenly distributed & do not use PWL
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
            C(iter_freq_n, 1) = g_i(freq_n, freq_m, epsilon, rbf_type);

        end

        for iter_freq_m = 1: N_freq

            freq_n = freq(1);
            freq_m = freq(iter_freq_m);
            R(1, iter_freq_m) = g_i(freq_n, freq_m, epsilon, rbf_type);

        end

        out_A_re_temp= toeplitz(C,R);

    else
    % compute using brute force

        for iter_freq_n = 1: N_freq

            for iter_freq_m = 1: N_freq

                freq_n = freq(iter_freq_n);
                freq_m = freq(iter_freq_m);

                % this is the usual PWL approximation
                if strcmp(rbf_type,'Piecewise linear')

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

                    % compute all RBF terms
                    out_A_re_temp(iter_freq_n, iter_freq_m) = g_i(freq_n, freq_m, epsilon, rbf_type);

                end

            end

        end

    end

    % the first and second row are reserved for L and R respectively
    out_A_re(:, 3:end) = out_A_re_temp;

end