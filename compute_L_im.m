function out_L= compute_L_im(freq)

omega = 2*pi*freq;
tau = 1./freq;
N_freq = numel(freq);

out_L = zeros(N_freq-1, N_freq+2);
out_L_temp = zeros(N_freq-1, N_freq);

for p = 1: (N_freq-1)
        
        delta_loc = log(tau(p+1)/tau(p));
        
        out_L_temp(p,p) = -1/delta_loc;
        out_L_temp(p,p+1) = 1/delta_loc;
        
end

out_L(:,3:end) = out_L_temp;

end

        