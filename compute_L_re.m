function out_L = compute_L_re(freq)

omega = 2*pi*freq;
tau = 1./freq;
N_freq = numel(freq);

out_L_temp = zeros(N_freq-1, N_freq+1);
out_L = zeros(N_freq-1, N_freq+2);

for p = 1: (N_freq-1)
        
        delta_loc = log(tau(p+1)/tau(p));
        
        out_L_temp(p,p+1) = -1/delta_loc;
        out_L_temp(p,p+2) = 1/delta_loc;
        
end
out_L(:,2:end) = out_L_temp;

end
        