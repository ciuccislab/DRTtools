function out_A_im = compute_A_im(freq)

omega = 2*pi*freq;
tau = 1./freq;
N_freq = numel(freq);

out_A_im = zeros(N_freq, N_freq+2);

for p = 1: N_freq
    for q = 1: N_freq
        
        if q ==1
            out_A_im(p, q+2) = 0.5*((omega(p)*tau(q)))/(1+(omega(p)*tau(q))^2)*log(tau(q+1)/tau(q));
        elseif q == N_freq
            out_A_im(p, q+2) = 0.5*((omega(p)*tau(q)))/(1+(omega(p)*tau(q))^2)*log(tau(q)/tau(q-1));
        else
            out_A_im(p, q+2) = 0.5*((omega(p)*tau(q)))/(1+(omega(p)*tau(q))^2)*log(tau(q+1)/tau(q-1));
        end
    end
end
        