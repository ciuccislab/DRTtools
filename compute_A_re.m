function out_A_re = compute_A_re(freq)

omega = 2*pi*freq;
tau = 1./freq;
N_freq = numel(freq);

out_A_re = zeros(N_freq, N_freq+2);
out_A_re(:,2) = 1;

for p = 1: N_freq
    
    for q = 1: N_freq
        
        if q ==1
            out_A_re(p, q+2) = 0.5*(1.)/(1+(omega(p)*tau(q))^2)*log(tau(q+1)/tau(q));
        elseif q == N_freq
            out_A_re(p, q+2) = 0.5*(1.)/(1+(omega(p)*tau(q))^2)*log(tau(q)/tau(q-1));
        else
            out_A_re(p, q+2) = 0.5*(1.)/(1+(omega(p)*tau(q))^2)*log(tau(q+1)/tau(q-1));
        end
        
    end
    
end

        