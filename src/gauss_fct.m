function gamma = gauss_fct(tau_vec, p_ref ,p)

gamma = zeros(size(tau_vec));
N_p = numel(p);

for i = 1:3:N_p
    R_0 = p_ref(i)*p(i);
    mu_log_tau = p_ref(i+1)*p(i+1);
    sigma = p_ref(i+2)*p(i+2);
    gamma = gamma + R_0 * exp(-(log(tau_vec) - mu_log_tau).^2./(2*sigma)^2);
end

gamma = gamma';

end