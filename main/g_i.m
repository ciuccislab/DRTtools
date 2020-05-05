function out_val = g_i(freq_n, freq_m, epsilon, rbf_type)

alpha = 2*pi*freq_n/freq_m;

% choose among positive definite RBFs
% choose a function from a switch
switch rbf_type
    case 'Gaussian'
        rbf = @(x) exp(-(epsilon*x).^2);
    case 'C0 Matern'
        rbf = @(x) exp(-abs(epsilon*x));
    case 'C2 Matern'
        rbf = @(x) exp(-abs(epsilon*x)).*(1+abs(epsilon*x));
    case 'C4 Matern'
        rbf = @(x) 1/3*exp(-abs(epsilon*x)).*(3+3*abs(epsilon*x)+abs(epsilon*x).^2);
    case 'C6 Matern'
        rbf = @(x) 1/15*exp(-abs(epsilon*x)).*(15+15*abs(epsilon*x)+6*abs(epsilon*x).^2+abs(epsilon*x).^3);
    case 'Inverse quadratic'
        rbf = @(x) 1./(1+(epsilon*x).^2);
    case 'Inverse quadric'
        rbf = @(x) 1./sqrt(1+(epsilon*x).^2);
    case 'Cauchy'
        rbf = @(x) 1./(1+abs(epsilon*x));
    otherwise
        warning('Unexpected RBF input');
end
    % end of switch


    integrand_g_i = @(x) 1./(1+alpha^2*exp(2*x)).*rbf(x);

    out_val = integral(integrand_g_i, -Inf, Inf,'RelTol',1E-9,'AbsTol',1e-9);


end
