function [out_gamma, freq_fine] = map_array_to_gamma(freq_map, freq_coll, x, epsilon, rbf_type)

% choose among positive definite RBFs
% rbf refer to the phi function. On multiply phi function with x, the
% magnitude at the specific frequency, one can give the gamma profile
% choose a function from a switch

switch rbf_type
    case 'Gaussian'
        rbf = @(y, y0) exp(-(epsilon*(y-y0)).^2);
    case 'C0 Matern'
        rbf = @(y, y0)  exp(-abs(epsilon*(y-y0)));
    case 'C2 Matern'
        rbf = @(y, y0)  exp(-abs(epsilon*(y-y0))).*(1+abs(epsilon*(y-y0)));
    case 'C4 Matern'
        rbf = @(y, y0)  1/3*exp(-abs(epsilon*(y-y0))).*(3+3*abs(epsilon*(y-y0))+abs(epsilon*(y-y0)).^2);
    case 'C6 Matern'
        rbf = @(y, y0)  1/15*exp(-abs(epsilon*(y-y0))).*(15+15*abs(epsilon*(y-y0))+6*abs(epsilon*(y-y0)).^2+abs(epsilon*(y-y0)).^3);
    case 'Inverse quadratic'
        rbf = @(y, y0)  1./(1+(epsilon*(y-y0)).^2);
    case 'Inverse quadric'
        rbf = @(y, y0)  1./sqrt(1+(epsilon*(y-y0)).^2);
    case 'Cauchy'
        rbf = @(y, y0)  1./(1+abs(epsilon*(y-y0))); 
    case 'Piecewise linear'
        out_gamma = x;
        freq_fine = freq_coll;
        return
        
    otherwise
        warning('Unexpected RBF input');
end
% end of switch

freq_fine = freq_map;

y0 = -log(freq_coll);
out_gamma = zeros(size(freq_map))';

for iter_freq_map = 1: numel(freq_map)

    freq_map_loc = freq_map(iter_freq_map);
    y = -log(freq_map_loc);
    out_gamma(iter_freq_map) = x'*rbf(y, y0);

end

end
