function out_val = compute_epsilon(freq, coeff, rbf_type, shape_control)

switch rbf_type

case 'Gaussian' % User selects Gaussian.
    rbf_gaussian_4_FWHM = @(x) exp(-(x).^2)-1/2;
    FWHM_coeff = 2*fzero(@(x) rbf_gaussian_4_FWHM(x), 1);
% end select gaussian

case 'C2 Matern' % User selects C2 Matern.
    rbf_C2_matern_4_FWHM = @(x) exp(-abs(x)).*(1+abs(x))-1/2;
    FWHM_coeff = 2*fzero(@(x) rbf_C2_matern_4_FWHM(x), 1);
% end select C2 Matern

case 'C4 Matern' % User selects C4 Matern.
    rbf_C4_matern_4_FWHM = @(x) 1/3*exp(-abs(x)).*(3+3*abs(x)+abs(x).^2)-1/2;
    FWHM_coeff = 2*fzero(@(x) rbf_C4_matern_4_FWHM(x), 1);
% end select C4 Matern

case 'C6 Matern' % User selects C6 Matern.
    rbf_C6_matern_4_FWHM = @(x) 1/15*exp(-abs(x)).*(15+15*abs(x)+6*abs(x).^2+abs(x).^3)-1/2;
    FWHM_coeff = 2*fzero(@(x) rbf_C6_matern_4_FWHM(x), 1);
% end select C6 Matern

case 'Inverse quadratic' % User selects inverse_quadratic.
    rbf_inverse_quadratic_4_FWHM = @(x)  1./(1+(x).^2)-1/2;
    FWHM_coeff =  2*fzero(@(x) rbf_inverse_quadratic_4_FWHM(x), 1);
% end select inverse_quadratic

case 'Inverse quadric' % User selects inverse_quadric.
    rbf_inverse_quadric_4_FWHM = @(x)  1./sqrt(1+(x).^2)-1/2;
    FWHM_coeff = 2*fzero(@(x) rbf_inverse_quadric_4_FWHM(x), 1);
% end select inverse_quadric

case 'Cauchy' % User selects cauchy.
    rbf_cauchy_4_FWHM = @(x)  1./(1+abs(x))-1/2;
    FWHM_coeff = 2*fzero(@(x) rbf_cauchy_4_FWHM(x) ,1);
% end select cauchy

case 'Piecewise linear'

    FWHM_coeff = 0 ;

end


switch shape_control

case 'FWHM Coefficient'

    delta = mean(diff(log(1./freq)));
    out_val  = coeff*FWHM_coeff/delta;

case 'Shape Factor'
    
    out_val = coeff;
        
end


end