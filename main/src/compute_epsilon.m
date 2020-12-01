function out_val = compute_epsilon(freq, coeff, rbf_type, shape_control)
    
%   this function is used to compute epsilon, i.e., the shape factor of
%   the rbf used for discretization. user can directly set the shape factor
%   by selecting 'Shape Factor' for the shape_control. alternatively, 
%   when 'FWHM Coefficient' is selected, the shape factor is such that 
%   the full width half maximum (FWHM) of the rbf equals to the average 
%   relaxation time spacing in log space over coeff, i.e., FWHM = delta(ln tau)/coeff

    switch rbf_type
        case 'Gaussian'
            rbf_gaussian_4_FWHM = @(x) exp(-(x).^2)-1/2;
            FWHM_coeff = 2*fzero(@(x) rbf_gaussian_4_FWHM(x), 1);
        case 'C2 Matern'
            rbf_C2_matern_4_FWHM = @(x) exp(-abs(x)).*(1+abs(x))-1/2;
            FWHM_coeff = 2*fzero(@(x) rbf_C2_matern_4_FWHM(x), 1);
        case 'C4 Matern'
            rbf_C4_matern_4_FWHM = @(x) 1/3*exp(-abs(x)).*(3+3*abs(x)+abs(x).^2)-1/2;
            FWHM_coeff = 2*fzero(@(x) rbf_C4_matern_4_FWHM(x), 1);
        case 'C6 Matern'
            rbf_C6_matern_4_FWHM = @(x) 1/15*exp(-abs(x)).*(15+15*abs(x)+6*abs(x).^2+abs(x).^3)-1/2;
            FWHM_coeff = 2*fzero(@(x) rbf_C6_matern_4_FWHM(x), 1);
        case 'Inverse quadratic'
            rbf_inverse_quadratic_4_FWHM = @(x)  1./(1+(x).^2)-1/2;
            FWHM_coeff =  2*fzero(@(x) rbf_inverse_quadratic_4_FWHM(x), 1);
        case 'Inverse quadric'
            rbf_inverse_quadric_4_FWHM = @(x)  1./sqrt(1+(x).^2)-1/2;
            FWHM_coeff = 2*fzero(@(x) rbf_inverse_quadric_4_FWHM(x), 1);
        case 'Cauchy'
            rbf_cauchy_4_FWHM = @(x)  1./(1+abs(x))-1/2;
            FWHM_coeff = 2*fzero(@(x) rbf_cauchy_4_FWHM(x) ,1);
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