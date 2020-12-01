function out_SHD = SHD(mu_P, Sigma_P, mu_Q, Sigma_Q)

%   squared Hellinger distance

    sigma_P = sqrt(diag(Sigma_P));
    sigma_Q = sqrt(diag(Sigma_Q)); 
    sum_cov = sigma_P.^2+sigma_Q.^2;
    prod_cov = sigma_P.*sigma_Q;
    out_SHD = 1- sqrt(2*prod_cov./sum_cov).*exp(-0.25*(mu_P-mu_Q).^2./sum_cov);
 
end