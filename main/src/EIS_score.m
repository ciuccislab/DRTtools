function out_scores = EIS_score(theta_0, freq_vec, Z_exp, out_dict_real, out_dict_imag, N_MC_samples)

%   this function compute the scores that quantify the quality of EIS data
   
%   we need the means and covariances for the score computation. 
%   we first extract the essential components from out_dict.
%   out_dict is a matlab structure.
    mu_Z_DRT_re = out_dict_real.mu_Z_DRT;
    mu_Z_DRT_im = out_dict_imag.mu_Z_DRT;
    mu_Z_H_re = out_dict_imag.mu_Z_H;
    mu_Z_H_im = out_dict_real.mu_Z_H;

    Sigma_Z_DRT_re = out_dict_real.Sigma_Z_DRT;
    Sigma_Z_DRT_im = out_dict_imag.Sigma_Z_DRT;
    Sigma_Z_H_re = out_dict_imag.Sigma_Z_H;
    Sigma_Z_H_im = out_dict_real.Sigma_Z_H;
    
%   1) s_mu - distance between means:
    discrepancy_re = norm(mu_Z_DRT_re-mu_Z_H_re)/(norm(mu_Z_DRT_re)+norm(mu_Z_H_re));
    s_mu_re = 1 - discrepancy_re;
    discrepancy_im = norm(mu_Z_DRT_im-mu_Z_H_im)/(norm(mu_Z_DRT_im)+norm(mu_Z_H_im));
    s_mu_im = 1 - discrepancy_im;
%   end s_mu computation
    
%   2) s_res - residual score:
%   real part
%   retrieve distribution of R_inf
    mu_R_inf = out_dict_real.mu_gamma(1);
    cov_R_inf = diag(out_dict_real.Sigma_gamma);% double check this line.
    cov_R_inf = cov_R_inf(1); 
%   we will also need omega an estimate of the error
    sigma_n_im = out_dict_imag.theta(1);

%   R_inf+Z_H_re-Z_exp has
%   mean:
    res_re = mu_R_inf + mu_Z_H_re - real(Z_exp);
%   std:
    band_re = sqrt(cov_R_inf + diag(Sigma_Z_H_re)+sigma_n_im.^2); %# this calculate the std of R+ Z_H_re +sigma_n_im^2<--shouldn't it be the real part instead? 
    s_res_re = res_score(res_re, band_re);
     
%   imaginary part
%   retrieve distribution of L_0     
    mu_L_0 = out_dict_imag.mu_gamma(1);
    cov_L_0 = diag(out_dict_imag.Sigma_gamma);% double check this line.
    cov_L_0 = cov_L_0(1);

%   we will also need omega
    omega_vec = 2*pi*freq_vec;
%   and an estimate of the error
    sigma_n_re = out_dict_real.theta(1);

%   R_inf+Z_H_re-Z_exp has
%   mean:
    res_im = omega_vec*mu_L_0 + mu_Z_H_im - imag(Z_exp);
%   std:
    band_im = sqrt((omega_vec.^2)*cov_L_0 + diag(Sigma_Z_H_im)+sigma_n_re.^2);
    s_res_im = res_score(res_im, band_im);
%   end s_res computation

%   3) s_HD - Squared Hellinger distance (SHD):
%   which is bounded between 0 and 1
    SHD_re = SHD(mu_Z_DRT_re, Sigma_Z_DRT_re, mu_Z_H_re, Sigma_Z_H_re);
    SHD_im = SHD(mu_Z_DRT_im, Sigma_Z_DRT_im, mu_Z_H_im, Sigma_Z_H_im);

%   we are going to score w.r.t. the Hellinger distance (HD)
%   the score uses 1 to mean good (this means close)
%   and 0 means bad (far away) => that's the opposite of the distance
    s_HD_re = 1 - mean(sqrt(SHD_re));
    s_HD_im = 1 - mean(sqrt(SHD_im));
%   end s_HD computation    

%   4) s_JSD - Jensen-Shannon Distance (JSD):
    JSD_re = JSD(mu_Z_DRT_re, Sigma_Z_DRT_re, mu_Z_H_re, Sigma_Z_H_re, N_MC_samples);
    JSD_im = JSD(mu_Z_DRT_im, Sigma_Z_DRT_im, mu_Z_H_im, Sigma_Z_H_im, N_MC_samples);

%   the JSD is a symmetrized relative entropy (discrepancy), so highest value means more entropy
%   we are going to reverse that by taking (log(2)-JSD)/log(2)
%   which means higher value less relative entropy (discrepancy)
    s_JSD_re = (log(2)-mean(JSD_re))/log(2);
    s_JSD_im = (log(2)-mean(JSD_im))/log(2);
%   end s_JSD computation
    
    out_scores = struct('s_res_re', s_res_re,...
                        's_res_im', s_res_im,...
                        's_mu_re', s_mu_re,...
                        's_mu_im', s_mu_im,...
                        's_HD_re', s_HD_re,...
                        's_HD_im', s_HD_im,...
                        's_JSD_re', s_JSD_re,...
                        's_JSD_im', s_JSD_im);
    
end
    
    
    