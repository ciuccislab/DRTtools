function out_JSD_vec = JSD(mu_P, Sigma_P, mu_Q, Sigma_Q, N_MC_samples)

%   compute Jensen-Shannon distance (JSD) 
    
    out_JSD_vec = zeros(numel(mu_P),1);

    for index = 1:numel(mu_P)
        
        mean_P = mu_P(index);
        cov_P = Sigma_P(index, index);
        
        mean_Q = mu_Q(index);
        cov_Q = Sigma_Q(index, index);

        % draw sample from the MVN
        x = mvnrnd(mean_P, cov_P, N_MC_samples);
        p_x = normpdf(x,mean_P,sqrt(cov_P)); % sqrt used because normpdf takes std (instead of variance of scipy.stats.multivariate_normal)
        q_x = normpdf(x,mean_Q,sqrt(cov_Q));
        m_x = (p_x+q_x)/2;
    
        % draw sample from the MVN
        y = mvnrnd(mean_Q, cov_Q, N_MC_samples);
        p_y = normpdf(y,mean_P,sqrt(cov_P)); % sqrt used because normpdf takes std (instead of variance of scipy.stats.multivariate_normal)
        q_y = normpdf(y,mean_Q,sqrt(cov_Q));
        m_y = (p_y+q_y)/2;
    
        dKL_pm = mean(log(p_x./m_x));
        dKL_qm = mean(log(q_y./m_y));
    
        out_JSD_vec(index) = 0.5*(dKL_pm+dKL_qm);
    
    end
    
end