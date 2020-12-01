function out_dict = HT_single_est(theta, Z_exp, A, A_H, M, N_freqs, N_taus)

%   step 1 - identify the vector of hyperparameters
    options = optimset('Display','iter','TolX', 1E-10, 'TolFun', 1E-10,'Algorithm','quasi-newton');
    opt_theta = fminunc(@(c) NMLL_fct(c, Z_exp, A, M, N_freqs, N_taus), log(theta), options);
%     options = optimset('Display','iter','TolX', 1E-10, 'TolFun', 1E-10,'Algorithm','sqp');
%     lower = [-10,-10,-10];
%     upper = [10,10,10];
%     opt_theta = fmincon(@(c) NMLL_fct(c, Z_exp, A, M, N_freqs, N_taus), log(theta),[],[],[],[],lower,upper,[],options); 
%   collect the optimized theta'
    opt_theta = exp(opt_theta);
    sigma_n = opt_theta(1);
    sigma_beta = opt_theta(2);
    sigma_lambda = opt_theta(3);

%   step 2 - compute the pdf's of data regression
%   $K_agm = A.T A +\lambda L.T L$
    W = 1/(sigma_beta^2)*eye(N_taus+1) + 1/(sigma_lambda^2)*M;
    K_agm = 1/(sigma_n^2)*(A'*A) + W;

%   Cholesky factorization
    L_agm = chol(K_agm,'lower'); % the default matrix for matlab and python are npt the same
    inv_L_agm = inv(L_agm);
    inv_K_agm = inv_L_agm'*inv_L_agm;

%   compute the gamma ~ N(mu_gamma, Sigma_gamma)
    Sigma_gamma = inv_K_agm;
    mu_gamma = 1/(sigma_n^2)*(Sigma_gamma*A')*Z_exp;

%   compute Z ~ N(mu_Z, Sigma_Z) from gamma
    mu_Z = A*mu_gamma;
    Sigma_Z = A*(Sigma_gamma*A') + sigma_n^2*eye(N_freqs);
    
%   compute Z_DRT ~ N(mu_Z_DRT, Sigma_Z_DRT) from gamma
    A_DRT = A(:,2:end);
    mu_gamma_DRT = mu_gamma(2:end);
    Sigma_gamma_DRT = Sigma_gamma(2:end,2:end);
    mu_Z_DRT = A_DRT*mu_gamma_DRT;
    Sigma_Z_DRT = A_DRT*(Sigma_gamma_DRT*A_DRT');
    
%   compute Z_H_conj ~ N(mu_Z_H_conj, Sigma_Z_H_conj) from gamma   
    mu_Z_H = A_H*mu_gamma(2:end);
    Sigma_Z_H = A_H*(Sigma_gamma(2:end,2:end)*A_H');

    out_dict = struct('mu_gamma', mu_gamma,...
                      'Sigma_gamma', Sigma_gamma,...
                      'mu_Z', mu_Z,...
                      'Sigma_Z', Sigma_Z,...
                      'mu_Z_DRT', mu_Z_DRT,...
                      'Sigma_Z_DRT', Sigma_Z_DRT,...
                      'mu_Z_H', mu_Z_H,...
                      'Sigma_Z_H', Sigma_Z_H,...
                      'theta', opt_theta);

end