function out_NMLL = NMLL_fct(theta, Z, A, M, N_freqs, N_taus)

    sigma_n = exp(theta(1));
    sigma_beta = exp(theta(2));
    sigma_lambda = exp(theta(3));

    W = 1/(sigma_beta^2)*eye(N_taus+1) + 1/(sigma_lambda^2)*M;
    W = 0.5*(W'+W); % making sure that W is positive definite
    
    K_agm = 1/(sigma_n^2)*(A'*A) + W; %invert of sigma
    K_agm = 0.5*(K_agm'+K_agm); % making sure that W is positive definite
    
    L_W = chol(W,'lower');
    L_agm = chol(K_agm,'lower');

%   compute mu_x
%   fast computation of the mean
    u1 = linsolve(L_agm, A'*Z);
    u = linsolve(L_agm', u1);
    mu_x = 1/(sigma_n^2)*u;

%   compute loss
    E_mu_x = 0.5/(sigma_n^2)*norm(A*mu_x-Z)^2 + 0.5*(mu_x'*(W*mu_x));

    val_1 = sum(log(diag(L_W)));
    val_2 = - sum(log(diag(L_agm))); 
    val_3 = - N_freqs/2.*log(sigma_n^2);
    val_4 = - E_mu_x;
    val_5 = - N_freqs/2*log(2*pi);

    out_NMLL = -(val_1+val_2+val_3+val_4+val_5);

end