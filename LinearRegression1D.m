clc;
close all;
clear global;

%data
x = [];
y = [];

%-----------------------------------------------------------------------------------
% Compute ELBO for example of 1D linear regression
% x: univariate input variable
% y: univariate output variable
% beta_mu: mean of variational density for beta
% beta_sd2: variance of variational density for beta
% tau2: variance of standardized effect size
% nu: parameter of variational density for sigma
% return ELBO
function ELBO = elbo(x,y, beta_mu, beta_sd2, tau2, nu)
    % setup
    n = length(y);
    sum_x2 = sum(x^2);
    sum_y2 = sum(y^2);
    sum_xy = sum(x*y);

    gamma = integral(@(t) t^((n+1)/2)*exp(-t),0,Inf);
    q_sigma2 = @(sigma2) nu / gamma*sigma2^(-(n+1)/2-1)*exp(-1/sigma2*nu);

    E_q_log = integral(@(sigma2) q_sigma2*log(sigma2)/sigma2,-Inf,Inf);
    E_p_y = n*log(2*pi)/4*(sum_y2-2*beta_mu*sum_xy+(beta_sd2+beta_mu^2)*sum_x2)*E_q_log;

    E_log_sigma2 = integral(@(sigma2) log(sigma2)*q_sigma2,-Inf,Inf);
    E_q_sigma2 = integral(@(sigma2) log(q_sigma2)*q_sigma2,-Inf,Inf);
    %E_q_sigma2 = 1/N*sum()  %% If the one above does not work, try Monte Carlo integration
    E_inv_sigma2 = (n+1)/(sum_y2-2*beta_mu*sum_xy+(beta_sd2+beta_mu^2)*(sum_x2+1/tau2));

    ELBO = E_p_y -3/2* E_log_sigma2 - E_q_sigma2 - (beta_sd2+beta_mu^2)/(2*tau2)*E_inv_sigma2+ log(beta_sd2/tau2)/2+1/2;
end

%----------------------------------------------------------------------------------------
% CAVI
function res = CAVI(x, y, tau2, epsilon)
    %set up
    n = length(y);
    sum_x2 = sum(x.^2);
    sum_y2 = sum(y.^2);
    sum_xy = sum(x.*y);

    beta_mu = sum_yx / (sum_x2 + 1 / tau2);

    %base case
    list_beta_sd2 = beta_sd2;
    list_nu = nu;
    ELBO = elbo(x,y, beta_mu, beta_sd2, tau2, nu);
    list_ELBO = ELBO;

    %conditions of convergence
    i = 1;
    while (ELBO >= epsilon)
        %get previous data
        beta_sd2_old = list_beta_sd2(i);

        %update beta_sd2 and nu
        E_qA = sum_y2 - 2 * sum_xy * beta_mu + (beta_sd2_old^2 + beta_mu^2) *(sum_x2 + 1 / tau2);
        beta_sd2_new = sqrt(((n + 1) / E_qA) / (sum_x2 + 1 / tau2));
        nu_new = 1 / 2 * E_qA;

        %calculate new ELBO
        ELBO_new = elbo(x,y,beta_sd2_new,tau2,nu_new);

        %update result lists
        list_beta_sd2 = [list_beta_sd2 beta_sd2_new];
        list_nu = [list_nu nu_new];
        list_ELBO = [list_ELBO ELBO_new];

        %update index
        i = i+1;
    end
    res = ["beta_mu" beta_mu; "beta_sd2" list_beta_sd2; "nu" list_nu; "ELBO" list_ELBO];
end

%----------------------------------------------------------------------------------------