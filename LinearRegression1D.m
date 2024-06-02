clc;
close all;
clear global;

% rng setup
[s1,s2] = RandStream.create('mlfg6331_64','NumStreams',2);

% Set up
n = 1e2;
sigma2 = 1;
beta = 0.3;

% Data Generate
x = randn(s1,1,n);
err = sqrt(sigma2).*randn(s2,1,n);
y = beta.*x + err;

res = CAVI(x,y,0.25, 1e-3);
test = elbo(x,y,0.36,0.02,5, 0.25,0);

%-----------------------------------------------------------------------------------
% Compute ELBO for example of 1D linear regression
% x: univariate input variable
% y: univariate output variable
% beta_mu: mean of variational density for beta
% beta_sd2: variance of variational density for beta
% tau2: variance of standardized effect size
% nu: parameter of variational density for sigma
% return ELBO
function ELBO = elbo(x, y, beta_mu, beta_sd2, nu, tau2, z)
    % setup
    n = length(y);
    sum_x2 = sum(x.^2);
    sum_y2 = sum(y.^2);
    sum_xy = sum(x.*y);

    %Monte Carlo Integration
    num_samples = 1e4;
    sigma2_samples = exprnd(1, num_samples, 1);
    sigma2_samples = max(sigma2_samples, 1e-6);
    sigma2_samples = min(sigma2_samples, 1e10);
    % Evaluate the integrand at the sampled points
    q_sig = nu^((n+1)/2)/gamma((n+1)/2).*sigma2_samples.^(-(n+1)/2-1).*exp(-nu./sigma2_samples);

    E_q_sigma2_val = log(q_sig).*q_sig;
    E_q_sigma2_val = E_q_sigma2_val(~isnan(E_q_sigma2_val));
    E_q_log_val = q_sig.*log(sigma2_samples)./sigma2_samples;
    E_q_log_val = E_q_log_val(~isnan(E_q_log_val));
    E_log_sigma2_val = log(sigma2_samples).*q_sig;
    E_log_sigma2_val = E_log_sigma2_val(~isnan(E_log_sigma2_val));
    % Compute the average value of the integrand
    E_q_sigma2 = mean(E_q_sigma2_val);
    E_q_log = mean(E_q_log_val);
    E_log_sigma2 = mean(E_log_sigma2_val);

    %E_q_log = integral(@(sigma2) gampdf(1./sigma2,(n+1)/2,1/nu).*log(sigma2)./sigma2,0,Inf);
    E_p_y = n*log(2*pi)/4*(sum_y2-2*beta_mu*sum_xy+(beta_sd2+beta_mu^2)*sum_x2)*E_q_log;
    %disp(E_p_y);
    %E_log_sigma2 = integral(@(sigma2) log(sigma2).*gampdf(1./sigma2,(n+1)/2,1/nu),0,Inf);
    %E_q_sigma2 = integral(@(sigma2) log(gampdf(1./sigma2,(n+1)/2,1/nu)).*gampdf(1./sigma2,(n+1)/2,1/nu),0,Inf);
    
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

    beta_mu = sum_xy / (sum_x2 + 1 / tau2);

    %base case
    z = 0;

    list_beta_sd2 = 1;
    list_nu = 3;
    beta_sd2 = 1;
    nu = 5;
    ELBO = elbo(x,y, beta_mu, beta_sd2, nu, tau2, z);
    disp(ELBO);
    list_ELBO = 0;

    beta_sd2_old = beta_sd2;
    ELBO_old = 0;
    ELBO_new = ELBO;

    while (abs(ELBO_old - ELBO_new) >= epsilon)
        %update beta_sd2 and nu
        E_qA = sum_y2 - 2 * sum_xy * beta_mu + (beta_sd2_old + beta_mu^2) *(sum_x2 + 1 / tau2);
        nu_new = 1 / 2 * E_qA;
        %disp(nu_new);
        beta_sd2_new = E_qA / (n + 1) / (sum_x2 + 1 / tau2);

        %updata ELBO
        list_ELBO = [list_ELBO ELBO_new];
        ELBO_old = ELBO_new;
        ELBO_new = elbo(x, y, beta_sd2_new ,nu_new, tau2, z);
        %disp(ELBO_new)

        %update result lists
        list_beta_sd2 = [list_beta_sd2 beta_sd2_new];
        list_nu = [list_nu nu_new];

        %save beta_sd2
        beta_sd2_old = beta_sd2_new;
    end
    key = ["beta_mu", "beta_sd2", "nu", "ELBO"];
    val = {beta_mu, list_beta_sd2, list_nu, list_ELBO};
    res = dictionary(key, val);
end

%----------------------------------------------------------------------------------------

