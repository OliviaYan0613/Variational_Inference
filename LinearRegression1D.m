clc;
close all;
clear global;

% Generate synthetic data
rng(6); % For reproducibility
n = 1e3;
%x = linspace(0, 100, n)';
x = 10*randn(n,1);
true_slope = 3;
%true_intercept = 0;
true_sigma = 1;
%y = true_slope * x + true_intercept + true_sigma * randn(n, 1);
rng(8);
y = true_slope * x + true_sigma * randn(n, 1);

tau2 = 0.25;

res = CAVI(x,y,tau2, 1e-4);
b_m = res(1);
b_sd = res(2);
sig_n = res(3);
disp(res);
disp(b_m);
disp(b_sd);
disp(sig_n);
%test = elbo(x,y,0.36,0.02,111, 0.25,0);

%---------------------------------------------------------------------------------------
% MCMC sampling for beta and sigma^2
num_samples = 1e4;
burn_in = 1000;
beta_samples = zeros(num_samples, 1);
sigma2_samples = zeros(num_samples, 1);

% Initial values for MCMC
beta_current = randn;
sigma2_current = 1;

for i = 1:(num_samples + burn_in)
    % Sample sigma^2 given beta and y
    alpha_post = (n / 2) + 0.5;
    beta_post = 0.5 * sum((y - beta_current * x).^2);
    sigma2_current = 1 / gamrnd(alpha_post, 1 / beta_post);
    
    % Sample beta given sigma^2 and y
    var_beta_post = 1 / (sum(x.^2) / sigma2_current + 1 / (tau2 * sigma2_current));
    mean_beta_post = var_beta_post * (sum(x .* y) / sigma2_current);
    beta_current = normrnd(mean_beta_post, sqrt(var_beta_post));
    
    % Store samples after burn-in
    if i > burn_in
        beta_samples(i - burn_in) = beta_current;
        sigma2_samples(i - burn_in) = sigma2_current;
    end
end

%----------------------------------------------------------------------------------
% Plot the results
figure;
subplot(2,1,1);
histogram(beta_samples, 'Normalization', 'pdf');
title('Sample of \beta');
xlabel('\beta');
ylabel('Density');
hold on;
beta_x = linspace(min(beta_samples), max(beta_samples), 100);
beta_y = normpdf(beta_x, mean(beta_samples), std(beta_samples));
beta_elbo = normpdf(beta_x, b_m, sqrt(b_sd));
plot(beta_x, beta_y, 'b', 'LineWidth', 2);
plot(beta_x, beta_elbo, 'r','LineWidth', 2);
legend('MCMC samples', 'Theoretical distribution','Distribution by ELBO');
hold off;

subplot(2,1,2);
histogram(sigma2_samples, 'Normalization', 'pdf');
title('Sample of \sigma');
xlabel('\sigma');
ylabel('Density');
hold on;
sigma2_x = linspace(min(sigma2_samples), max(sigma2_samples), 100);
sigma2_y = normpdf(sigma2_x, mean(sigma2_samples), std(sigma2_samples));
sigma2_elbo = gampdf(1./sigma2_x, (n+1)/2, 1/sig_n)./(sigma2_x.^2);
plot(sigma2_x, sigma2_y, 'b', 'LineWidth', 2);
plot(sigma2_x, sigma2_elbo, 'r', 'LineWidth', 2);
legend('MCMC samples', 'Theoretical distribution', 'Distribution by ELBO');
hold off;

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
    num_samples = 1e5;
    sigma2_samples = exprnd(1, num_samples, 1);
    sigma2_samples = max(sigma2_samples, 1e-6);
    sigma2_samples = min(sigma2_samples, 1e10);
    % Evaluate the integrand at the sampled points
    q_sig = nu^((n+1)/2)/gamma((n+1)/2).*sigma2_samples.^(-(n+1)/2-1).*exp(-nu./sigma2_samples);
    q_sig = min(max(q_sig, 1e-6), 1e6);

    E_q_sigma2_val = log(q_sig).*q_sig;
    E_q_sigma2_val = E_q_sigma2_val(~isnan(E_q_sigma2_val));
    %E_q_log_val = q_sig.*log(sigma2_samples)./sigma2_samples;
    %E_q_log_val = E_q_log_val(~isnan(E_q_log_val));
    %E_log_sigma2_val = log(sigma2_samples).*q_sig;
    %E_log_sigma2_val = E_log_sigma2_val(~isnan(E_log_sigma2_val));
    % Compute the average value of the integrand
    E_q_sigma2 = mean(E_q_sigma2_val);
    %disp(E_q_sigma2);
    %E_q_log = mean(E_q_log_val);
    %disp(E_q_log);
    %E_log_sigma2 = mean(E_log_sigma2_val);
    %disp(E_log_sigma2);

    E_q_log = integral(@(sigma2) gampdf(1./sigma2,(n+1)/2,1/nu).*log(sigma2)./sigma2,0,Inf);
    E_p_y = n*log(2*pi)/4 *(sum_y2 - 2*beta_mu*sum_xy + (beta_sd2 + beta_mu^2)*sum_x2)*E_q_log;
    %disp(E_p_y);
    E_log_sigma2 = integral(@(sigma2) log(sigma2).*gampdf(1./sigma2,(n+1)/2,1/nu),0,Inf);
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
    list_nu = 5;
    beta_sd2 = 1;
    nu = 5;
    ELBO = elbo(x,y, beta_mu, beta_sd2, nu, tau2, z);
    %disp(ELBO);
    list_ELBO = 0;

    beta_sd2_old = beta_sd2;
    ELBO_old = 0;
    ELBO_new = ELBO;

    for i = 1:1e6
        if (abs(ELBO_old - ELBO_new) >= epsilon)

            list_ELBO = [list_ELBO ELBO_new];
            ELBO_old = list_ELBO(i+1);
            %disp(list_ELBO);
            %disp(ELBO_old);

            %update beta_sd2 and nu
            E_qA = sum_y2 - 2 * sum_xy * beta_mu + (beta_sd2_old + beta_mu^2) *(sum_x2 + 1 / tau2);
            nu_new = 1 / 2 * E_qA;
            %disp(nu_new);
            beta_sd2_new = E_qA / (n + 1) / (sum_x2 + 1 / tau2);

            %update result lists
            list_beta_sd2 = [list_beta_sd2 beta_sd2_new];
            %disp(list_beta_sd2);
            list_nu = [list_nu nu_new];
            %disp(list_nu);

            %save beta_sd2
            beta_sd2_old = beta_sd2_new;
            
            %updata ELBO
            ELBO_new = elbo(x, y, beta_mu, beta_sd2_new ,nu_new, tau2, z);
            %disp(ELBO_new);
        end
    end
    %key = ["beta_mu", "beta_sd2", "nu", "ELBO"];
    %val = {beta_mu, list_beta_sd2, list_nu, list_ELBO};
    res = [beta_mu,list_beta_sd2(end),list_nu(end),list_ELBO(end)];
    %res = dictionary(key, val_rst);
end

%----------------------------------------------------------------------------------------

