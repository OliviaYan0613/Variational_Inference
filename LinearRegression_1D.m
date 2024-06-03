clear all
% Define the parameters
N = 100; 
tau = 1; 
max_iter = 100;
tol = 1e-3;

% Generate the true beta and sigma^2 values
sigma2_true = 1 ; % Sample sigma^2 from an improper Jeffreys' prior
beta_true = 0.6; %normrnd(0, sqrt(abs(sigma2_true)) * tau) % Sample beta from N(0, tau^2)

% Generate the predictor variable x
x = randn(N, 1); % Predictor variable

% Generate the response variable y
y = beta_true * x + sqrt(sigma2_true) * randn(N, 1); % Response variable

% Initialization
sigma2_2 = 1;
mu = sum(x .* y) / (sum(x.^2) + (1 / tau^2));

% Perform CAVI
for i = 1:max_iter
    % Calculate E_q(sigma_2)[1/sigma_2]
    v = 0.5 * (sum(y.^2) + (sum(x.^2) + (1 / tau^2)) * (sigma2_2 + mu^2) - 2 * sum(x .* y) * mu);
    % Update the parameters
    sigma2_2 = ((N + 1) / 2) / (v * (sum(x.^2) + (1 / tau^2)));
    ELBO = fun_ELBO(sigma2_2, mu, N, x, y, tau);
    if i > 1
        if abs(ELBO_old - ELBO)<tol
            break
        end
    end
    ELBO_old = ELBO;
end

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
    alpha_post = (N / 2) + 0.5;
    beta_post = 0.5 * sum((y - beta_current * x).^2);
    sigma2_current = 1 / gamrnd(alpha_post, 1 / beta_post);
    
    % Sample beta given sigma^2 and y
    var_beta_post = 1 / (sum(x.^2) / sigma2_current + 1 / (tau^2 * sigma2_current));
    mean_beta_post = var_beta_post * (sum(x .* y) / sigma2_current);
    beta_current = normrnd(mean_beta_post, sqrt(var_beta_post));
    
    % Store samples after burn-in
    if i > burn_in
        beta_samples(i - burn_in) = beta_current;
        sigma2_samples(i - burn_in) = sigma2_current;
    end
end

% Visualization
figure;

% Beta distribution comparison
subplot(2, 1, 1);
histogram(beta_samples, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'DisplayName', 'MCMC');
hold on;
x_vals = linspace(min(beta_samples), max(beta_samples), 100);
y_vals = normpdf(x_vals, mu, sqrt(sigma2_2));
plot(x_vals, y_vals, 'r-', 'LineWidth', 2, 'DisplayName', 'CAVI');
xlabel('\beta');
ylabel('Density');
title('Comparison of \beta Distributions');
legend;
grid on;

% Sigma^2 distribution comparison
subplot(2, 1, 2);
histogram(sigma2_samples, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'DisplayName', 'MCMC');
hold on;
x_vals = linspace(min(sigma2_samples), max(sigma2_samples), 100);
y_vals = gampdf(1 ./ x_vals, (N + 1) / 2, 1 / ((sum(y.^2) + (sum(x.^2) + (1 / tau^2)) * (sigma2_2 + mu^2) - 2 * sum(x .* y) * mu) / 2)) .* (1 ./ (x_vals.^2));
plot(x_vals, y_vals, 'r-', 'LineWidth', 2, 'DisplayName', 'CAVI');
xlabel('\sigma^2');
ylabel('Density');
title('Comparison of \sigma^2 Distributions');
legend;
grid on;

% Function to compute the ELBO
function [ELBO] = fun_ELBO(sigma2_2, mu, N, x, y, tau)
    num_samples = 1e6;
    alpha = (N + 1) / 2;
    beta = 0.5 * (sum(y.^2) + (sum(x.^2) + (1 / tau^2)) * (sigma2_2 + mu^2) - 2 * sum(x .* y) * mu);
    samples = 1 ./ gamrnd(alpha, 1 / beta, num_samples, 1);
    
    % Compute expectations
    E0 = mean(normpdf(samples, mu, sqrt(sigma2_2)));
    E1 = mean(log(samples));
    E2 = mean(alpha * log(beta) - gammaln(alpha) - (alpha + 1) * log(samples) - beta ./ samples);
    E3 = (N + 1) / (2 * beta);
    E4 = (N / 4) * log(2 * pi) * (sum(y.^2) - 2 * mu * sum(x .* y) + (sigma2_2 + mu^2) * sum(x.^2));
    
    % Compute ELBO
    ELBO = E4 + (0.5 * log(sigma2_2) - 0.5 * E1 - log(tau) - ((sigma2_2 + mu^2) * E3) / (2 * tau^2) + 0.5 + E0 - E2);
end
