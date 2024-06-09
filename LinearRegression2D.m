clc;
close all;
clear global;

% Generate synthetic data
rng(6); % For reproducibility
n = 1e4;
x = 100*randn(n,2);
%disp(x);
true_slope = [1;1];
%true_intercept = 0;
noise = 1;
rng(8);
y = x * true_slope + noise * randn(n, 1);

beta_pr_mu = [0; 0];
%beta_pr_sigma2 = [0.01 0; 0 0.01];
beta_pr_sigma2 = [0.1; 0.1]; % the first entry is sigma11 and the second entry is sigma22, sigma12 = sigma21 = 0

%beta_mu = [3;1];
%beta_sigma2 = [0.1;0.1];
%test = elbo(x, y, beta_mu, beta_sigma2, beta_pr_mu, beta_pr_sigma2, noise);

res = CAVI(x, y, beta_pr_mu, beta_pr_sigma2, noise);
disp(res);

%-----------------------------------------------------------------------------------------------
% Exact posterior
sig_pr = [0.1 0; 0 0.1];
mu_pr = [0;0];
sig_post = inv(inv(sig_pr) + x'*x/noise);
mu_post = (mu_pr'/sig_pr+y'*x/noise)/(inv(sig_pr)+x'*x/noise);

%------------------------------------------------------------------------------------------------------------

% Plot the results as histograms
% Define parameters of the Gaussian distribution
mu = res{1};
sig = [res{2}(1) 0; 0 res{2}(2)];

% Generate grid of x and y values
x = linspace(true_slope(1)-1e-2, true_slope(1)+1e-2, 100);
y = linspace(true_slope(2)-1e-2,true_slope(2)+1e-2, 100);
[X, Y] = meshgrid(x, y);
XY = [X(:), Y(:)];

% Calculate PDF values for each point on the grid
pdf_values_ext = mvnpdf(XY, mu_post, sig_post);
pdf_values_ELBO = mvnpdf(XY, mu, sig);
pdf_values_ext = reshape(pdf_values_ext, size(X));
pdf_values_ELBO = reshape(pdf_values_ELBO, size(X));

% Plot the Gaussian distribution as a contour plot
figure;
contour(X, Y, pdf_values_ext, 20, 'DisplayName', 'Exact'); % Adjust the number of contour levels as needed
hold on;
contour(X, Y, pdf_values_ELBO, 20, '--','DisplayName', 'VI'); 
title('2D Gaussian Distribution (Contour Plot)');
xlabel('\beta_1');
ylabel('\beta_2');
legend('show');
colorbar; % Add color bar for the contour levels

saveas(gcf, 'comparisonLR2D.png');
%----------------------------------------------------------------------------------

function ELBO = elbo(x, y, beta_mu, beta_sigma2, beta_pr_mu, beta_pr_sigma2, noise)
    x1 = x(:,1);
    x2 = x(:,2);
    
    E_log_py = -1/2/noise * (sum(y.^2)+ sum(x1.^2)*(beta_mu(1)^2+beta_sigma2(1))...
        +sum(x2.^2)*(beta_mu(2)^2+beta_sigma2(2))-2*sum(y.*x1)*beta_mu(1)...
        -2*sum(y.*x2)*beta_mu(2)+2*sum(x1.*x2)*beta_mu(1)*beta_mu(2));
    E_log_q = -1/2*(beta_sigma2(1)^2-beta_mu(1)^2*beta_sigma2(1)+beta_sigma2(2)^2-beta_mu(2)^2*beta_sigma2(2));
    E_log_pb = -1/2*(beta_pr_sigma2(1)*(beta_sigma2(1)+(beta_mu(1)-beta_pr_mu(1))^2)...
        +beta_pr_sigma2(2)*(beta_sigma2(2)+(beta_mu(2)-beta_pr_mu(2))^2));

    ELBO = E_log_py - E_log_q + E_log_pb;
end

%----------------------------------------------------------------------------------------------------------------------

function res = CAVI(x, y, beta_pr_mu, beta_pr_sigma2, noise)
    x1 = x(:,1);
    x2 = x(:,2);

    epsilon = 1e-5;

    %base case
    beta_mu_old = [0;0];
    beta_sigma2_old = [1;1];
    ELBO_old = 0;
    ELBO_new = elbo(x, y, beta_mu_old, beta_sigma2_old, beta_pr_mu, beta_pr_sigma2, noise);
 
    %beta_sigma2 (does not change during updating)
    beta_sigma2(1) = sum(x1.^2)/noise+beta_pr_sigma2(1);
    beta_sigma2(2) = sum(x2.^2)/noise+beta_pr_sigma2(2);

     for i = 1:1e10
         if (abs(ELBO_old - ELBO_new) >= epsilon)

            ELBO_old = ELBO_new;

            %update beta_mu
            beta_mu_new(1) = (sum(y.*x1)/noise-sum(x1.*x2)*beta_mu_old(1)+beta_pr_mu(1)*beta_pr_sigma2(1))...
                /beta_sigma2(1);
            beta_mu_new(2) = (sum(y.*x2)/noise-sum(x1.*x2)*beta_mu_old(2)+beta_pr_mu(2)*beta_pr_sigma2(2))...
                /beta_sigma2(2);
             
            %save beta_mu
            beta_mu_old(1) = beta_mu_new(1);
            beta_mu_old(2) = beta_mu_new(2);
             
            %updata ELBO
            ELBO_new = elbo(x, y, beta_mu_old, beta_sigma2_old, beta_pr_mu, beta_pr_sigma2, noise);
            %disp(ELBO_new);
         end
     end
     res = {beta_mu_new; 1./beta_sigma2; ELBO_new; i};
end