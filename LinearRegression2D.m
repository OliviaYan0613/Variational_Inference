clc;
close all;
clear global;

% Generate synthetic data
rng(123); % For reproducibility
n = 10;
x = randn(n,2);
%disp(x);
%x1 = randn(n,1);
%rng(234)
%x2 = x1 + randn(n,1);
%x = [x1 x2];
true_slope = [2;3];
%true_intercept = 0;
noise = 0.1;
rng(345);
y = x * true_slope + sqrt(noise) * randn(n, 1);

beta_pr_mu = [1; 2];
beta_pr_sigma2_mx = inv([1 0; 0 1]);
beta_pr_sigma2 = [beta_pr_sigma2_mx(1,1); beta_pr_sigma2_mx(2,2)];
%beta_pr_sigma2 = [0.01; 0.01]; % the first entry is sigma11 and the second entry is sigma22, sigma12 = sigma21 = 0

%beta_mu = [3;1];
%beta_sigma2 = [0.1;0.1];
%test = elbo(x, y, beta_mu, beta_sigma2, beta_pr_mu, beta_pr_sigma2, noise);

res = CAVI(x, y, beta_pr_mu, beta_pr_sigma2, noise);
disp(res);

%-----------------------------------------------------------------------------------------------
% Exact posterior
%sig_pr = [beta_pr_sigma2(1) 0; 0 beta_pr_sigma2(2)];
sig_pr = beta_pr_sigma2_mx;
mu_pr = beta_pr_mu;
sig_post = inv(sig_pr + x'*x/noise);
mu_post = (mu_pr'/sig_pr+y'*x/noise)/(inv(sig_pr)+x'*x/noise);

%------------------------------------------------------------------------------------------------------------

% Plot the results as histograms
% Define parameters of the Gaussian distribution
mu = res{1};
sig = [res{2}(1) 0; 0 res{2}(2)];

% Generate grid of x and y values
x = linspace(true_slope(1)-3,true_slope(1)+3, 1000);
y = linspace(true_slope(2)-3, true_slope(2)+3, 1000);
[X, Y] = meshgrid(x, y);
XY = [X(:), Y(:)];

% Calculate PDF values for each point on the grid
pdf_values_ext = mvnpdf(XY, mu_post, sig_post);
pdf_values_ELBO = mvnpdf(XY, mu, sig);
pdf_values_ext = reshape(pdf_values_ext, size(X));
pdf_values_ELBO = reshape(pdf_values_ELBO, size(X));

% Plot the 2D normal distribution
%figure;
%surf(x, y, pdf_values_ext);
%hold on;
%surf(x, y, pdf_values_ELBO);
%xlabel('\beta_1');
%ylabel('\beta_2');
%zlabel('Probability Density');
%title('2D Normal Distribution');
%shading interp;  % Smooth the surface for better visualization
%colorbar;  % Add color bar to indicate the density values
%hold off;

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

%saveas(gcf, 'comparisonLR2D.png');
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
    beta_mu_old = [1;1];
    beta_sigma2_old = [0.1;0.1];
    ELBO_old = 0;
    ELBO_new = elbo(x, y, beta_mu_old, beta_sigma2_old, beta_pr_mu, beta_pr_sigma2, noise);
    %list_ELBO = [ELBO_old];

    %beta_sigma2 (does not change during updating)
    beta_sigma2(1) = sum(x1.^2)/noise+beta_pr_sigma2(1);
    beta_sigma2(2) = sum(x2.^2)/noise+beta_pr_sigma2(2);

    for i = 1:100
        %if (abs(ELBO_old - ELBO_new) >= epsilon)

        ELBO_old = ELBO_new;
        %list_ELBO = [list_ELBO ELBO_new];

        %update beta_mu
        beta_mu_new(1) = (sum(y.*x1)/noise-sum(x1.*x2)*beta_mu_old(2)/noise+beta_pr_mu(1)*beta_pr_sigma2(1))...
            /beta_sigma2(1);
        beta_mu_new(2) = (sum(y.*x2)/noise-sum(x1.*x2)*beta_mu_old(1)/noise+beta_pr_mu(2)*beta_pr_sigma2(2))...
            /beta_sigma2(2);
            
        %save beta_mu
        beta_mu_old(1) = beta_mu_new(1);
        beta_mu_old(2) = beta_mu_new(2);
            
        %updata ELBO
        ELBO_new = elbo(x, y, beta_mu_old, beta_sigma2_old, beta_pr_mu, beta_pr_sigma2, noise);
        %disp(ELBO_new);
        k = i;
        %end
    end

    %figure;
    %elbo_x = 1:length(list_ELBO);
    %plot(elbo_x,list_ELBO, 'b', 'LineWidth', 2);
    %title('ELBO');
    %xlabel('iterations');
    %hold on;

    res = {beta_mu_new; 1./beta_sigma2; ELBO_new; k};
end