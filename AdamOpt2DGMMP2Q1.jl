using Printf, ForwardDiff, Distributions, Random, LinearAlgebra, Plots, Flux

# setup
beta_true =  [3.0 2.0]';
#beta_true = 3
n = 10;

x = randn(n,2);
#x1 = randn(n);
#x2 = x1 + 0.01*randn(n);
#x = [x1 x2]
y = x*beta_true + sqrt(noise)*randn(n);

# Start Point
mu_pr = [2.0 3.0]';
sigma2_pr = [2.0 0.5; 0.5 2.0];
noise = 0.01;

sigma2_pr_diag = [sigma2_pr[1,1] sigma2_pr[2,2]]';
sigma2_pr_anti_diag = sigma2_pr[1,2]
z0 = [mu_pr; sigma2_pr_diag; sigma2_pr_anti_diag];

# prior
mu1 = [3.0 3.5]'
sigma2_1 = [1.0 0.5; 0.5 1.0];
mu1 = [2.0 3.0]'
sigma2_2 = [1.1 0.5; 0.5 1.5];
weight = [0.3 0.7]                  # must sum to 1

# p(y|beta)
function p_y(beta)
    prob = exp(-0.5*((y - x*beta)'*(y - x*beta))[1]/noise)/(2*pi*sqrt(noise))^length(y)
    prob = max(prob[1],1e-300)
    return prob
end
# p(beta)
function p_b(beta)
    g1 = MvNormal(mu1, sigma2_1)
    g2 = MvNormal(mu2, sigma2_2)
    prob = weight[1]*pdf(g1, beta)+weight[2]*pdf(g2, beta)
    return prob
end
# q(beta)
function q_b(beta,mu,sigma2)
    sigma2_mx = [sigma2[1] sigma2[3]; sigma2[3] sigma2[2]]
    prob = pdf(MvNormal(vec(mu), sigma2_mx), beta)
    return prob
end

function ELBO(z)
    res = 0
    mu_vec = vec(z[1:length(mu_pr)])
    sigma2_vec = z[length(mu_pr)+1:length(z)]
    sigma2_mx = [sigma2_vec[1] sigma2_vec[3]; sigma2_vec[3] sigma2_vec[2]]
    N = 10
    #beta = sampling(q_b, N, mu_vec, sigma2_vec);
    beta = rand(MvNormal(mu_vec, sigma2_mx), N)
    #mean_beta = mean(beta)
    log_p_y = 0
    log_p_b = 0
    log_q_b = 0
    for i = 1:size(beta)[2]
        b = [beta[1, i] beta[2, i]]'
        log_p_y = log_p_y + log(p_y(b)[1])
        log_p_b = log_p_b + log(p_b(b)[1])
        log_q_b = log_q_b + log(q_b(b,mu_vec,sigma2_vec)[1])
        #val = log(p_y(b)[1]*p_b(b)[1])-log(q_b(b,mu_vec,sigma2_vec)[1])
        #res = res + val/N
        #display(res)
    end
    res = (log_p_y + log_p_b - log_q_b)/length(beta)
    return res
end
 
# define gradient of ELBO
function neg_ELBO(z)
    return -ELBO(z)
end

function G_ELBO(z) 
    diff = ForwardDiff.gradient(neg_ELBO, z)
    #diff  = Flux.gradient(neg_ELBO, z)[1]
    return diff
end

# Adam Optimization
function adam_optimization(z0, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=5000, tol=0.1)
    z = z0
    m = zeros(length(z0))
    v = zeros(length(z0))
    t = 0
    lr = alpha
    ELBO_list = []
    g_list = []

    for i in 1:max_iter
        t += 1
        g = G_ELBO(z)
        push!(g_list,log(norm(g)))
        push!(ELBO_list, ELBO(z))
        
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g .^ 2)

        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)

        z_try = z - lr * m_hat ./ (sqrt.(v_hat) .+ epsilon)

        if (z_try[3] > 0) &&(z_try[4] > 0) && (z_try[3]*z_try[4]-z_try[5]^2 >= 0)
            z = z_try
            #display(z)
            lr = alpha
        else
            lr = 0.5*lr
        end
        
        if norm(g) < tol
            println("Converged in $i iterations")
        end
    end

    #Plot
     # Contour plot
    # Mean and covariance matrix for the Gaussian distribution
    mu_post = vec(z[1:length(mu_pr)]);
    sigma2_post = z[length(mu_pr)+1:length(z)];
    Sigma2 = diagm(sigma2_post[1:2]);
    Sigma2[1, 2] = sigma2_post[3];
    Sigma2[2, 1] = sigma2_post[3]
    # Create a grid of x and y values
    dx = range(beta_true[1]-0.5, stop=(beta_true[1]+0.5), length=100)
    dy = range(beta_true[2]-0.5, stop=(beta_true[2]+0.5), length=100)

    # Evaluate the Gaussian density at each point in the grid
    Z = [pdf(MvNormal(mu_post, Sigma2), [xi, yi]) for xi in dx, yi in dy]
    Z = reshape(Z, length(dx), length(dy))'

    Z_theo = [pdf(MvNormal(mu_theo, sigma2_theo), [xi, yi]) for xi in dx, yi in dy]
    Z_theo = reshape(Z_theo, length(dx), length(dy))'

    p1 = Plots.contour(dx, dy, Z, xlabel="beta_1", ylabel="beta_2", title="2D Gaussian Distribution Contour Map", fill=false, c=:blues, color=:blue, colorbar=true, ratio = 1.0)
    #savefig("AdamOpt.png")
    p2 = Plots.contour(dx, dy, Z_theo, xlabel="beta_1", ylabel="beta_2", title="2D Gaussian Distribution Contour Map", fill=false, c=:reds, color=:red, colorbar=true, ratio = 1.0)
    #savefig("Theoredical_GD.png")
    Plots.plot(p1, p2, layout=(1, 2), size=(1000, 400))
    savefig("AdamOptim.png")

    x_i = 1:length(ELBO_list)
    p3 = Plots.plot(x_i, ELBO_list, xlabel = "iterates", ylabel = "ELBO", title = "ELBO with Time")
    p4 = Plots.plot(x_i, g_list, xlabel = "iterates", ylabel = "Log of Gradient of ELBO", title = "Log of Gradient of ELBO with Time")
    Plots.plot(p3, p4, layout=(1, 2), size=(1000, 400))
    savefig("ELBO_adam.png")

    println("Reached maximum iterations")
    return z
end

adam_optimization(z0)