using Printf, ForwardDiff, Distributions, Random, LinearAlgebra, Flux, Plots

# Define True Distribution
# Define the means and standard deviations of the two Gaussian distributions
mu1, sigma1 = 1.0, 1.0
mu2, sigma2 = 7.0, 1.0

# Create the two Gaussian distributions
g1 = Normal(mu1, sigma1)
g2 = Normal(mu2, sigma2)

# Define the mixture weights (must sum to 1)
weights = [0.5, 0.5]

# Create the Gaussian mixture model
gmm = MixtureModel([g1, g2], weights)

# Sample from the GMM
n = 50
samples = rand(gmm, n)

# Define prior
mean_pr = 3.0
mu_sig_pr = 5.0
mu_pr = [mean_pr mu_sig_pr]'

alpha_pr = 2.0
beta_pr = 1.0
sigma2_pr = [alpha_pr beta_pr]'

w_pr = 1.0

#z0 = [mean_pr; sigma2_pr]
z0 = [mu_pr; sigma2_pr]
#z0 = [mu_pr sigma2_pr w_pr]

# p(x|mu,sigma)
function p_x(z)
    mu = z[1]
    sigma2 = z[2]
    prob = pdf(Normal(mu, sigma2), samples)
    prob = max(prob[1],1e-300)
    return prob
end
# p(mu,sigma)
function p_z(z)
    mu = z[1]
    sigma2 = z[2]
    prob = pdf(Normal(mu_pr[1], mu_pr[2]), mu)*pdf(InverseGamma(sigma2_pr[1], sigma2_pr[2]), sigma2)
    return prob
end
# q(mu,sigma)
function q_z(z, z_post)
    mu = z[1]
    sigma2 = z[2]
    mean_post = z_post[1]
    mu_sig_post = z_post[2]
    alpha_post = z_post[3]
    beta_post = z_post[4]
    prob = pdf(Normal(mean_post, mu_sig_post), mu)*pdf(InverseGamma(alpha_post, beta_post), sigma2)
    #prob = pdf(Normal(mean_post, 1), mu)*pdf(InverseGamma(alpha_pr, beta_pr), sigma2)
    return prob
end

function ELBO(z_post)
    res = 0
    mean = z_post[1]
    mu_sig = z_post[2]
    alpha = ForwardDiff.value(z_post[3])
    beta = ForwardDiff.value(z_post[4])
    
    N = 10
    mu_samp = rand(Normal(mean, mu_sig), N)
    sig_samp = rand(InverseGamma(alpha, beta), N)
    samp = [mu_samp sig_samp]

    log_p_x = 0
    log_p_z = 0
    log_q_z = 0
    
    for i = 1:size(samp)[2]
        s = [samp[i, 1] samp[i, 2]]'
        log_p_x = log_p_x + log(p_x(s))
        log_p_z = log_p_z + log(p_z(s))
        log_q_z = log_q_z + log(q_z(s, z_post))
    end
    res = (log_p_x + log_p_z - log_q_z)/N
    return res
end
 
# define gradient of ELBO
function neg_ELBO(z)
    return -ELBO(z)
end

function G_ELBO(z) 
    diff = ForwardDiff.gradient(neg_ELBO, z)
    #diff = ForwardDiff.derivative(neg_ELBO, z)
    #diff  = Flux.gradient(neg_ELBO, z)[1]
    return diff
end

# Adam Optimization
function adam_optimization(z0, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=5000, tol=0.01)
    z = z0
    m = zeros(length(z0))
    v = zeros(length(z0))
    t = 0
    lr = alpha
    ELBO_list = []
    log_g_list = []
    g_mean_list = []

    for i in 1:max_iter
        t += 1
        g = G_ELBO(z)
        push!(log_g_list,log(norm(g)))
        push!(g_mean_list,mean(log_g_list))

        if (i%10==0)
            push!(ELBO_list, ELBO(z))
        end
        
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g .^ 2)

        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)

        z_try = z - lr * m_hat ./ (sqrt.(v_hat) .+ epsilon)
        #display(z_try)

        if (z[2] >= 0) && (z[3] > 0) && (z[4] > 0)
            z = z_try
            lr = alpha
        else
            lr = 0.5*lr
        end
    end

    #Plot
    #x = -2:0.1:10
    #Plots.histogram(samples, bins=10, normalize=true, alpha=0.5, label="Samples")
    #Plots.plot!(x,pdf(Normal(z[1],1), x), lw=2, label="Approx. PDF")
    #Plots.plot!(x,pdf(gmm, x), lw=2, label="GMM PDF")
    #Plots.plot!(x,pdf(Normal(mean(samples), std(samples)), x), lw=2, label="Sample Distribution")
    #savefig("AdamGaussianMix_1Gapp2G.png")

    # Create a grid of x and y values
    dx = range(min(mu1,mu2)-1.0, stop=(max(mu1,mu2)+1.0), length=100)
    dy = range(0, stop=(max(sigma1,sigma2)+2.0), length=100)

    # Evaluate the Gaussian density at each point in the grid
    Z = [pdf(Normal(z[1], z[2]), xi)*pdf(InverseGamma(ForwardDiff.value(z[3]),ForwardDiff.value(z[4])),yi) for xi in dx, yi in dy]
    Z = reshape(Z, length(dx), length(dy))'

    Plots.contour(dx, dy, Z, xlabel="mu", ylabel="sigma2", title="2D Gaussian Distribution Contour Map", fill=false, colorbar=true, ratio = 1.0)
    savefig("AdamGaussianMix_1Gapp2G.png")

    x_i = 1:length(ELBO_list)
    y_i = 1:length(log_g_list)
    p3 = Plots.plot(x_i*100, ELBO_list, xlabel = "iterates", ylabel = "ELBO", title = "ELBO with Time")
    p4 = Plots.plot(y_i, log_g_list, xlabel = "iterates", ylabel = "Log of Gradient of ELBO", title = "Log of Gradient of ELBO with Time")
    #p4 = Plots.plot!(y_i, g_mean_list, label = "mean",linewidth = 2)
    Plots.plot(p3, p4, layout=(1, 2), size=(1000, 400))
    savefig("ELBO_AdamGaussianMix_1Gapp2G.png")

    return z
end

adam_optimization(z0)