using Printf, ForwardDiff, Distributions, Random, LinearAlgebra, Flux, Plots

# Define True Distribution
# Define the means and standard deviations of the two Gaussian distributions
mu1, sigma1 = 1.0, 0.5
mu2, sigma2 = 5.0, 1.0

# Create the two Gaussian distributions
g1 = Normal(mu1, sigma1)
g2 = Normal(mu2, sigma2)

# Define the mixture weights (must sum to 1)
weights = [0.4, 0.6]

# Create the Gaussian mixture model
gmm = MixtureModel([g1, g2], weights)

# Sample from the GMM
samples = rand(gmm, 10)

# Plot the histogram of the samples and the PDF of the GMM
#histogram(samples, bins=50, normalize=true, alpha=0.5, label="Samples")
#plot!(x -> pdf(gmm, x), -2:0.1:10, lw=2, label="GMM PDF")

# Define prior
mu_pr = 3.0
sigma2_pr = 2.0
w_pr = 1.0

z0 = mu_pr

# p(x|mu,sigma)
function p_x(mu)
    prob = pdf(Normal(mu, 1), samples)
    prob = max(prob[1],1e-300)
    return prob
end
# p(mu,sigma)
function p_z(mu)
    prob = pdf(Normal(mu_pr, sigma2_pr), mu)
    return prob
end
# q(mu,sigma)
function q_z(mu, mu_post)
    prob = pdf(Normal(mu_post, 1), mu)
    return prob
end

function ELBO(z)
    res = 0
    #mu_vec = vec(z[1:length(mu_pr)])
    mu_vec = z
    #sigma2_vec = z[length(mu_pr)+1:length(z)]
    #sigma2_mx = [sigma2_vec[1] sigma2_vec[4] sigma2_vec[5]; sigma2_vec[4] sigma2_vec[2] sigma2_vec[6]; sigma2_vec[5] sigma2_vec[6] sigma2_vec[3]]
    N = 10
    mu_samp = rand(Normal(mu_vec, 1), N)
    #mean_beta = mean(beta)
    log_p_x = 0
    log_p_z = 0
    log_q_z = 0
    #for i = 1:size(mu_samp)
    #    mu = mu_samp[i]
    for mu in mu_samp
        log_p_x = log_p_x + log(p_x(mu))
        log_p_z = log_p_z + log(p_z(mu))
        log_q_z = log_q_z + log(q_z(mu, mu_vec))
    end
    res = (log_p_x + log_p_z - log_q_z)/length(mu_samp)
    return res
end
 
# define gradient of ELBO
function neg_ELBO(z)
    return -ELBO(z)
end

function G_ELBO(z) 
    #diff = ForwardDiff.gradient(neg_ELBO, z)
    diff = ForwardDiff.derivative(neg_ELBO, z)
    #diff  = Flux.gradient(neg_ELBO, z)[1]
    return diff
end

# Function to check positive semi-definiteness
#function is_positive_definite(C)
#    eigenvalues = eigen(C).values
#    return all(λ -> λ > 0, eigenvalues)
#end

# Adam Optimization
function adam_optimization(z0, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000, tol=0.01)
    z = z0
    #m = zeros(length(z0))
    #v = zeros(length(z0))
    m = 0
    v = 0
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

        z = z - lr * m_hat ./ (sqrt.(v_hat) .+ epsilon)

        #if is_positive_definite(sigma2_mx) == true
        #    z = z_try
        #    #display(z)
        #    lr = alpha
        #else
        #    lr = 0.5*lr
        #end
    end

    #Plot
    x = -2:0.1:10
    Plots.plot(x,pdf(Normal(z,1), x), lw=2, label="Approx. PDF")
    Plots.plot!(x,pdf(gmm, x), lw=2, label="GMM PDF")
    savefig("AdamGaussianMix.png")

    x_i = 1:length(ELBO_list)
    y_i = 1:length(log_g_list)
    p3 = Plots.plot(x_i*100, ELBO_list, xlabel = "iterates", ylabel = "ELBO", title = "ELBO with Time")
    p4 = Plots.plot(y_i, log_g_list, xlabel = "iterates", ylabel = "Log of Gradient of ELBO", title = "Log of Gradient of ELBO with Time")
    p4 = Plots.plot!(y_i, g_mean_list, label = "mean",linewidth = 2)
    Plots.plot(p3, p4, layout=(1, 2), size=(1000, 400))
    savefig("ELBO_adamGaussianMix.png")

    return z
end

adam_optimization(z0)