using Printf, ForwardDiff, Distributions, Random, LinearAlgebra, Plots, Flux

#Random.seed!(13);

# setup
beta_true =  3.0;
n = 10;
noise = 0.01;

x = randn(n,1);

#g1 = Normal(2.0, 1.0)
#g2 = Normal(3.0, 0.5)
#weights = [0.5, 0.5]
#x = rand(MixtureModel([g1, g2], weights), n)

#x1 = randn(n);
#x2 = x1 + 0.01*randn(n);
#x = [x1 x2]

y = x*beta_true + sqrt(noise)*randn(n);

p1 = Plots.scatter(x, y, xlabel = "x", ylabel = "y", title = "Data distribution")


# prior
mu_pr = [2.0 3.0]';
sigma2_pr = [1.0 1.0]';
w_pr = [0.5 0.5]'

# Start Point
z0 = [2.0, 1.0]

# p(y|beta)
function p_y(beta)
    prob = exp(-0.5*((y - x*beta)'*(y - x*beta))[1]/noise)/(2*pi*sqrt(noise))^n
    prob = max(prob[1],1e-300)
    return prob
end
# p(beta)
function p_b(beta)
    g1 = Normal(mu_pr[1], sigma2_pr[1])
    g2 = Normal(mu_pr[2], sigma2_pr[2])
    prob = w_pr[1]*pdf(g1,beta)+w_pr[2]*pdf(g2,beta)
    return prob
end
# q(beta)
function q_b(beta,mu,sigma2)
    prob = pdf(Normal(mu, sigma2), beta)
    return prob
end

function ELBO(z)
    res = 0
    mu = z[1]
    sigma2 = z[2]
    N = 10
    #beta = sampling(q_b, N, mu_vec, sigma2_vec);
    beta = rand(Normal(mu, sigma2), N)
    #mean_beta = mean(beta)
    log_p_y = 0
    log_p_b = 0
    log_q_b = 0
    for b in beta
        log_p_y = log_p_y + log(p_y(b)[1])
        log_p_b = log_p_b + log(p_b(b)[1])
        log_q_b = log_q_b + log(q_b(b,mu,sigma2)[1])
    end
    res = (log_p_y + log_p_b - log_q_b)/N
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

        if (z_try[2] > 0)
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
    mu_post = z[1];
    sigma2_post = z[2];

    # Create a grid of x values
    dx = range(beta_true-0.5, stop=(beta_true+0.5), length=200)

    # Evaluate the Gaussian density at each point in the grid
    Z = [pdf(Normal(mu_post, sigma2_post), xi) for xi in dx]

    p2 = Plots.plot(dx, Z, xlabel="beta_1", ylabel="probability density", title="1D Gaussian Distribution Contour Map", fill=false, colorbar=true)
    Plots.plot(p1, p2, layout=(1, 2), size=(1000, 400))
    savefig("Adam1DGMMP2Q1.png")

    x_i = 1:length(ELBO_list)
    p3 = Plots.plot(x_i, ELBO_list, xlabel = "iterates", ylabel = "ELBO", title = "ELBO with Time")
    p4 = Plots.plot(x_i, g_list, xlabel = "iterates", ylabel = "Log of Gradient of ELBO", title = "Log of Gradient of ELBO with Time")
    Plots.plot(p3, p4, layout=(1, 2), size=(1000, 400))
    savefig("ELBO_Adam1DGMMP2Q1.png")

    println("Reached maximum iterations")
    return z
end

adam_optimization(z0)