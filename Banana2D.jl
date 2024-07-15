using Printf, ForwardDiff, Distributions, Random, LinearAlgebra, Plots, Flux, Optimisers

# distribution: (x1,x2) ~ N(0,sigma2) z1=alpha*x1 z2=x2/alpha-beta(x1^2+alpha)
# take alpha=beta=1 sigma2=0.9

n = 50
alpha = 1
beta = 1

mu = [0,0]
sigma2 = [1.0 0.9; 0.9 1.0]

function trans(x)
    z=x
    z[2] = x[2]/alpha - beta*(x[1]^2+alpha)
    return z
end

function joint_pdf(z)
    x = z[1]
    y = z[2] + x^2 + 1
    return pdf(MvNormal(mu, sigma2), [x, y])
end

# sampling
Sample_x = rand(MvNormal(mu,sigma2),n)
Sample_z = zeros((2, n))
for i in 1:n
    Sample_z[:,i] = trans(Sample_x[:,i])
end

# joint pdf contour map after transformation
x_hat = LinRange(mu[1]-4, mu[1]+4, 100)
y_hat = LinRange(mu[2]-11,mu[2]+2, 100)
Z = zeros(length(x_hat),length(y_hat))
for i in 1:length(x_hat)
    for j in 1:length(y_hat)
        Z[i, j] = joint_pdf([x_hat[i], y_hat[j]])
    end
end

p1 = Plots.contour(x_hat, y_hat, Z', xlabel="z1", ylabel="z2", title="2D Gaussian Distribution Contour Map", fill=false, c=:reds, colorbar=true)
p2 = Plots.scatter(Sample_z[1,:]',Sample_z[2,:]',xlims=(-4,4), ylims=(-11, 2),label = false,color=:blue)


# Start points
mu1_pr = [0,-12]
mu2_pr = [12, 0]
sigma1_pr = [20.0 0.5; 0.5 20.0]
sigma2_pr = [20.0 0.5; 0.5 20.0]
sigma1_pr_diag = [sigma1_pr[1,1] sigma1_pr[2,2]]';
sigma1_pr_anti_diag = sigma1_pr[1,2]
sigma2_pr_diag = [sigma2_pr[1,1] sigma2_pr[2,2]]';
sigma2_pr_anti_diag = sigma2_pr[1,2]
w_pr = 0.5            # represent the weight for Gaussian 1, weight for Gaussian 2 = 1-w_pr
z0 = [mu1_pr;mu2_pr; sigma1_pr_diag; sigma1_pr_anti_diag;sigma2_pr_diag; sigma2_pr_anti_diag;w_pr];

# p(x) for each data point x
function p_x(x)
    prob = joint_pdf(x)
    prob = max(prob[1],1e-300)
    return prob
end

# q_x for each data point x with parameters z
function q_x(x, z)
    mu1 = z[1:2]
    mu2 = z[3:4]
    sigma1 = [z[5] z[7];z[7] z[6]]
    sigma2 = [z[8] z[10];z[10] z[9]]
    w = z[11]
    g1 = MvNormal(mu1,sigma1)
    g2 = MvNormal(mu2,sigma2)
    prob = w*pdf(g1, x)+(1-w)*pdf(g2, x)
    return prob
end

# ELBO with parameters z
function ELBO(z)
    res = 0

    z_val = zeros(length(z))
    for v in 1:length(z)
        z_val[v] = ForwardDiff.value(z[v])
    end

    mu1 = [z_val[1], z_val[2]]
    mu2 = [z_val[3], z_val[4]]
    sigma1 = [z_val[5] z_val[7]; z_val[7] z_val[6]]
    sigma2 = [z_val[8] z_val[10]; z_val[10] z_val[9]]
    w = z_val[11]

    N = 50
    # draw samples from GMM
    c = rand(N)
    samp = zeros((2,N))
    for i in 1:N
        if c[i]<= w
            samp[:,i] = rand(MvNormal(mu1, sigma1)) 
        else
            samp[:,i] = rand(MvNormal(mu2, sigma2))
        end
    end
    #display(samp)
    log_p_x = zeros(N)
    log_q_x = zeros(N)
    for k = 1:N
        b = [samp[1, k] samp[2, k]]'
        log_p_x[k] = log(p_x(b)[1])
        log_q_x[k] = log(q_x(b,z_val)[1])
    end
    res = mean(log_p_x)-mean(log_q_x)
    return res
end

# define gradient of ELBO
function neg_ELBO(z)
    return -ELBO(z)
end

function G_ELBO(z) 
    diff = ForwardDiff.gradient(neg_ELBO, z)
    #ps = Flux.params([mu1, mu2,sigma1, sigma2, w])
    #diff = Flux.gradient(() -> neg_ELBO(mu1, mu2,sigma1, sigma2, w), ps)
    return diff
end

opt = Optimisers.Adam()

function AdamOpt(z)
    # Training loop
    epochs = 1000
    for epoch in 1:epochs
        # Compute the gradient of the loss with respect to the model parameters
        #grads = Flux.gradient(Flux.params(model)) do
        #    loss(X, y)
        #end
        #grads = ForwardDiff.gradient(() -> neg_ELBO(z), Flux.params(ELBO))
        grads = G_ELBO(z)
        Optimisers.update(Optimisers.setup(opt, z), z, grads)
        if epoch % 100 == 0
            elbo = ELBO(z)
            display(ELBO(z))
            println("Epoch $epoch")
        end
    end
    return z'
end

AdamOpt(z0)

#Plots.plot(p1, p2, layout=(1, 2), size=(1000, 400))