using Printf, ForwardDiff, Distributions, Random, LinearAlgebra, Plots, Flux, Optimisers

# distribution: (x1,x2) ~ N(0,sigma2) z1=alpha*x1 z2=x2/alpha-beta(x1^2+alpha)
# take alpha=beta=1 sigma2=0.9

n = 50
alpha = 1
beta = 1

mu = [0,0]
sigma = [1.0 0.9; 0.9 1.0]

function trans(x)
    z=x
    z[2] = x[2]/alpha - beta*(x[1]^2+alpha)
    return z
end

function joint_pdf(z)
    x = z[1]
    y = z[2] + x^2 + 1
    return pdf(MvNormal(mu, sigma), [x, y])
end

# sampling
Sample_x = rand(MvNormal(mu,sigma),n)
Sample_z = zeros((2, n))
for i in 1:n
    Sample_z[:,i] = trans(Sample_x[:,i])
end

# joint pdf contour map after transformation
x_hat = LinRange(mu[1]-10, mu[1]+10, 500)
y_hat = LinRange(mu[2]-10,mu[2]+10, 500)
Z = zeros(length(x_hat),length(y_hat))
for i in 1:length(x_hat)
    for j in 1:length(y_hat)
        Z[i, j] = joint_pdf([x_hat[i], y_hat[j]])
    end
end
Z = (Z.-minimum(Z))./(maximum(Z)-minimum(Z))

# Start points
mu1_pr = [2,0]
mu2_pr = [2,0]
sigma1_pr = [5.0 1.0; 1.0 5.0]
sigma2_pr = [5.0 -1.0; -1.0 5.0]
sigma1_pr_diag = [sigma1_pr[1,1] sigma1_pr[2,2]]';
sigma1_pr_anti_diag = sigma1_pr[1,2]
sigma2_pr_diag = [sigma2_pr[1,1] sigma2_pr[2,2]]';
sigma2_pr_anti_diag = sigma2_pr[1,2]
w_pr = 0.5            # represent the weight for Gaussian 1, weight for Gaussian 2 = 1-w_pr
z0 = [mu1_pr;mu2_pr; sigma1_pr_diag; sigma1_pr_anti_diag;sigma2_pr_diag; sigma2_pr_anti_diag;w_pr];

# p(x) for each data point x
function log_p_x(x)
    prob = joint_pdf(x)
    prob1 = log(max(prob[1],1e-300))
    return prob1
end

# q_x for each data point x with parameters z
function log_q_x(x, z)
    mu1 = z[1:2]
    mu2 = z[3:4]
    sigma1 = [z[5] z[7];z[7] z[6]]
    sigma2 = [z[8] z[10];z[10] z[9]]
    w = z[11]
    #g1 = MvNormal(mu1,sigma1)
    #g2 = MvNormal(mu2,sigma2)
    prob = w*pdf(MvNormal(mu1,sigma1), x)+(1-w)*pdf(MvNormal(mu2,sigma2), x)
    prob1 = max(prob[1],1e-500)
    prob2 = log(prob1)
    return prob2
end
#----------------------------------------------------------------------------------------------------------------

# ELBO with parameters z
function ELBO(z)
    #z_val = zeros(length(z))
    #for v in 1:length(z)
    #    z_val[v] = ForwardDiff.value(z[v])
    #end
    z_val = z

    mu1 = [z_val[1], z_val[2]]
    mu2 = [z_val[3], z_val[4]]
    sigma1 = [z_val[5] z_val[7]; z_val[7] z_val[6]]
    sigma2 = [z_val[8] z_val[10]; z_val[10] z_val[9]]
    w = z_val[11]

    N = 20
    # draw samples from GMM
    c = rand(N)
    #samp = zeros((2,N))
    #samp = []
    samp_num1 = 0
    for i in 1:N
        if c[i]<= w
            samp_num1 +=1
        end
    end

    samp1 = rand(MvNormal(mu1,sigma1),samp_num1)
    samp2 = rand(MvNormal(mu2,sigma2),N-samp_num1)
    samp = hcat(samp1, samp2)
    val1 = map(x -> log_q_x(x,z), eachcol(samp))
    val2 = map(x -> log_p_x(x), eachcol(samp))

    #log_p_x = zeros(N)
    #log_q_x = zeros(N)
    #log_q_x = []
    #for k = 1:N
    #    b = samp[k]
        #log_p_x[k] = log(p_x(b)[1])
        #log_q_x[k] = log(q_x(b,z_val)[1])
    #    push!(log_q_x, log(q_x(b,z_val)[1]))
    #end 
    #res = mean(log_p_x)-mean(log_q_x)
    #res = -mean(log_q_x(si,z_val) for si in samp)
    return mean(val2)-mean(val1)
end

# define gradient of ELBO
function neg_ELBO(z)
    return -ELBO(z)
end

function G_ELBO(z) 
    #diff = ForwardDiff.gradient(neg_ELBO, z)
    #display(diff')
    #ps = Flux.params([mu1, mu2,sigma1, sigma2, w])
    diff = Flux.gradient(() -> neg_ELBO(z), Flux.params(z))
    return diff
end

#function AdamOpt(z)
#    # Training loop
#    epochs = 1000
#    for epoch in 1:epochs
        # Compute the gradient of the loss with respect to the model parameters
        #grads = Flux.gradient(Flux.params(model)) do
        #    loss(X, y)
        #end
        #grads = ForwardDiff.gradient(() -> neg_ELBO(z), Flux.params(ELBO))
#        grads = G_ELBO(z)
#        Optimisers.update(Optimisers.setup(opt, z), z, grads)
#        if epoch % 100 == 0
#            elbo = ELBO(z)
#            display(ELBO(z))
#            println("Epoch $epoch")
#        end
#    end
#    return z'
#end

#AdamOpt(z0)

function should_update(z_val)
    sigma1 = [z_val[5] z_val[7]; z_val[7] z_val[6]]
    sigma2 = [z_val[8] z_val[10]; z_val[10] z_val[9]]
    w = z_val[11]
    eigenvalues1 = eigen(sigma1).values
    eigenvalues2 = eigen(sigma2).values
    return all(eigenvalues1 .> 0) && all(eigenvalues2 .> 0) && (0<=w<=1)
end

function adam_optimization(z0, alpha=0.001, max_iter=1e4)
    z = z0
    m = zeros(length(z0))
    v = zeros(length(z0))
    t = 0
    lr = alpha
    ELBO_list = []
    g_list = []

    for i in 1:max_iter
        t += 1
        g = G_ELBO(z)[z]
        push!(g_list,log(norm(g)))
        push!(ELBO_list, ELBO(z))
        
        #m = beta1 * m + (1 - beta1) * g
        #v = beta2 * v + (1 - beta2) * (g .^ 2)

        #m_hat = m / (1 - beta1^t)
        #v_hat = v / (1 - beta2^t)

        #z_try = z - lr * m_hat ./ (sqrt.(v_hat) .+ epsilon)

        #if (z_try[5] > 0) &&(z_try[6] > 0) && (z_try[5]*z_try[6]-z_try[7]^2 >= 0)&&(z_try[8] > 0) &&(z_try[9] > 0) && (z_try[8]*z_try[9]-z_try[10]^2 >= 0)&&(0<=z_try[11]<=1)
        #    z = z_try
            #display(z)
        #    lr = alpha
        #else
        #    lr = 0.5*lr
        #end

        opt = Flux.ADAM(lr)
        z_pri = z
        Flux.Optimise.update!(opt, z, g)
        if should_update(z) == true
            lr = alpha
        else
            z = z_pri
            lr = 0.5*lr
        end

        #if norm(g) < tol
        #    println("Converged in $i iterations")
        #end

        if i%100 == 0
            display(i)
            display(z')
        end
    end

        # Plot
        z_val = z
        mu1 = [z_val[1], z_val[2]]
        mu2 = [z_val[3], z_val[4]]
        sigma1 = [z_val[5] z_val[7]; z_val[7] z_val[6]]
        sigma2 = [z_val[8] z_val[10]; z_val[10] z_val[9]]
        w = z_val[11]
        g1 = MvNormal(mu1, sigma1)
        g2 = MvNormal(mu2, sigma2)
        G = [(w*pdf(g1, [xi, yi])+(1-w)*pdf(g2, [xi, yi])) for xi in x_hat, yi in y_hat]
        G = (G.-minimum(G))./(maximum(G)-minimum(G))

        Plots.contour(x_hat, y_hat, Z', xlabel="z1", ylabel="z2", title="2D Gaussian Distribution Contour Map", fill=false, c=:reds, colorbar=false)
        Plots.contour!(x_hat, y_hat, G',c=:blues)
        Plots.scatter!(Sample_z[1,:]',Sample_z[2,:]',xlims=(-10,10), ylims=(-10, 10),label = false,color=:green)
        savefig("Banana2D.png")

        x_i = 1:length(ELBO_list)
        p3 = Plots.plot(x_i, ELBO_list, xlabel = "iterates", ylabel = "ELBO", title = "ELBO with Time")
        p4 = Plots.plot(x_i, g_list, xlabel = "iterates", ylabel = "Log of Gradient of ELBO", title = "Log of Gradient of ELBO with Time")
        Plots.plot(p3, p4, layout=(1, 2), size=(1000, 400))
        savefig("ELBO_Banana2D.png")

        return z'
end

adam_optimization(z0)