using Printf, ForwardDiff, Distributions, Random, LinearAlgebra, GLMakie, Flux, Plots
#using CairoMakie

#Random.seed!(13);

# setup
beta_true =  [3.0 2.0 3.0]';
#beta_true = 3
n = 10;

mu_pr = [2.0 3.0 3.0]';
#mu_pr = 1
sigma2_pr = [2.0 0.5 0.1; 0.5 2.0 0.1; 0.1 0.1 2.0];
#sigma2_pr = 1
noise = 0.01;

sigma2_pr_diag = [sigma2_pr[1,1] sigma2_pr[2,2] sigma2_pr[3,3]]';
sigma2_pr_anti_diag = [sigma2_pr[1,2] sigma2_pr[1,3] sigma2_pr[2,3]]';
#sigma2_pr_diag = 1;
z0 = [mu_pr; sigma2_pr_diag; sigma2_pr_anti_diag];

x = randn(n,3);
#x = randn(n);
#Random.seed!(197)
y = x*beta_true + sqrt(noise)*randn(n);

# Theoredical posterior
sigma2_theo = inv(inv(sigma2_pr)+x'*x/noise)
sigma2_theo = round.(sigma2_theo, digits=15)
mu_theo = vec(((mu_pr'*inv(sigma2_pr)+y'*x/noise)*sigma2_theo)')
display(mu_theo')
display(sigma2_theo)

# prior
# p(y|beta)
function p_y(beta)
    #prob = exp(-0.5*(y - x*beta)'*(y - x*beta)/noise)/(2*pi*sqrt(noise))
    prob = exp(-0.5*((y - x*beta)'*(y - x*beta))[1]/noise)/(2*pi*sqrt(noise))^length(y)
    #prob = pdf(Normal(x*beta, noise), y)
    prob = max(prob[1],1e-300)
    return prob
end
# p(beta)
function p_b(beta)
    #prob = exp(-0.5*(beta - mu_pr)'*inv(sigma2_pr)*(beta - mu_pr))/sqrt((2*pi)^2*norm(sigma2_pr))
    #prob = exp(-0.5*(beta - mu_pr)'*inv(sigma2_pr)*(beta - mu_pr))
    prob = pdf(MvNormal(vec(mu_pr), sigma2_pr), beta)
    #prob = max(prob[1],1e-200)
    return prob
end
# q(beta)
function q_b(beta,mu,sigma2)
    sigma2_mx = [sigma2[1] sigma2[4] sigma2[5]; sigma2[4] sigma2[2] sigma2[6]; sigma2[5] sigma2[6] sigma2[3]]
    #prob = exp(- 0.5*(beta - mu)'*inv(sigma2_mx)*(beta - mu))/sqrt((2*pi)^2*norm(sigma2_mx))
    #prob = exp(- 0.5*(beta - mu)'*inv(sigma2_mx)*(beta - mu))
    prob = pdf(MvNormal(vec(mu), sigma2_mx), beta)
    #prob = max(prob[1],1e-200)
    return prob
end

function ELBO(z)
    res = 0
    mu_vec = vec(z[1:length(mu_pr)])
    sigma2_vec = z[length(mu_pr)+1:length(z)]
    sigma2_mx = [sigma2_vec[1] sigma2_vec[4] sigma2_vec[5]; sigma2_vec[4] sigma2_vec[2] sigma2_vec[6]; sigma2_vec[5] sigma2_vec[6] sigma2_vec[3]]
    N = 500
    #beta = sampling(q_b, N, mu_vec, sigma2_vec);
    beta = rand(MvNormal(mu_vec, sigma2_mx), N)
    #mean_beta = mean(beta)
    log_p_y = 0
    log_p_b = 0
    log_q_b = 0
    for i = 1:size(beta)[2]
        b = [beta[1, i] beta[2, i] beta[3, i]]'
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

# Function to check positive semi-definiteness
function is_positive_definite(C)
    eigenvalues = eigen(C).values
    return all(λ -> λ > 0, eigenvalues)
end

# Adam Optimization
function adam_optimization(z0, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=10000, tol=1.0)
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
        push!(g_list,norm(g))
        push!(ELBO_list, ELBO(z))
        
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g .^ 2)

        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)

        z_try = z - lr * m_hat ./ (sqrt.(v_hat) .+ epsilon)

        sigma2_vec = z_try[length(mu_pr)+1:length(z)]
        sigma2_mx = [sigma2_vec[1] sigma2_vec[4] sigma2_vec[5]; sigma2_vec[4] sigma2_vec[2] sigma2_vec[6]; sigma2_vec[5] sigma2_vec[6] sigma2_vec[3]]

        if is_positive_definite(sigma2_mx) == true
            z = z_try
            #display(z)
            lr = alpha
        else
            lr = 0.5*lr
        end
        
        if norm(g) < tol
            println("Converged in $i iterations")
            return z
        end
    end

    #Plot
     # Contour plot
    # Mean and covariance matrix for the Gaussian distribution
    mu_post = vec(z[1:length(mu_pr)]);
    sigma2_post = z[length(mu_pr)+1:length(z)];
    Sigma2 = diagm(sigma2_post[1:3]);
    Sigma2[1, 2] = Sigma2[2, 1] = sigma2_post[4];
    Sigma2[1, 3] = Sigma2[3, 1] = sigma2_post[5];
    Sigma2[2, 3] = Sigma2[3, 2] = sigma2_post[6];

    # Create a grid of x and y values
    dx = range(beta_true[1]-0.5, stop=(beta_true[1]+0.5), length=50)
    dy = range(beta_true[2]-0.5, stop=(beta_true[2]+0.5), length=50)
    dz = range(beta_true[3]-0.5, stop=(beta_true[3]+0.5), length=50)

    # Evaluate the Gaussian density at each point in the grid
    P = [pdf(MvNormal(mu_post, Sigma2), [xi, yi, zi]) for xi in dx, yi in dy, zi in dz]
    P_theo = [pdf(MvNormal(mu_theo, sigma2_theo), [xi, yi, zi]) for xi in dx, yi in dy, zi in dz]

    # Flatten the arrays for scatter plot
    x_scatter = repeat(dx, inner=length(dy)*length(dz))
    y_scatter = repeat(dy', inner=(length(dx), length(dz)))[:]
    z_scatter = repeat(dz, inner=(length(dx) * length(dy)))
    P_scatter = reshape(P, length(dx)*length(dy)*length(dz))
    P_theo_scatter = reshape(P_theo, length(dx)*length(dy)*length(dz))

    # creating 3D fig
    fig1 = Figure()
    ax1 = Axis3(fig1)
    GLMakie.scatter!(ax1, x_scatter, y_scatter, z_scatter, markersize=2, color=:blue)
    GLMakie.scatter!(ax1, x_scatter, y_scatter, z_scatter, color=P_scatter, markersize=2, colormap=:viridis)
    fig1[1, 1] = ax1
    save("./AdamOptim3D.png", fig1)

    fig2 = Figure()
    ax2 = Axis3(fig2)
    GLMakie.scatter!(ax2, x_scatter, y_scatter, z_scatter, markersize=2, color=:blue)
    GLMakie.scatter!(ax2, x_scatter, y_scatter, z_scatter, color=P_theo_scatter, markersize=2, colormap=:plasma)
    fig2[1, 1] = ax2
    save("./AdamOptim3DTheo.png", fig2)

    # create 2D for each two of beta
    # beta1 & beta2
    p12 = [pdf(MvNormal(mu_post, Sigma2), [xi, yi, beta_true[3]]) for xi in dx, yi in dy]
    p12 = reshape(p12, length(dx), length(dy))'
    p12_theo = [pdf(MvNormal(mu_theo, sigma2_theo), [xi, yi, beta_true[3]]) for xi in dx, yi in dy]
    p12_theo = reshape(p12_theo, length(dx), length(dy))'
    f1 = Plots.contour(dx, dy, p12, xlabel="beta_1", ylabel="beta_2", title="2D Gaussian Distribution Contour Map", fill=false, c=:blues, color=:blue, colorbar=true, ratio = 1.0)
    f2 = Plots.contour(dx, dy, p12_theo, xlabel="beta_1", ylabel="beta_2", title="2D Gaussian Distribution Contour Map", fill=false, c=:reds, color=:red, colorbar=true, ratio = 1.0)
    Plots.plot(f1, f2, layout=(1, 2), size=(1000, 400))
    savefig("AdamOptim_beta1&2.png")

    # beta1 & beta3
    p13 = [pdf(MvNormal(mu_post, Sigma2), [xi, beta_true[2], zi]) for xi in dx, zi in dz]
    p13 = reshape(p13, length(dx), length(dz))'
    p13_theo = [pdf(MvNormal(mu_theo, sigma2_theo), [xi, beta_true[2], zi]) for xi in dx, zi in dz]
    p13_theo = reshape(p13_theo, length(dx), length(dz))'
    f3 = Plots.contour(dx, dz, p13, xlabel="beta_1", ylabel="beta_3", title="2D Gaussian Distribution Contour Map", fill=false, c=:blues, color=:blue, colorbar=true, ratio = 1.0)
    f4 = Plots.contour(dx, dz, p13_theo, xlabel="beta_1", ylabel="beta_3", title="2D Gaussian Distribution Contour Map", fill=false, c=:reds, color=:red, colorbar=true, ratio = 1.0)
    Plots.plot(f3, f4, layout=(1, 2), size=(1000, 400))
    savefig("AdamOptim_beta1&3.png")

    # beta2 & beta3
    p23 = [pdf(MvNormal(mu_post, Sigma2), [beta_true[1], yi, zi]) for yi in dy, zi in dz]
    p23 = reshape(p13, length(dy), length(dz))'
    p23_theo = [pdf(MvNormal(mu_theo, sigma2_theo), [beta_true[1], yi, zi]) for yi in dy, zi in dz]
    p23_theo = reshape(p13_theo, length(dy), length(dz))'
    f5 = Plots.contour(dy, dz, p23, xlabel="beta_2", ylabel="beta_3", title="2D Gaussian Distribution Contour Map", fill=false, c=:blues, color=:blue, colorbar=true, ratio = 1.0)
    f6 = Plots.contour(dy, dz, p23_theo, xlabel="beta_2", ylabel="beta_3", title="2D Gaussian Distribution Contour Map", fill=false, c=:reds, color=:red, colorbar=true, ratio = 1.0)
    Plots.plot(f5, f6, layout=(1, 2), size=(1000, 400))
    savefig("AdamOptim_beta2&3.png")

    #combined_fig = CairoMakie.hstack(fig1, fig2)
    #save("./AdamOptim3D.png", combined_fig)

    x_i = 1:length(ELBO_list)
    p3 = Plots.plot(x_i, ELBO_list, xlabel = "iterates", ylabel = "ELBO", title = "ELBO with Time")
    p4 = Plots.plot(x_i, g_list, xlabel = "iterates", ylabel = "Gradient of ELBO", title = "Gradient of ELBO with Time")
    Plots.plot(p3, p4, layout=(1, 2), size=(1000, 400))
    savefig("ELBO_adam3D.png")

    println("Reached maximum iterations")
    display(z[1:3])
    mx = [z[4] z[7] z[8]; z[7] z[5] z[9]; z[8] z[9] z[6]]
    display(mx)
    #return z'
end

adam_optimization(z0)