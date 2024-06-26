using Printf, ForwardDiff, Distributions, Random, LinearAlgebra, Plots, Flux

#Random.seed!(13);

# setup
beta_true =  [4.0 2.0]';
#beta_true = 3
n = 10;

MaxIter = 5000;
#tol = 1e-4; 

mu_pr = [2.0 3.0]';
#mu_pr = 1
sigma2_pr = [2.0 1.0; 1.0 2.0];
#sigma2_pr = 1
noise = 0.01;

sigma2_pr_diag = [sigma2_pr[1,1] sigma2_pr[2,2]]';
sigma2_pr_anti_diag = sigma2_pr[1,2]
#sigma2_pr_diag = 1;
z0 = [mu_pr; sigma2_pr_diag; sigma2_pr_anti_diag];

x = randn(n,2);
#x = randn(n);
#Random.seed!(197)
y = x*beta_true + sqrt(noise)*randn(n);

# Theoredical posterior
sigma2_theo = inv(inv(sigma2_pr)+x'*x/noise)
mu_theo = vec(((mu_pr'*inv(sigma2_pr)+y'*x/noise)*sigma2_theo)')

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
    sigma2_mx = [sigma2[1] sigma2[3]; sigma2[3] sigma2[2]]
    #prob = exp(- 0.5*(beta - mu)'*inv(sigma2_mx)*(beta - mu))/sqrt((2*pi)^2*norm(sigma2_mx))
    #prob = exp(- 0.5*(beta - mu)'*inv(sigma2_mx)*(beta - mu))
    prob = pdf(MvNormal(vec(mu), sigma2_mx), beta)
    #prob = max(prob[1],1e-200)
    return prob
end

# define loss function
# MC sampling
#function sampling(pdf, N, mu, sigma2; b_min=-10, b_max=10)
#    samples = []
#    while length(samples) < N
#        #Random.seed!(873)
#        b1 = rand() * (b_max - b_min) + b_min        
#        #Random.seed!(745)
#        b2 = rand() * (b_max - b_min) + b_min
#        #Random.seed!(134)
#        b = [b1; b2]
#        y = rand()        # Sample y uniformly within [0, 1]
#        if y <= pdf(b, mu, sigma2)[1]
#            push!(samples, b)
#        end
#    end
#    return samples
#end

function ELBO(z)
    res = 0
    mu_vec = vec(z[1:length(mu_pr)])
    sigma2_vec = z[length(mu_pr)+1:length(z)]
    sigma2_mx = [sigma2_vec[1] sigma2_vec[3]; sigma2_vec[3] sigma2_vec[2]]
    N = 100
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

# Gradient descent method
function SteepestDescent(z0,alpha)
    # setup for steepest descent
    z = z0;
    lr = alpha;
    grad_list = [];
    mean_grad = [];
    ELBO_list = [];
    # perform steepest descent iterations
    for iter = 1:MaxIter
        Fval = ELBO(z);
        Fgrad = G_ELBO(z);
        push!(grad_list,norm(Fgrad));
        push!(mean_grad,mean(grad_list));
        push!(ELBO_list, Fval);
        
        # perform steepest descent step
        z_try = zeros(3)
        for k = length(mu_pr)+1:length(z)
            z_try[k-2] = z[k] - 0.75*alpha*Fgrad[k]
        end
        if (z_try[1] > 0) &&(z_try[2] > 0) && (z_try[1]*z_try[2]-z_try[3]^2 >= 0)
            z[length(mu_pr)+1:length(z)] = z_try
            z[1:length(mu_pr)] = z[1:length(mu_pr)] - lr*Fgrad[1:length(mu_pr)]
            lr = alpha
        else
            lr = 0.5*lr
        end
        #z[k] = max(z_try,1e-6)
        #Fval_old = Fval;
 
        # print how we're doing, every 10 iterations
        if (iter%100==0)
          #@printf("iter: %d, alpha: %f, %f\t, %f\t, %f\n", iter, alpha, z[1:length(mu)], z[length(mu)+1:length(z)], Fval[1])
          @printf("iter: %d, alpha: %f, %f\n", iter, lr, Fval[1])
          display(z')
        end
 
    end
    # plot
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

        p1 = contour(dx, dy, Z, xlabel="beta_1", ylabel="beta_2", title="2D Gaussian Distribution Contour Map", fill=false, c=:blues, color=:blue, colorbar=true, ratio = 1.0)
        savefig("GD.png")
        p2 = contour(dx, dy, Z_theo, xlabel="beta_1", ylabel="beta_2", title="2D Gaussian Distribution Contour Map", fill=false, c=:reds, color=:red, colorbar=true, ratio = 1.0)
        savefig("Theoredical_GD.png")
        plot(p1, p2, layout=(1, 2), size=(1000, 400))
        savefig("Plot1.png")

        x_i = 1:length(grad_list)
        plot(x_i, grad_list, xlabel = "iterates", ylabel = "norm of gradient of ELBO", title = "gradient of ELBO with Time")
        plot!(x_i, mean_grad, label = "mean")
        savefig("gradient.png")

        plot(x_i, ELBO_list, xlabel = "iterates", ylabel = "ELBO", title = "ELBO with Time")
        savefig("ELBO.png")
    return z';
 end

# Gradient Descent with Armijo
function SteepestDescentArmijo(z0, c1)

    # parameters for Armijo's rule
    alpha0 = 0.001;    # initial value of alpha, to try in backtracking
    eta = 0.01;       # factor with which to scale alpha, each time you backtrack
    MaxBacktrack = 20;  # maximum number of backtracking steps

    # setup for steepest descent
    z = z0;
    successflag = false;

    grad_list = [];
    mean_grad = [];
    ELBO_list = [];

    # perform steepest descent iterations
    for iter = 1:MaxIter
        alpha = alpha0;
        Fval = neg_ELBO(z);
        #display(Fval);
        Fgrad = G_ELBO(z);
        #display(Fgrad);
        push!(grad_list,norm(Fgrad));
        push!(mean_grad,mean(grad_list));
        push!(ELBO_list,-Fval);
        
        if norm(Fgrad)[1] < tol
            display(z')
            @printf("Converged after %d iterations, function value %f\n", iter, -Fval)
            successflag = true;
            break;
        end
        
        # perform line search
        for k = 1:MaxBacktrack
            z_try = z
            z_try1 = zeros(3)
            for p = length(mu_pr)+1:length(z)
                z_try1[p-2] = z[p] - 0.75*alpha*Fgrad[p]
            end
            if (z_try1[1] > 0) &&(z_try1[2] > 0) && (z_try1[1]*z_try1[2]-z_try1[3]^2 >= 0)
                z_try[length(mu_pr)+1:length(z)] = z_try1
                z_try[1:length(mu_pr)] = z[1:length(mu_pr)] - alpha*Fgrad[1:length(mu_pr)]
                Fval_try = neg_ELBO(z_try);
                if (Fval_try > Fval - c1*alpha *(Fgrad'*Fgrad)[1])
                #if (Fval_try > Fval - c1*alpha * norm(Fgrad))
                    alpha = alpha * eta;
                else
                    Fval = Fval_try;
                    z = z_try;
                    break;
                end
            else
                alpha = alpha * eta;
            end
        end

        # print how we're doing, every 10 iterations
        if (iter%100==0)
            @printf("iter: %d: alpha: %f, %f\n", iter, alpha, -Fval)
            display(z')
        end
    end

    #if successflag == false
    #    @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, ELBO(z))
    #end

    # plot
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

        p1 = contour(dx, dy, Z, xlabel="beta_1", ylabel="beta_2", title="2D Gaussian Distribution Contour Map", fill=false, c=:blues, color=:blue, colorbar=true, ratio = 1.0)
        savefig("GD_armijo.png")
        p2 = contour(dx, dy, Z_theo, xlabel="beta_1", ylabel="beta_2", title="2D Gaussian Distribution Contour Map", fill=false, c=:reds, color=:red, colorbar=true, ratio = 1.0)
        savefig("Theoredical_GD.png")
        plot(p1, p2, layout=(1, 2), size=(1000, 400))
        savefig("Plot2.png")

        x_i = 1:length(grad_list)
        plot(x_i, grad_list, xlabel = "iterates", ylabel = "norm of gradient of ELBO", title = "gradient of ELBO with Time")
        plot!(x_i, mean_grad, label = "mean")
        savefig("gradient_armijo.png")

        plot(x_i, ELBO_list, xlabel = "iterates", ylabel = "ELBO", title = "ELBO with Time")
        savefig("ELBO_armijo.png")

    return z';
end
SteepestDescent(z0, 0.001);
#SteepestDescentArmijo(z0, 1e-3);