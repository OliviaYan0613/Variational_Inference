using Printf, ForwardDiff, Distributions, Random, LinearAlgebra, Plots, Flux

Random.seed!(123);

SteepestDescent(z0, 0.01);
#SteepestDescentArmijo(z0, 1e-3);

# setup
beta_true =  [1.0 2.0]';
#beta_true = 3
n = 10;

MaxIter = 1000;
tol = 1e-4; 

mu_pr = [2.0 2.0]';
#mu_pr = 1
sigma2_pr = [1.0 0.0; 0.0 2.0];
#sigma2_pr = 1
noise = 0.01;

sigma2_pr_diag = [1.0 2.0]';
#sigma2_pr_diag = 1;
z0 = [mu_pr; sigma2_pr_diag];

x = randn(n,2);
#x = randn(n);
y = x*beta_true + noise*randn(n);

# prior
# p(y|beta)
function p_y(beta)
    prob = exp(-0.5*(y - x*beta)'*(y - x*beta)/noise)
    prob = max(prob[1],1e-200)
    return prob
end
# p(beta)
function p_b(beta)
    prob = exp(-0.5*(beta - mu_pr)'*inv(sigma2_pr)*(beta - mu_pr))/(2*pi)/sqrt(norm(sigma2_pr))
    #prob = max(prob[1],1e-200)
    return prob
end
# q(beta)
function q_b(beta,mu,sigma2)
    sigma2_mx = [sigma2[1] 0; 0 sigma2[2]]
    prob = exp(- 0.5*(beta - mu)'*inv(sigma2_mx)*(beta - mu))/(2*pi)/sqrt(norm(sigma2_mx))
    #prob = max(prob[1],1e-200)
    return prob
end

# define loss function
# MC sampling
function sampling(pdf, N, mu, sigma2; b_min=-10, b_max=10)
    samples = []
    while length(samples) < N
        b1 = rand() * (b_max - b_min) + b_min        
        b2 = rand() * (b_max - b_min) + b_min
        b = [b1; b2]
        y = rand()        # Sample y uniformly within [0, 1]
        if y <= pdf(b, mu, sigma2)[1]
            push!(samples, b)
        end
    end
    return samples
end

function ELBO(z)
    res = 0
    mu_vec = z[1:length(mu_pr)]
    sigma2_vec = z[length(mu_pr)+1:length(z)]
    N = 100
    beta = sampling(q_b, N, mu_vec, sigma2_vec);
    mean_beta = mean(beta)
    log_p_y = 0
    log_p_b = 0
    log_q_b = 0
    for i = 1:length(beta)
        b = beta[i]
        log_p_y = log_p_y + log(p_y(mean_beta)[1])
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
    #diff  = gradient(neg_ELBO, z)
    return diff
end

# Gradient descent method
function SteepestDescent(z0,alpha)
    # setup for steepest descent
    z = z0;
    successflag = false;
    Fval_old = 0;
    #mu_iter = [];
    #sigma2_iter = [];
 
    # perform steepest descent iterations
    for iter = 1:MaxIter
        Fval = neg_ELBO(z);
        Fgrad = G_ELBO(z);
        #push!(mu_iter,z[1:length(mu_pr)]);
        #push!(sigma2_iter,z[length(mu_iter)+1:length(z)]);
        if sqrt(Fgrad'*Fgrad)[1] < tol
        #if abs(Fval - Fval_old) < tol
            @printf("Converged after %d iterations, function value %f\n", iter, -Fval)
            successflag = true;
            # plot
                # Contour plot
                # Mean and covariance matrix for the Gaussian distribution
                mu_post = vec(z[1:length(mu_pr)]);
                sigma2_post = z[length(mu_pr)+1:length(z)];
                Sigma2 = diagm(sigma2_post);
                # Create a grid of x and y values
                dx = range(-3, stop=5, length=100)
                dy = range(-3, stop=5, length=100)

                # Evaluate the Gaussian density at each point in the grid
                Z = [pdf(MvNormal(mu_post, Sigma2), [xi, yi]) for xi in dx, yi in dy]
                Z = reshape(Z, length(dx), length(dy))'

                contour(dx, dy, Z, xlabel="\beta_1", ylabel="\beta_2", title="2D Gaussian Distribution Contour Map", ratio = 1)
                #savefig("Plot1.png")
            break;
        end
        # perform steepest descent step
        #@print(x);
        #for k = 1:length(mu_pr)
        #    z[k] = z[k] - alpha*Fgrad[k]
        #end
        z[1:length(mu_pr)] = z[1:length(mu_pr)] - alpha*Fgrad[1:length(mu_pr)]
        for k = length(mu_pr)+1:length(z)
            z_try = z[k] - 0.1*alpha*Fgrad[k]
            if (z_try > 1e-6)
                z[k] = z_try
            end
            #z[k] = max(z_try,1e-6)
        end
        Fval_old = Fval;
 
        # print how we're doing, every 10 iterations
        if (iter%10==0)
          #@printf("iter: %d, alpha: %f, %f\t, %f\t, %f\n", iter, alpha, z[1:length(mu)], z[length(mu)+1:length(z)], Fval[1])
          @printf("iter: %d, alpha: %f, %f\n", iter, alpha, -Fval[1])
          display(z')
        end
 
    end
    if successflag == false
        @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, ELBO(z))
    end
    return z';
 end

# Gradient Descent with Armijo
function SteepestDescentArmijo(z0, c1)

    # parameters for Armijo's rule
    alpha0 = 10.0;    # initial value of alpha, to try in backtracking
    eta = 0.5;       # factor with which to scale alpha, each time you backtrack
    MaxBacktrack = 20;  # maximum number of backtracking steps

    # setup for steepest descent
    z = z0;
    successflag = false;
    #mu_iter = [];
    #sigma2_iter = [];

    # perform steepest descent iterations
    for iter = 1:MaxIter
        alpha = alpha0;
        Fval = neg_ELBO(z);
        #display(Fval);
        Fgrad = G_ELBO(z);
        #display(Fgrad);
        #push!(mu_iter,z[1:length(mu_pr)]);
        #push!(sigma2_iter,z[length(mu_iter)+1:length(z)]);
        if sqrt(Fgrad'*Fgrad)[1] < tol
            display(z')
            @printf("Converged after %d iterations, function value %f\n", iter, -Fval)
            successflag = true;
            # plot
                # Contour plot
                # Mean and covariance matrix for the Gaussian distribution
                mu_post = z[1:length(mu_pr)];
                sigma2_post = z[length(mu_pr)+1:length(z)];
                Sigma2 = diagm(sigma2_post);
                # Create a grid of x and y values
                dx = range(1, stop=5, length=100)
                dy = range(8, stop=12, length=100)

                # Create a grid of points
                X = [xi for xi in dx, yi in dy]
                Y = [yi for xi in dx, yi in dy]

                # Evaluate the Gaussian density at each point in the grid
                Z = pdf(MvNormal(mu_post, Sigma2), hcat(X[:], Y[:]))
                Z = reshape(Z, length(x), length(y))'

                contour(dx, dy, Z, xlabel="X", ylabel="Y", title="2D Gaussian Distribution Contour Map")
                #savefig("Plot1.png")
            break;
        end

        # perform line search
        for k = 1:MaxBacktrack
            z_try = z - alpha*Fgrad;
            z_try[length(mu_pr)+1:length(z_try)] = z[length(mu_pr)+1:length(z_try)] - 0.1*alpha*Fgrad[length(mu_pr)+1:length(z_try)]
            Fval_try = neg_ELBO(z_try);
            if (Fval_try > Fval - c1*alpha *(Fgrad'*Fgrad)[1])
                alpha = alpha * eta;
            else
                Fval = Fval_try;
                z = z_try;
                break;
            end
        end

        # print how we're doing, every 10 iterations
        if (iter%10==0)
            @printf("iter: %d: alpha: %f, %f\n", iter, alpha, -Fval)
            display(z')
        end
    end

    if successflag == false
        @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, ELBO(z))
    end

    return z';
end

