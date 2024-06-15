# Steepest descent to solve a least squares problem

# Load packages and set parameters
using Printf
using Plots
#using Random
tol = 1e-5;
eps = 1e-3;
maxiter = 1000;

# read data file with N entries
N = 1000;
include("data.jl")
Q = 1/N * A'*A;

# define loss function
function L(x)
   val = 1/N*sum(0.5*(A*x - d).^2)
   return val
end

# define gradient of loss function
function DL(x)
   gradx = sum((A*x-d).*A[:,1]);
   grady = sum((A*x-d).*A[:,2]);
   return 1/N*[gradx; grady]
end

# define gradient of i-th individual loss function
function DLind(x,i)
   gradx = (A[i,:]'*x-d[i]).*A[i,1];
   grady = (A[i,:]'*x-d[i]).*A[i,2];
   return [gradx; grady]
end


# gradient algorithm with exact line search
function GradientDescent(x0)
   x = x0;
   # vector to store iterates
   xvec = zeros(2,maxiter);
   xvec[:,1] = x0;
   # perform steepest descent iterations
   # alpha = 0.1
   for iter = 1:maxiter
       Lval = L(x);
       Lgrad = DL(x);
       if sqrt(Lgrad'*Lgrad) < tol
          @printf("Converged after %d iterations with value %f\n", iter, Lval)
          break;
       end
       alpha = (Lgrad'*Lgrad) / (Lgrad'*Q*Lgrad);
       x = x - alpha*Lgrad;
       if (iter%1==0)
          @printf("iter: %d: alpha: %f, %f, %f, %f\n", iter, alpha, x[1], x[2], Lval)
       end
       xvec[:,iter+1] = x;
   end
   return xvec
end


# stochatic gradient descent
function MomentumDescent(x0)
   x = x0;
   # vector to store iterates
   xvec = zeros(2,maxiter+1);
   #xvec = zeros(2,N+1);
   xvec[:,1] = x0;
   # generate random order of individual loss function
   order = [];
   #for k = 1:1000
      #if length(order) < maxiter
         #s = shuffle(rng,Vec(1:N));
         append!(order,randperm(N));
         #@printf("first element %f and length of order %f", order[1], length(order));
      #end
   #end
   # perform stochastic gradient descent
   for iter = 1:500
      alpha = 0.1;
       # **** TODO: SGD method  *****
      i = order[iter];
      #Lval = L(x);
      Lgrad = DLind(x,i);
      #if sqrt(Lgrad'*Lgrad) < tol
         #@printf("Converged after %d iterations with value %f\n", iter, Lval);
         #@printf("Converged after %d iterations", iter);
         #break;
      #end
      #alpha = (Lgrad'*Lgrad) / (Lgrad'*Q*Lgrad);
      x = x - alpha*Lgrad;
      if (iter%10==0)
         #@printf("iter: %d: alpha: %f, %f, %f, %f\n", iter, alpha, x[1], x[2], Lval);
         @printf("iter: %d: alpha: %f, %f, %f\n", iter, alpha, x[1], x[2]);
      end
      xvec[:,iter+1] = x;
   end
   for iter = 501:1000
      alpha = 0.005;
       # **** TODO: SGD method  *****
      i = order[iter];
      #Lval = L(x);
      Lgrad = DLind(x,i);
      #if sqrt(Lgrad'*Lgrad) < tol
         #@printf("Converged after %d iterations with value %f\n", iter, Lval);
         #@printf("Converged after %d iterations", iter);
         #break;
      #end
      #alpha = (Lgrad'*Lgrad) / (Lgrad'*Q*Lgrad);
      x = x - alpha*Lgrad;
      if (iter%10==0)
         #@printf("iter: %d: alpha: %f, %f, %f, %f\n", iter, alpha, x[1], x[2], Lval);
         @printf("iter: %d: alpha: %f, %f, %f\n", iter, alpha, x[1], x[2]);
      end
      xvec[:,iter+1] = x;
   end
   return xvec
end



# visualization of contour lines
x = -0.5:0.02:1.5;
y = -0.5:0.02:1.5;
Z = zeros(length(y), length(x));
for i=1:length(x)
   for j=1:length(y)
     Z[j,i] = L([x[i]; y[j]]);
   end
end
contourf(x,y,Z,levels=25);

lx = MomentumDescent([1;0.3])
plot!(lx[1,1:20:1000],lx[2,1:20:1000], lm = 2, label = "iterates")
savefig("sgdDescentChange.png")


# command to visualize first 100 iterates
# xvec = GradientDescent([1;1])
# plot!(xvec[1,1:1:100], xvec[2,1:100], lw = 2, label = "iterates")
