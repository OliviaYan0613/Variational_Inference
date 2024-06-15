# Implementation of steepest descent algorithm, with 
# -- constant step size (SteepestDescent), and
# -- step size chosen via Armijo's rule (SteepestDescentArmijo)
#
# The built-in minimization problem is
#           min   0.5*x'*Q*x - b'*x + 1
#
# Run this as (for example):
# >> include("steep.jl");
# >> x0 = [10;10];
# >> alpha = 0.1;
# >> x = SteepestDescent(x0,alpha);
# >> c1 = 1e-3;
# >> y = SteepestDescentArmijo(x0, c1);
#


using Printf

# matrix and vector used in quadratic form. 
# defined here, because they are used in both F(x), and DF(x)
Q = [20 -1 -1; -1 2 -1;-1 -1 3];
b = [18;0;1];
#x0 = [2;2];
AD = [0.1 0 0; 0 1 0; 0 0 1];



# set parameters here, for all gradient descent algorithms
tol = 1e-10;     # tolerance on norm of gradient
MaxIter = 1e6;  # maximum number of iterations of gradient descent


# define function
#function F(x)
   #val = 0.5*x'*Q*x - b'*x + 1
   #return val
#end

function F(x)
   val = 100((x[2])-(x[1])^2)^2+(1-(x[1]))^2
   return val
end

# define gradient
#function DF(x)
   #grad = Q*x-b;
   #gradient adjustment
   #grad = AD*grad
   #return grad
#end

function DF(x)
   grad = [400*((x[1])^3)+(2-400*(x[2]))*(x[1])-2; 200((x[2])-(x[1])^2)]
   return grad
end

#
# steepest descent algorithm, with constant step size
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
#    alpha = step size. Constant, in this algorithm.
# output: 
#    x = final point
#
function SteepestDescent(x0,alpha)

   # setup for steepest descent
   x = x0;
   successflag = false;
   x1 = [];
   y1 = [];

   # perform steepest descent iterations
   for iter = 1:MaxIter
       Fval = F(x);
       Fgrad = DF(x);
       push!(x1,x[1]);
       push!(y1,x[2]);
       if sqrt(Fgrad'*Fgrad) < tol
          @printf("Converged after %d iterations, function value %f\n", iter, Fval)
          successflag = true;
          # plot
            # Contour plot
            f(x, y) = 100(y-x^2)^2+(1-x)^2 # surface function 
            x = range(0, 2.5, length=100)                 # x values
            y = range(0.5, 3, length=50)                  # y values
            z = f.(x', y)                             # define z as f(x,y)
            plt4 = contour(x, y, z)                     # plot
            xlabel!("x_1")
            ylabel!("x_2")
            title!("Contour plot")
            display(plt4)

            # overlay a line plot on the contour plot
            plot!(x1,y1, label="GD",marker=(:circle,2),color=:blue)
            display(plt4)

            savefig("Plot1.png")
          break;
       end
       # perform steepest descent step
       #@print(x);
       x = x - alpha*Fgrad;

       # print how we're doing, every 10 iterations
      #if (iter%10==0)
         #@printf("iter: %d, alpha: %f, %f, %f, %f\n", iter, alpha, x[1], x[2], Fval)
      #end

   end
   if successflag == false
       @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, F(x))
   end
   return x;
end



#
# steepest descent algorithm, with Armijo's rule for backtracking
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
#    c1 = slope, in Armijo's rule.
# output: 
#    x = final point
#
function SteepestDescentArmijo(x0, c1)

   # parameters for Armijo's rule
   alpha0 = 10.0;    # initial value of alpha, to try in backtracking
   eta = 0.5;       # factor with which to scale alpha, each time you backtrack
   MaxBacktrack = 20;  # maximum number of backtracking steps

   # setup for steepest descent
   x = x0;
   successflag = false;   
   x1 = [];
   y1 = [];

   # perform steepest descent iterations
   for iter = 1:MaxIter

      alpha = alpha0;
      Fval = F(x);
      Fgrad = DF(x);
      push!(x1,x[1]);
      push!(y1,x[2]);

      # check if norm of gradient is small enough
      if sqrt(Fgrad'*Fgrad) < tol
         @printf("Converged after %d iterations, function value %f\n", iter, Fval)
         successflag = true;
         # plot
            # Contour plot
            f(x, y) = 100(y-x^2)^2+(1-x)^2 # surface function 
            x = range((minimum(x1)-1), 2.5, length=100)                 # x values
            y = range(0.5, 4, length=50)                  # y values
            z = f.(x', y)                             # define z as f(x,y)
            plt4 = contour(x, y, z)                     # plot
            xlabel!("x_1")
            ylabel!("x_2")
            title!("Contour plot")
            display(plt4)

            # overlay a line plot on the contour plot
            plot!(x1,y1, label="GD with Armijo",marker=(:circle,2),color=:blue)
            display(plt4)

            savefig("Plot1.png")
         break;
      end

      # perform line search
      for k = 1:MaxBacktrack
         x_try = x - alpha*Fgrad;
         Fval_try = F(x_try);
         if (Fval_try > Fval - c1*alpha *Fgrad'Fgrad)
            alpha = alpha * eta;
         else
            Fval = Fval_try;
            x = x_try;
            break;
         end
      end

      # print how we're doing, every 10 iterations
      if (iter%10==0)
         @printf("iter: %d: alpha: %f, %f, %f, %f\n", iter, alpha, x[1], x[2], Fval)
         #@printf("iter: %d, %f, %f, %f\n" , iter, Fgrad[1],Fgrad[2],Fgrad[3])
         @printf("iter: %d, %f, %f\n" , iter, Fgrad[1],Fgrad[2])
      end
   end

   if successflag == false
       @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, F(x))
   end

   return x;
end



