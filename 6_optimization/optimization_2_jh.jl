
using PrettyTables, Plots, LaTeXStrings, LinearAlgebra, NLsolve, Optim, Roots, Calculus

#TAKEAWAYS (SLIDES 22 - 38) - multidimensional local unconstrained optimization of single-valued functions (gradient: nx1, Hessian: nxn)
#unconstrained local optimization - we used Optim
#always verify convergence

#newton method (SLIDES 27,28)
#fast, can fail to converge, uses lots of memory (requires computing, storing Hessian)
#deals with flat region much better than gradient descent (slide 31)
#status can show Failure, even if objective function and gradient are miniscule 
#e.g. |g(x)| gradient condition failed, you can decide for yourself if it's good enough
#if you know exact Hessian, consider using newton

#bfgs (slide 34) - uses approximate pos. def. H (less memory), but might fail to converge
#SO FAR: WE ONLY DID UNCONSTRAINED OPTIMIZATION

#CONSTRAINED OPTIMIZATION (IMPORTANT SLIDE 38): avoid as much as possible
#change problem into unconstrained one or put penalty in obj function for violating the implicit constraint
#if you have to: use existing packages NLopt, ForwardDiff


rosenbrock(x) = (1.0 .- x[1]).^2 .+ 100.0 .* (x[1] .- x[2].^2).^2 #R^2 -> R function; easy to verify global min are [1,1], [1,-1]
#Rosenbrock function - used as a performance test for optimization algos 
grid = -1.5:0.11:1.5;

plot(grid,grid,(x,y)->rosenbrock([x, y]),st=:surface,camera=(50,20))
plot(grid,grid,(x,y)->rosenbrock([x, y]),st=:contour,color=:turbo, levels = 20,clabels=true, cbar=false, lw=1)


## a function to plot the optimization results
function plot_optim(res, start, x,y,ix; offset = 0)
	contour(x, y, (x,y)->sqrt(rosenbrock([x, y])), fill=false, 	color=:turbo, legend=false, levels = 50)
    xtracemat = hcat(Optim.x_trace(res)...)
    plot!(xtracemat[1, (offset+1):ix], xtracemat[2, (offset+1):ix], mc = :white, lab="")
    scatter!(xtracemat[1:1,2:ix], xtracemat[2:2,2:ix], mc=:black, msc=:red, lab="")
    scatter!([1.], [1.], mc=:blue, msc=:blue,markersize = 8, lab="minimum")
    scatter!([start[1]], [start[2]], mc=:yellow, msc=:black, label="start", legend=true)
    scatter!([Optim.minimizer(res)[1]], [Optim.minimizer(res)[2]], mc=:black, msc=:black, label="last", legend=true)
end

## gradient descent (slide 31)
x0 = [1.0, 0.5] #depending on the starting point it might not work well in areas that function is very flat
res_descent = optimize(rosenbrock, x0, GradientDescent(), Optim.Options(store_trace=true, extended_trace=true, iterations = 5000))
#Status: success - ALWAYS CHECK
res_descent.minimizer #look at LHS Julia panel -> [~1,~1] - point at which the objective f is minimized
res_descent.minimum #8.757658838312685e-17
#look at LHS Julia panel -> final value of objective function f -> look at the top - it is very close to 0 as it should, good
plot_optim(res_descent, x0, -0:0.01:1.5, -0:0.01:1.5,10) #x0 = start (yellow dot)

## newton's method (slide 27-28)
res_newton = optimize(rosenbrock, x0, Newton(), Optim.Options(store_trace=true, extended_trace=true, iterations = 5000))
#Status: success, only 9 iterations
res_newton.minimizer #also very close to [1,1] - good
plot_optim(res_newton, x0, -0:0.01:1.5, -0:0.01:1.5,10) 


## bfgs (slide 34) - instead of hessian using positive definite approximation 
res_bfgs = optimize(rosenbrock, x0, BFGS(), Optim.Options(store_trace=true, extended_trace=true, iterations = 20000))
#Status: failure with x0 = [1.0, 0.5] and 20000 iterations, let's check another x0 (even though gave [~1,~1] as min)

x0 = [0.5, 1.0]
res_bfgs = optimize(rosenbrock, x0, BFGS(), Optim.Options(store_trace=true, extended_trace=true, iterations = 20000)) #Status: success
plot_optim(res_bfgs, x0, -0:0.01:1.5, -0:0.01:1.5,9) #with 10 as last input it didn't work

########## NOT VERY IMPORTANT UNTIL THE END - METHODS WHICH ARE OFTEN INEFFICIENT #####################

x0 = [1.0, 0.5]
## bfgs (slide 34) with a stupid stopping criterion
res_stupid = optimize(rosenbrock, x0, BFGS(), Optim.Options(store_trace=true, extended_trace=true,iterations = 10,g_tol = 1e-1, show_trace = true ))
plot_optim(res_stupid, x0, -0:0.01:1.5, -0:0.01:1.5,8)

## nelder mead (slides 23-26) - slow, for smooth problems outperformed by other methods
res_nm = optimize(rosenbrock, x0, NelderMead(), Optim.Options(iterations = 10, show_trace = true )) #Status:failure
res_nm.minimizer #VERY BAD [0.65, 0.81] INSTEAD OF [1,1]

#the above are local optimization methods - looking around a specific point
#there are also global methods (we haven't used them at class)

## simulated annealing - THE FIRST GLOBAL METHOD USED SO FAR
#temperature measuring
# - the closer it gets to solution, the cooler it gets - then jumps with smaller steps
res_sa = optimize(rosenbrock, x0, SimulatedAnnealing(), Optim.Options(store_trace=true, extended_trace=true, iterations = 100000)) #Status: failure
plot_optim(res_sa, x0, -0:0.01:1.5, -0:0.01:1.5,9000)
res_sa.minimizer #[~1,~-1] instead of [1,1] - FOUND THE OTHER MINIMUM [1,-1], STRANGE, BECAUSE FAR AWAY FROM x0 = [1.0, 0.5]
#DOESN'T WORK VERY WELL HERE AND IN GENERAL TOO

########## END OF NOT VERY IMPORTANT - METHODS WHICH ARE OFTEN INEFFICIENT #######################