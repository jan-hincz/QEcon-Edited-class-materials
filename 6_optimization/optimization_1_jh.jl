
using PrettyTables, Plots, LaTeXStrings, LinearAlgebra, NLsolve, Optim, Roots, Calculus

#TAKEAWAYS (SLIDES 1 - 21) - unidimensional local unconstrained optimization (only Optim, Roots and inefficient Calculus used)
#often won't work well with multimodal functions if initial_x far away from the solution

#ALWAYS check convergence, if solution makes sense and is robust to changes in initial_x
#use existing packages to solve problems, understand obj function to choose appriopriate algo
#Roots - for solving univariate nonlinear equations, NLsolve also for multivariate nonlinear equations
#Optim for unconstrained optimization; NLopt for constrained or global optimization
#packages doing automatic differentiation use dual numbers -> more complicated, easy to encounter errors (ForwardDiff, used sts in NLsolve,NLopt)
#ForwardDiff for complicated tasks (like DSGE modelling, very popular)
#Calculus package mainly used for classes - inefficient

#CONSTRAINED OPTIMIZATION (IMPORTANT SLIDE 38): avoid as much as possible
#change problem into unconstrained one or put penalty in obj function for violating the implicit constraint
#if you have to: use existing packages NLopt, ForwardDiff

#no free lunch theorem Wolpert, Macready (1997)
#no optimization algo better than any other algo on all problems

#types of numerical algos (SLIDE 5)
#comparison methods - easy to converge, but slow
#e.g. GoldenSection() in optimize() of Optim package
#gradient methods - use info on gradient of objective f and constraints
#curvature methods - require strong conditions, but fast; e.g. newton (SLIDES 14-15, 19)
#newton method requires C^2, won't work well with multimodal functions - DON'T USE IT MANUALLY



f_univariate(x)     = 2x^2+3x+1
dfdx_univariate(x)  = 4x+3

plot(f_univariate, -2.0, 1.0, label = L"f(x) = 2x^2+3x+1", xlabel = L"x", ylabel = L"f(x)", title = "Univariate Function", lw = 2)
plot!(dfdx_univariate, -2.0, 1.0, label = L"f'(x) = 4x+3", lw = 2)


#use bisection on f' TO FIND f'(x) = 0; package Roots for solving non-linear univariate functions - see nonlinear_1_jh.jl
find_zero(dfdx_univariate, (-2,2), Bisection(), verbose = true, atol = 1e-14) #Converged to: -0.75 as it should

# use golden section on f (primitive function); package Optim
optimize(f_univariate, -2.0, 1.0, GoldenSection(),atol = 1e-14) #-0.75; Convergence: true
#brackets -2, 1, atol -> stop when the level of the interval is 1e-14
#it is economic with respect to # of iterations, here: bisection used lower # only because we gave it a f' rather than f)


############# UNTIL THE END: NOT VERY IMPORTANT: MANUAL FUNCTIONS, WHICH ARE NOT VERY SMART ################################

# use newton's method; #we need to give f' and f'': packages like NLSolve and Optim don't need them
function newton(f,dfdx,dfdx2,x0; ε=10e-6, δ=10e-6, maxcounter = 100, verbose = false)
    # this algorithm is from Judd (1998), page 98 (SLIDE 19)
    x_old = x0
    x_new = 2*abs(x0) + 1
    counter = 1
    guesses = []

    while ((abs(dfdx(x_old)) > δ) || (abs(x_new-x_old) > ε * (1+abs(x_old))))
        guesses = push!(guesses,x_old)
        if verbose
            println("Iteration = $counter")    
            println("Point = $x_old")    
            println("Value = $(f(x_old))")  
            println("Derivative = $(dfdx(x_old))")  
            println("")
        end


        if counter > maxcounter
            println("Maximum number of iterations ($maxcounter) reached")
            break
        end

        counter += 1
        x_old = x_new
        x_new = x_old - dfdx(x_old)/dfdx2(x_old)
        
        
    end

    guesses = push!(guesses,x_new)
    return (argmin = x_new, val = f(x_new), derivative  = dfdx(x_new), points = guesses, iteration = counter)
end


dfdx2_univariate(x)  = 4.0 #at the top we have f and f' = 4x + 3

newton(f_univariate,dfdx_univariate,dfdx2_univariate,100.0) #argmin = -0.75; good


# what if we do not have derivatives? [and we don't want to use NLSolve or Optim]
# we can use Calculus.jl package to get finite differences (NOT EFFICIENT, MAINLY FOR CLASSES)

# finite differences use the fact that the derivative is the limit of the difference quotient
# f′(x) = lim_{h->0} (f(x+h) - f(x))/h
# so we can approximate it with a small h

# this is usually not the best way to get derivatives 
# here we need to do it like this to use our "newton" function
# this function needs functions as arguments

f_derivative(x)     = derivative(f_univariate,x) #function in Calculus package
f_2nd_derivative(x) = second_derivative(f_univariate,x) #function in Calculus package

f_derivative(3) #slightly less precise number: 14.999999999962176
dfdx_univariate(3) #f' defined explicitly at the top; precise number: 15 

f_2nd_derivative(3.0) #slightly above 4, even though we know f'' = 4 for all x
f_2nd_derivative(10.0) #slightly above 4, even though we know f'' = 4 for all x

plot(dfdx_univariate, -2.0, 1.0, label = L"f'(x) = 4x+3", lw = 2) #precise f' (defined at the top)
plot!(f_derivative, -2.0, 1.0, label = L"f'(x) = 4x+3 (approx)", lw = 2, linestyle = :dot) #Calculus f'
plot!(dfdx2_univariate, -2.0, 1.0, label = L"f''(x) = 4", lw = 2) #precise f'' (defined at the top)
plot!(f_2nd_derivative, -2.0, 1.0, label = L"f''(x) = 4 (approx)", lw = 2, linestyle = :dot) #Calculus f''
#equivalent plots very close - good
############# END OF: NOT VERY IMPORTANT: MANUAL FUNCTIONS, WHICH ARE NOT VERY SMART ################################