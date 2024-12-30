
using PrettyTables, Plots, LaTeXStrings, LinearAlgebra, NLsolve, Optim, Roots, Calculus



f_univariate(x)     = 2x^2+3x+1
dfdx_univariate(x)  = 4x+3

plot(f_univariate, -2.0, 1.0, label = L"f(x) = 2x^2+3x+1", xlabel = L"x", ylabel = L"f(x)", title = "Univariate Function", lw = 2)
plot!(dfdx_univariate, -2.0, 1.0, label = L"f'(x) = 4x+3", lw = 2)


# use bisection on f'
find_zero(dfdx_univariate, (-2,2), Bisection(), verbose = true, atol = 1e-14) #-0.75 as it should

# use golden section on f
optimize(f_univariate, -2.0, 1.0, GoldenSection(),atol = 1e-14)
#brackets -2, 1, atol stop when the level of the interval is 1e-14
#it is economic with respect to # of iterations (here: bisection used lower number only because we gave it a derivative function rather than the primitive)

# use newton's method
function newton(f,dfdx,dfdx2,x0; ε=10e-6, δ=10e-6, maxcounter = 100, verbose = false) #we need to give 1st and 2nd derivative in our functions: packages like NLSolve and Optim don't need them
    # this algorithm is from Judd (1998), page 98
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


dfdx2_univariate(x)  = 4.0

newton(f_univariate,dfdx_univariate,dfdx2_univariate,100.0)


# what if we do not have derivatives? [and we don't use NLSolve or Optim]
# we can use Calculus.jl package to get finite differences, it's not very efficient though

# finite differences use the fact that the derivative is the limit of the difference quotient
# f′(x) = lim_{h->0} (f(x+h) - f(x))/h
# so we can approximate it with a small h

# this is usually not the best way to get derivatives 
# here we need to do it like this to use our "newton" function
# this function needs functions as arguments

f_derivative(x)     = derivative(f_univariate,x) #function in calculus package
f_2nd_derivative(x) = second_derivative(f_univariate,x)

f_derivative(3) #slightly different numbers
dfdx_univariate(3) #slightly different numbers

f_2nd_derivative(3.0)
f_2nd_derivative(10.0)

plot(dfdx_univariate, -2.0, 1.0, label = L"f'(x) = 4x+3", lw = 2)
plot!(f_derivative, -2.0, 1.0, label = L"f'(x) = 4x+3 (approx)", lw = 2, linestyle = :dot)
plot!(dfdx2_univariate, -2.0, 1.0, label = L"f''(x) = 4", lw = 2)
plot!(f_2nd_derivative, -2.0, 1.0, label = L"f''(x) = 4 (approx)", lw = 2, linestyle = :dot)

#calculus or Optim (check) package mainly used for classes
#packages doing automatic differentiation use dual numbers -> morecomplicated easy to encounter errors
#investigate whether this is the case for NLSolve and Optim or calculus