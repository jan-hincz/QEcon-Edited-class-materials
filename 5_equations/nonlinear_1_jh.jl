# Examples from/based on Fundamentals of Numerical Computation, Julia Edition. Tobin A. Driscoll and Richard J. Braun

using PrettyTables, Plots, LaTeXStrings, LinearAlgebra, NLsolve, Roots

#TAKEAWAYS (SLIDES 29 - 45) - solving methods for nonlinear (mainly) univariate functions

#packages: Roots for a univariate function, NLsolve more general - use it for more complicated tasks
#1. GENERAL LESSON: NO GUARANTEE A NON-LINEAR JULIA FUNCTION WILL FIND ANY ROOTS
#2. ALWAYS VERIFY THAT THIS IS THE CASE: "Convergence: true"/"Converged to .." etc.
#3. NO GUARANTEE JULIA WILL FIND ALL ROOTS
#4. TRY MANY STARTING POINTS

#secant method (slides 43-44) - modified Newton method
#+: often quicker than Newton method

#Newton method (slides 41-42)
#+: fast; works for multivariate functions (slide 46)
#-: good guess needed; not guaranteed to converge
#-: gotta calculate derivative at each step
#-: function has to be sufficiently smooth

#bisection method
#+: in theory guaranteed to converge to a root
#-: slow; if multiple roots, will find just one of them (general problem with root-finding algos)
#-: works only for univariate functions; vectors as inputs won't work (slide 31)



# example using NLsolve package

f(x) = x.^2 .- 3 .+ x .* sin.( 1 ./ x .+ x .^ 2 ) #use broadcasting .

plot(f, -30, 30, label="f(x) =  x^2 - 3 + x * sin(1/x + x^2 )", legend=:topleft)
plot!(zero, -30, 30, label="y=0")

#IT'S NOT BISECTION (WE ONLY CHOSE one point 2.5, not a and b)
guess = 2.5 #look at the plot; it will search close to it
nlsolve(f,[guess],ftol=1e-14,show_trace=true) #stop when |f(x)| <= ftol=1e-14
#Convergence: true; zero: 2.19972..; JUST 5 iterations
#ALWAYS VERIFY THAT THIS IS THE CASE: "Convergence: true"

nlsolve(f,[guess],method=:newton,ftol=1e-14,show_trace=true) #Convergence: true; zero: 2.19972..
#again just 5 iterations


#package Roots; BISECTION METHOD (we see from plot root is between a=0.1 and b=3)

find_zero(f, (0.1,3), Bisection(), verbose = true, atol = 1e-14) #"Converged to: 2.1997235195725415" like before 
#ALWAYS VERIFY THAT THIS IS THE CASE: "Converged to: .."
#but 51 iterations instead of 5 like above: BISECTION IS SLOW
#it won't work for any a and b (e.g. -300,300), because the initial function might not be computable by Julia

#package Roots; SECANT METHOD
find_zero(f, (1.4,1.5), Order1(), verbose = true) #SECANT METHOD; 35 iterations; Converged to: 2.199723519572541
find_zero(f, 1.4, verbose = true) #Converged to: 2.199723519572541; modified Secant; JUST 5 ITERATIONS


#############UNTIL THE END: NOT VERY IMPORTANT: MANUAL FUNCTIONS, WHICH ARE NOT VERY SMART ################################

# bisection method - MANUAL FUNCTION, NOT VERY SMART, E.G. NO CAP ON # OF ITERATIONS
function bisection(f,a,b,tolerance) #tolerance here related to length of the interval b-a, not value of the function
    b > a || error("b must be greater than a")
    f(a)*f(b) > 0 && error("f(a) and f(b) must have opposite signs")
    while abs(b-a) > tolerance
    c = (a+b)/2
    sign(f(c)) == sign(f(a)) ? a = c : b = c #recall: 1-line conditional statements
    #if true -> a = c, if false -> b = c (slide 31)
    end
    return (a+b)/2
end

bisection(f,1,3,1e-14) #2.19972351957254


# newton's method (slide 41)

f(x) = x*exp(x) - 2;
dfdx(x) = exp(x)*(x+1);
r = nlsolve(x -> f(x[1]),[1.]).zero

plot(f, -1, 1, label="f(x) = x*exp(x) - 2", legend=:topleft)

x = [BigFloat(10);zeros(7)] #BigFloat will allow better than double precision
for k = 1:7
    x[k+1] = x[k] - f(x[k]) / dfdx(x[k]) #from slide 38
end
r = x[end] #number so long, because we used BigFloat

ϵ = @. Float64(x[1:end-1] - r) #we can see the number of precise digits goes up by 2 (error gets smaller)
logerr = @. log(abs(ϵ))
[ logerr[i+1]/logerr[i] for i in 1:length(logerr)-1 ] # p = 2 (slide 41) 
#we can see it converges to 2.sth, but NEWTON DOESN'T ALWAYS CONVERGE

### implement newton's method - version of loop above with stopping algo (so that it doesn't iterate infinitely if no conversion)
function newton(f,dfdx,x₁;maxiter=40,ftol=100*eps(),xtol=100*eps()) #after ; -> default values; eps = machine epsilon
    x = [float(x₁)]
    y = f(x₁)
    Δx = Inf   # for initial pass below, measures distance between 2 consecutive iterations
    k = 1 #counts iterations

    while (abs(Δx) > xtol) && (abs(y) > ftol)
        dydx = dfdx(x[k])
        Δx = -y/dydx            # Newton step; slide 41
        push!(x,x[k]+Δx)        # append new estimate

        k += 1
        y = f(x[k])
        if k==maxiter
            @warn "Maximum number of iterations reached."
            break   # exit loop
        end
    end
    return x
end

x = newton(f,dfdx,1.0) #last entry 0.852605502013726 is our found solution
#won't converge for every number

myscatter = scatter(;xlim = [0.825,1.1],ylim = [-0.25,3])
plot!(f, 0.825, 1.1, label="f(x) = x*exp(x) - 2", legend=:topleft)
animation = @animate for (ind,point) in enumerate(x)
    scatter!(myscatter,[point], [f(point)], alpha = 1 - ind/length(x),ms = 5 + 10*ind/length(x),color = :blue,lab="")
end

gif(animation,fps = 5) #shows consecutive iterations, the last being our solution


### babylonian method - ANCIENT METHOD
f(x) = x^2 - 2 #we now 2^0.5 and -(2^0.5) are solutions
dfdx(x) = 2x
x = 1.0 #closer to 2^0.5; won't find -(2^0.5)
x = newton(f,dfdx,x) #last entry gets very close to positive root of 2

ϵ = @. Float64(x[1:end-1] - sqrt(2)) #very small errors


### bad case 

f(x) = sign(x) * sqrt(abs(x))
plot(f, -1, 1, label="f(x) = sign(x) * sqrt(abs(x))", legend=:topleft) 
#function is steep close to the root -> f'(x) = Inf -> slide 41: new step = old step

dfdx(x) = 1/(2*sqrt(abs(x)))

x = 1.0
x = newton(f,dfdx,x) #never ending cycle 1.0 -1.0, would be easily findable with bisection

myscatter = scatter(;xlim = [-2,2])
plot!(f, -2, 2, label="f(x) = sign(x) * sqrt(abs(x))", legend=:topleft)
animation = @animate for (ind,point) in enumerate(x)
    scatter!(myscatter,[point], [f(point)], alpha =  ind/length(x),ms = 5 + 10*ind/length(x),color = :blue,lab="")
end

gif(animation,fps = 5) ##never ending cycle 1.0 -1.0

###END OF NOT VERY IMPORTANT: MANUAL FUNCTIONS, WHICH ARE NOT VERY SMART ################################
