# Examples from/based on Fundamentals of Numerical Computation, Julia Edition. Tobin A. Driscoll and Richard J. Braun

using PrettyTables, Plots, LaTeXStrings, LinearAlgebra, NLsolve, Roots
#Roots for a univariate function, NLsolve more general - use it for more complicated tasks



# example using NLsolve 

f(x) = x.^2 .- 3 .+ x .* sin.( 1 ./ x .+ x .^ 2 )

plot(f, -30, 30, label="f(x) =  x^2 - 3 + x * sin(1/x + x^2 )", legend=:topleft)
plot!(zero, -30, 30, label="y=0")

#IT'S NOT BISECTION (WE ONLY CHOSE one point 2.5)
guess = 20.5 #look at the plot; it will search close to it
nlsolve(f,[guess],ftol=1e-14,show_trace=true) #stop when |f(x)| <= ftol=1e-14
#ALWAYS VERIFY THAT THIS IS THE CASE: "Converge: true"; zero: 2.19972..

nlsolve(f,[guess],method=:newton,ftol=1e-14,show_trace=true)

# example using Roots (bisection; we see it's between 1 and 3)

find_zero(f, (0.1,3), Bisection(), verbose = true, atol = 1e-14) #2.199723519572541 like before
#but 51 iterations instead of 5 like above
#it won't work for any a and b (e.g. -300,300), because the initial function might not be computable by Julia

find_zero(f, (1.4,1.5), Order1(), verbose = true)
find_zero(f, 1.4, verbose = true)



# bisection method - MANUAL FUNCTION, NOT VERY SMART, E.G. NO CAP ON # OF ITERATIONS
function bisection(f,a,b,tolerance) #tolerance here related to length of the interval b-a, not value of the function
    b > a || error("b must be greater than a")
    f(a)*f(b) > 0 && error("f(a) and f(b) must have opposite signs")
    while abs(b-a) > tolerance
    c = (a+b)/2
    sign(f(c)) == sign(f(a)) ? a = c : b = c #recall: 1-line conditional statements
    #if true -> a = c, if false -> b = c
    end
    return (a+b)/2
end


bisection(f,1,3,1e-14) #2.19972351957254


# newton's method

f(x) = x*exp(x) - 2;
dfdx(x) = exp(x)*(x+1);
r = nlsolve(x -> f(x[1]),[1.]).zero

plot(f, -1, 1, label="f(x) = x*exp(x) - 2", legend=:topleft)

x = [BigFloat(10);zeros(7)] #BigFloat will allow better than double precision
for k = 1:7
    x[k+1] = x[k] - f(x[k]) / dfdx(x[k]) #from slide 39
end
r = x[end] #so long,because we used BigFloat

ϵ = @. Float64(x[1:end-1] - r) #we can see the number of precise digits goes up by 2 (error gets smaller)
logerr = @. log(abs(ϵ))
[ logerr[i+1]/logerr[i] for i in 1:length(logerr)-1 ] # p = 2 
#we can see it converges to 2.sth, but NEWTON DOESN'T ALWAYS CONVERGE

### implement newton's method - version of loop above with stopping algo (so that it doesn't iterate infinitely if no conversion)
function newton(f,dfdx,x₁;maxiter=40,ftol=100*eps(),xtol=100*eps()) #after ; -> default values; eps = machine epsilon
    x = [float(x₁)]
    y = f(x₁)
    Δx = Inf   # for initial pass below, measures distance between 2 consecutive iterations
    k = 1 #counts iterations

    while (abs(Δx) > xtol) && (abs(y) > ftol)
        dydx = dfdx(x[k])
        Δx = -y/dydx            # Newton step; slide 39
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

x = newton(f,dfdx,1.0) #won't converge for every number

myscatter = scatter(;xlim = [0.825,1.1],ylim = [-0.25,3])
plot!(f, 0.825, 1.1, label="f(x) = x*exp(x) - 2", legend=:topleft)
animation = @animate for (ind,point) in enumerate(x)
    scatter!(myscatter,[point], [f(point)], alpha = 1 - ind/length(x),ms = 5 + 10*ind/length(x),color = :blue,lab="")
end


gif(animation,fps = 5)


### babylonian method - ANCIENT METHOD
f(x) = x^2 - 2
dfdx(x) = 2x
x = 1.0
x = newton(f,dfdx,x) #gets very close to root of 2

ϵ = @. Float64(x[1:end-1] - sqrt(2))


### bad case 

f(x) = sign(x) * sqrt(abs(x))
plot(f, -1, 1, label="f(x) = sign(x) * sqrt(abs(x))", legend=:topleft) #function is steep close to root -> f'(x) = Inf -> slide 39: new step = old step

dfdx(x) = 1/(2*sqrt(abs(x)))

x = 1.0
x = newton(f,dfdx,x) #never ending cycle 1.0 -1.0, would be easily findable with bisection

myscatter = scatter(;xlim = [-2,2])
plot!(f, -2, 2, label="f(x) = sign(x) * sqrt(abs(x))", legend=:topleft)
animation = @animate for (ind,point) in enumerate(x)
    scatter!(myscatter,[point], [f(point)], alpha =  ind/length(x),ms = 5 + 10*ind/length(x),color = :blue,lab="")
end

gif(animation,fps = 5)

# what happens here??? 

#1. GENERAL LESSON: NO GUARANTEE A NON-LINEAR JULIA FUNCTION WILL FIND ANY ROOTS
#2. ALWAYS VERIFY COVERGENCE: TRUE
#3. NO GUARANTEE JULIA WILL FIND ALL ROOTS
#4. TRY MANY STARTING POINTS