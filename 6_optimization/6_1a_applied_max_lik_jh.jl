## Remember to add necessary packages to the environment:
using Plots,NLopt,Statistics,Distributions,ForwardDiff

#TAKEAWAYS

# Optimization libraries (like NLopt) usually put strict requirements on the object that is optimized:
## Both Params and Grad have to be vectors! (line ca. 42-3)
## See: https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
#line ca. 82-3: optimize() function is here from NLopt, not Optim - don't use both packages at the same to avoid error
#line ca. 172-4: @. := broadcasting macro - broadcasts every operation

## Define a data vector: 
y_data = [1,1,1,0,1,0]
## Define the likelihood function for the Bernoulli distribution (slide 4):
function Bern_LogLik(p,y)
    return sum(y)*log(p) + sum((1 .- y))*log(1 -p)
end

## Where is the maximum?
Bern_LogLik(0.65,y_data)
Bern_LogLik(4/6,y_data) #Likelihood maximised (slide 3)
Bern_LogLik(0.67,y_data)

## Plot the likelihood function
plot(p -> Bern_LogLik(p,y_data),0,1,lw=3,label="Log Likelihood of Bernoulli") #x-axis (p) scale from 0 to 1
vline!([4/6],lw=3,label="MLE")

plot(p -> Bern_LogLik(p,y_data),0.1,0.9,lw=3,label="Log Likelihood of Bernoulli") #x-axis (p) scale from 0.1 to 0.9
vline!([4/6],lw=3,label="MLE")


## OPTIMIZATION:
## Optimization libraries (like NLopt) usually put strict requirements on the object that is optimized:
#p = 0.4
#DGP = Bernoulli(p)
#y_data = rand(DGP,100)
#histogram(y_data,bins=-0.05:0.1:1.05,ylabel="Frequency",xlabel="y",label="Data")
#mean(y_data)

## Params: Vector of parameters (just p in this case)
## Grad: Will store the gradient of the function at the point params
## Both Params and Grad have to be vectors!
function nlopt_objective_fn(params::Vector, grad::Vector,y) ## y will be the data supplied
    if length(grad) > 0
        grad[1] =  sum(y)/params[1] - sum(1 .- y)/(1 - params[1])
    end #we define gradient (slide 4), 1 parameter -> scalar
    obj = log(params[1])*sum(y) + log(1 -params[1])*sum((1 .- y)) #Likelihood function: green-highlighted on slide 4
    println("Params, Function, Gradient: ",round(params[1],digits=5),", ",round(obj,digits=5),", ",round(grad[1],digits=5)) 
    return obj
end

## Define properties of the optimizer
opt = NLopt.Opt(:LD_MMA, 1) #package NLopt; algo and dimensionality (1 dimension - we only search for p)
#:LD_MMA := Method of Moving Asymptotes, gradient-based algo for non-linear constrained optimization
## See: https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
opt.lower_bounds = [ 0.0]   # lower bound for params
opt.upper_bounds = [ 1.0]   # upper bound for params

## NLopt will terminate when the first one of the specified termination conditions is met
## Tolerance on the on the function values. The algorithm stops if from one iteration to the next:
#opt.ftol_rel    = 0.0001  # |Δf|/|f|  < tol_rel 
opt.ftol_abs     = 0.001   # |Δf|      < tol_abs

### Tolerance on the parameters. The algorithm stops if from one iteration to the next:
#opt.xtol_rel    = 0.0001  # |Δx|/|x|  < tol_rel 
#opt.xtol_abs    = 0.0001  # |Δx|      < tol_abs

#Note: tol_rel is independent of any absolute scale factors or units

## Or you can specify the maximum number of evaluations:
#opt.maxeval = 2000 - if no conversion/error than it won't calculate indefinitely

## Supply opt with the function to be maximized
## NOTE: supply only a function of (params, grad), that is why I use a wrapper function!
opt.max_objective = (params,grad)->nlopt_objective_fn(params, grad,y_data)
#wrapper objective function for NLopt optimization, we input data and get otpimizing parameters (p), gradient 

# A wrapper function is simply:
example_fun = (x)->x^2
example_fun(5) #5^2 = 25
## Run the optimization, provide an initial guess
#optimize() function is here from NLopt, not Optim - don't use both packages at the same to avoide error
max_f, max_param, ret = optimize(opt, [0.6]) #0.666743 - close to the real one 2/3

## max_f        = value of the function at the maximum
## max_param    =  value of the parameters at the maximum (p in this case)
## ret          = stopping criteria used 
println("Stopping criteria used: ", ret) #FTOL_REACHED
println("Number of evaluations: ", opt.numevals) #6


#############################        CONCEPT CHECK:         ############################ 
# Maximize the log likelihood of a Poisson distribution! Follow the steps below!
########################################################################################  

## This is the data vector:
y_data = [2,0,1,2,2,2,0,2,1,1] #each y is a natural number
## Recall: length(y_data) will give you N
N = length(y_data)

## Define the log likelihood function for a Poisson distribution to be graphed:
function Poiss_LogLik(λ,y)
    return -N*λ +log(λ)*sum(y) # Fill in the log likelihood function here (slide 8 - green)
end #without the constant - factorial function of data (slide 8)
plot(λ -> Poiss_LogLik(λ,y_data),0,5,lw=3,label="Log Likelihood")

## Define the NLopt objective function for the Poisson distribution:
function nlopt_objective_fn(params::Vector, grad::Vector,y)
    println("The parameters are: ",params) 
    if length(grad) > 0
        ## Put here the gradient of the log likelihood function as a function of params[1] = lambda
        grad[1] = -N + sum(y_data)/params[1] #based on slide 8
        println("The gradient is:    ",grad) 
    end
    obj = -N*params[1] +log(params[1])*sum(y_data) #likelihood elements depending on lambda
    println("Params, Function, Gradient: ",round(params[1],digits=5),", ",round(obj,digits=5),", ",round(grad[1],digits=5)) 
    #no factorial part needed
    ## Return the log likelihood function as a function of params[1]
    return obj # Fill in the log likelihood function here
end


## Define properties of the optimizer
opt = NLopt.Opt(:LD_MMA, 1) # algorithm and dimensionality (1 parameter lambda)
opt.lower_bounds    = [ 0.0] # lower bound
opt.ftol_abs        = 0.0001 # tolerance
opt.maxeval         = 100

## Define the function to be maximized: 
opt.max_objective = (params,grad)->nlopt_objective_fn(params, grad,y_data) # Fill in the wrapper function here
max_f, max_param, ret   = optimize(opt, [1]) #optimization procedure, lambda_MLE = 1 as initial guess
#plot suggests optimum lambda will be close to 1
#1.29995
println("Stopping criteria used:", ret) #FTOL_REACHED 
println("Number of evaluations: ", opt.numevals) #8


## Calculate the standard errors: #see slide 9
function Poiss_LogLikHess(λ,y) #2nd derivative of LogLik wrt lambda (slide 8)
    return -sum(y)/(λ[1]^2)
end
hs = Poiss_LogLikHess(max_param,y_data) #using max_param - optimizing lambda 1.2995
se = sqrt(-1/hs) #slide 9


#############################        LOGIT EXAMPLE (slide 11):         ############################ 
F(x) = exp(x)/(1+exp(x)) #assume β0 = 0, β1 = 1
F(2999) # an issue! dividing big number by big number -> we got NaN
F(x) = 1/(1/(exp(x))+1)
F(2999) # no issue!
plot(F,-10,10,lw=3,label="The probability of 1")

## Suppose this is our vector of data
y_vec      = [1,1,1,0,1,0]
x_vec = [-10,-1,2,-3,40,50]

## Prepare the function to be used in log-likelihood
function F(x;probability_of=1)
    if probability_of == 1
        return 1/(1/exp(x)+1)!=0 ? 1/(1/exp(x)+1) : eps(0.0) 
    else #probability_of y = 0
        return 1/(exp(x)+1)!=0 ? 1/(exp(x)+1) : eps(0.0)
#if probability_of y = 0 is not 0 it returns 1/(exp(x)+1), otherwise it returns small number to avoid division by 0
    end
    
end
plot(x->F(x;probability_of=1),-10,10,lw=3,label="The probability of 1")
plot!(x->F(x;probability_of=0),-10,10,lw=3,label="The probability of 0")

## Log Likelihood function for Logit (slide 14):
function Logit_LogLik(params::Vector,y,x) 
    vec_1 = @. y*log(F(params[1]+params[2]*x;probability_of=1)) #F defined above, x turns into β0 + β1*x
    vec_0 = @. (1-y)*log(F(params[1]+params[2]*x;probability_of=0))
    return sum(vec_1 .+ vec_0) #@. := broadcasting macro - broadcasts every operation
end
## Plot the log likelihood function in 3D:
n = 100
animation = @animate for i in range(0, stop = 2π, length = n)

    surface(0.5:0.1:1.5, -0.1:0.01:0.01, (x,y) -> Logit_LogLik([x,y],y_vec,x_vec), st=:surface, c=:blues, legend=false,camera = (30 * (1 + cos(i)), 40));

end
gif(animation,fps = 50)

## Define the NLopt objective function for the Logit #similar to above, but we have to define grad as well
function nlopt_fn(params::Vector, grad::Vector,y,x)
    function Logit_LogLik(params::Vector,y,x) 
        return sum(@. y*log(F(params[1]+params[2]*x;probability_of=1)) + (1-y)*log(F(params[1]+params[2]*x;probability_of=0)))
    end
    if length(grad) > 0
        ## Here we use the ForwardDiff package to calculate the gradient
        grad .=  ForwardDiff.gradient(vec->Logit_LogLik(vec,y,x), params)
    end
    obj = Logit_LogLik(params,y,x)
    println("Params, Function, Gradient: ",round.(params,digits=5),", ",round(obj,digits=5),", ",round.(grad,digits=5)) 

    return obj 
end

opt = NLopt.Opt(:LD_MMA, 2) #2 parameters = dimensions: β0, β1
NLopt.max_objective!(opt, (params,grad)->nlopt_fn(params, grad,y_vec,x_vec))
opt.lower_bounds = [-5,-5] # lower bound
opt.upper_bounds = [15,15] # lower bound
opt.maxeval      = 200
opt.xtol_rel     = 1e-4     # tolerance
max_f, max_param, ret = NLopt.optimize(opt, [0.1, 0.1]) #initial guess: β0 = 0.1; β1 = 0.1
#β0 = 1.13721; β1 = -0.02924

## Calculate the standard errors:
hess = ForwardDiff.hessian(vec -> Logit_LogLik(vec,y_vec,x_vec), max_param) #2nd der of LogLik wrt parameters
#2 by 2 Hessian matrix -> we'll use diagonal entries to find standard errors of estimators
std_err_β0 = sqrt(-inv(hess)[1,1]) #like on slide 9; 1.1219
std_err_β1 = sqrt(-inv(hess)[2,2]) #like on slide 9; 0.038277