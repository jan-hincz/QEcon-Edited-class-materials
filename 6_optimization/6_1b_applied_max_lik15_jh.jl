## VERY SIMILAR TO 6_1a file, Remember to add necessary packages to the environment:
using Plots,NLopt,Statistics,Distributions,ForwardDiff,DelimitedFiles

## Provide underlying p
p = 0.2
DGP = Bernoulli(p) #Data generating process
## One draw from the Bernoulli distribution:
rand(DGP,1) #0 or 1
## 100 draws from the Bernoulli distribution:
y_data = rand(DGP,100)
histogram(y_data,bins=-0.05:0.1:1.05,ylabel="Frequency",xlabel="y",label="100 draws from Bernoulli")


function nlopt_objective_fn(params::Vector, grad::Vector,y) ## y will be the data supplied
    ## Params: Vector of parameters (just p in this case)
    ## Grad: Vector storing the gradient of the function
    ## y: The data supplied

    if length(grad) > 0
        grad[1] =  sum(y)/params[1] - sum(1 .- y)/(1 - params[1]) #slide 4: gradient
    end
    obj = log(params[1])*sum(y) + log(1 -params[1])*sum((1 .- y)) #LogLik - slide 4 (green-highlighted)
    println("Params, Function, Gradient: ",round(params[1],digits=7),", ",round(obj,digits=7),", ",round(grad[1],digits=7)) 
    return obj
end

## Optimization procedure:
## Define the optimizer used: 
opt = NLopt.Opt(:LD_MMA, 1) # algorithm and dimensionality
## See: https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/

## Define the bounds: 
opt.lower_bounds = [ 0.0]   # lower bound for params
opt.upper_bounds = [ 1.0]   # upper bound for params

## NLopt will terminate when the first one of the specified termination conditions is met
## Tolerance on the on the function values. The algorithm stops if from one iteration to the next:
#opt.ftol_rel    = 0.0001  # |Δf|/|f|  < tol_rel 
opt.ftol_abs     = 0.0000001   # |Δf|      < tol_abs

### Tolerance on the parameters. The algorithm stops if from one iteration to the next:
#opt.xtol_rel    = 0.0001  # |Δx|/|x|  < tol_rel 
opt.xtol_abs    = 0.0001  # |Δx|      < tol_abs

#Note: tol_rel is independent of any absolute scale factors or units

## Or you can specify the maximum number of evaluations:
opt.maxeval = 200

## Supply opt. with the function to be maximized
## NOTE: supply only a function of (params, grad), that is why I use a wrapper function!
opt.max_objective = (params,grad)->nlopt_objective_fn(params, grad,y_data)
## Run the optimization, provide an initial guess
max_f, max_param, ret = optimize(opt, [0.00001]) #[0.00001] as initial guess for vector of params (just p);   

## max_f        = value of the function at the maximum
## max_param    =  value of the parameters at the maximum (p in this case);  p = 0.199982 : very close to underlying p = 0.2
## ret          = stopping criteria used 
println("Stopping criteria used: ", ret) #XTOL_REACHED
println("Number of evaluations: ", opt.numevals) #13


#############################        LOGIT EXAMPLE (slide 8):         ############################ 
F_bad(x)    = exp(x)/(1+exp(x))
F_bad(2) 
F_bad(2999) # an issue! NaN: see slide 8
F_good(x)   = 1/(1/(exp(x))+1)
F_good(2)
F_good(2999)     # 1; no issue!
plot(F_good,-10,10,lw=3,label="The probability of 1")

## Prepare the function to be used in log-likelihood: CDF of logistic distribution
function F(input_val;probability_of=1)
    if probability_of == 1
        return 1/(1/exp(input_val)+1)!=0 ? 1/(1/exp(input_val)+1) : eps(0.0)
    else
        return 1/(exp(input_val)+1)!=0 ? 1/(exp(input_val)+1) : eps(0.0)
    end
end

## The probability of 1 if the input value is 2
F(2;probability_of=1) #(e^2)/(1+e^2): see slide 8
## The probability of 0 if the input value is 2
F(2;probability_of=0) #1/(1+e^2)

## The probability of 1 if the input value is 10
F(10;probability_of=1)
## The probability of 0 if the input value is 10
F(10;probability_of=0)

plot(input->F(input;probability_of=1),-10,10,lw=3,label="The probability of 1")
plot!(input->F(input;probability_of=0),-10,10,lw=3,label="The probability of 0")


## Suppose this is our vector of data
y_vec   = [1,1,1,0,1,0]
x_vec   = [-10,-1,2,-3,40,50] #x below on the plot

plot(x_vec,y_vec,seriestype=:scatter,legend=false,xlabel="x",ylabel="Binary outcome")
correlation = cor(x_vec,y_vec) #-0.321


## Let's assume that:
β0 = 0.5
β1 = 0.1

## Let's calculate the log likelihood for the first observation:
y_vec[1] #1
x_vec[1] #-10

## The log likelihood for the first observation: #see slide 12
y_vec[1]*log(F(β0+β1*x_vec[1];probability_of=1)) + (1-y_vec[1])*log(F(β0+β1*x_vec[1];probability_of=0)) #-0.974
#F defined at line ca. 72

## The log likelihood for the second observation:
y_vec[2]*log(F(β0+β1*x_vec[2];probability_of=1)) + (1-y_vec[2])*log(F(β0+β1*x_vec[2];probability_of=0)) #-0.513

## The log likelihood for the "First part" (see slide 12) - the one for (prob that y = 1):
y_vec .* log.(F.(β0.+β1.*x_vec;probability_of=1)) #vector - you can get to 1st part of log lik in one go

## Note the annoying ".". The easier way (broadcasting macro @.):
@. y_vec*log(F(β0+β1*x_vec;probability_of=1)) #vector

## The "second part" of the log likelihood (see slides):
@. (1-y_vec)*log(F(β0+β1*x_vec;probability_of=0)) #vector

## The log likelihood for all observations:
@. y_vec*log(F(β0+β1*x_vec;probability_of=1)) + (1-y_vec)*log(F(β0+β1*x_vec;probability_of=0)) #vector for all 6 obs
sum(@. y_vec*log(F(β0+β1*x_vec;probability_of=1)) + (1-y_vec)*log(F(β0+β1*x_vec;probability_of=0))) #-sum of individual obs = LogLik = -8.2035
## And you can simply sum this vector up using the sum() function:

################# Concept check! #################
## Fill in this Logit_LogLik a function: 
function Logit_LogLik(params::Vector,y,x) 
    #####INPUTS TO THE FUNCTION:#####
    ## params: a vector of two parameters β0,β1
    ### params[1]: the first parameter (β0 on slides)
    ### params[2]: the second parameter (β1 on slides)
    ## y: a vector of binary {0,1} outcomes
    ## x: a vector x variable
    
    First_part  = @. y*log(F(params[1]+params[2]*x;probability_of=1)) 
    #vs. lines ca. 128-9: changed y_vec and x_vec to y and vec and β0,β1 to params[1] and params[2] - look at function syntax (line ca. 134)
    Second_part = @. (1-y)*log(F(params[1]+params[2]*x;probability_of=0))
    Sum         = sum(First_part .+ Second_part)
    return Sum
end

beta_vector =[β0,β1]
Logit_LogLik(beta_vector,y_vec,x_vec) #inputs defined above; -8.203543342966274

##################################################

## Plot the log likelihood function in 3D:
n = 100
animation = @animate for i in range(0, stop = 2π, length = n)

    surface(0.5:0.1:1.5, -0.1:0.01:0.01, (x,y) -> Logit_LogLik([x,y],y_vec,x_vec), st=:surface, c=:blues, legend=false,camera = (30 * (1 + cos(i)), 40));
end
gif(animation,fps = 50)


## Define the NLopt objective function for the Logit:
function nlopt_fn(params::Vector, grad::Vector,y,x)
    function Logit_LogLik(params::Vector,y,x) #like above, just one line
        Sum = sum(@. y*log(F(params[1]+params[2]*x;probability_of=1)) + (1-y)*log(F(params[1]+params[2]*x;probability_of=0)))
        return Sum
    end
    if length(grad) > 0
        ## Here we use the ForwardDiff package to calculate the gradient - automatic differentiation
        grad .=  ForwardDiff.gradient(temp_params->Logit_LogLik(temp_params,y,x), params)
    end
    obj = Logit_LogLik(params,y,x)
    println("Params, Function, Gradient: ",round.(params,digits=5),", ",round(obj,digits=5),", ",round.(grad,digits=5)) 
    return obj 
end

## Define the optimizer used:
opt = NLopt.Opt(:LD_MMA, 2) #2 dimensions: β0,β1
## Define the objective function:
NLopt.max_objective!(opt, (params,grad)->nlopt_fn(params, grad,y_vec,x_vec)) #y_vec, x_vec as inputs, see nlopt_fn above
## Define the lower bounds for the two parameters:
opt.lower_bounds = [-15,-15] 
## Define the upper bounds for the two parameters:
opt.upper_bounds = [15,15]   
## Define the stopping criteria:
opt.maxeval      = 2000
opt.xtol_rel     = 1e-10     
## Perform optimization on the object defined and the initial guess:
max_f, max_param, ret = NLopt.optimize(opt, [0.1, 0.1]) #initial guess: [β0,β1] = [0.1, 0.1] 
# [β0,β1] = [1.13715, -0.02923]

## Calculate the standard errors:
hess = ForwardDiff.hessian(max_param -> Logit_LogLik(max_param,y_vec,x_vec), max_param) #2nd der of LogLik wrt β0,β1 
std_err_β0 = sqrt(-inv(hess)[1,1]) #1.12; equivalent to slide 9 of 6_1a (6_1a; line ca. 212)
std_err_β1 = sqrt(-inv(hess)[2,2]) #0.038


############################## Concept check! ##############################
## Suppose you were asked to model the relationship between the 
## probability of getting to college and the (average) parents education. 
############################################################################

## Read in the data:
## Warning you may need to change the path to the data!
data = readdlm("6_optimization\\data\\Admit_edParents.csv",',')


## The first column of data is the college admission {0,1}: y
y_data = data[:,1] #slide 6

## The second column is the average parents education: x
x_data = data[:,2] #slide 6


## Plot the data:
college  = plot(x_data,y_data,seriestype=:scatter,legend=false,xlabel="Average parents education",ylabel="College admission")
title!("College admission vs parents education")


### ESTIMATE THE MODEL! ###
## Use the above algorithm to estimate the parameters of the logit model!
## You can copy most parts of the code, but make sure it works.
## NLopt objective function nlopt_fn() for the Logit is already defined above (line ca. 164):


## Define the optimizer used:
opt = NLopt.Opt(:LD_MMA, 2) #2 dimensions: β0,β1
## Define the objective function:
NLopt.max_objective!(opt, (params,grad)->nlopt_fn(params, grad,y_data,x_data)) #y_data, x_data as inputs
## Define the lower bounds for the two parameters:
opt.lower_bounds = [-15,-15]
## Define the upper bounds for the two parameters:
opt.upper_bounds = [15,15] 
## Define the stopping criteria:
opt.maxeval      = 2000
opt.xtol_rel     = 1e-10   
## Perform optimization on the object defined and the initial guess:
max_f, max_param, ret = NLopt.optimize(opt, [0.1, 0.1]) #[β0,β1] = [-12.30053, 3.90963]


hess_2 = ForwardDiff.hessian(max_param -> Logit_LogLik(max_param,y_data,x_data), max_param)
std_err_2_β0 = sqrt(-inv(hess_2)[1,1]) #0.73
std_err_2_β1 = sqrt(-inv(hess_2)[2,2]) #0.238

Z_2_β0 = max_param[1]/std_err_2_β0 #Z-statistic = -16.81575 (Z not t, because MLE, not OLS)
Z_2_β1 = max_param[2]/std_err_2_β1 # 16.405