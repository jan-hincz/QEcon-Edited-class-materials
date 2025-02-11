using Distributions, LinearAlgebra,Plots, Random, QuantEcon

#TAKEAWAYS
#markov chains allow to model heterogeneous agents
#slide 19: Tauchen vs. Rouwenhorst method (AR(1) discretization)


## Seting a random seed for reproducibility
Random.seed!(1111)

## Define a function that simulates a Markov chain (time series of states)
function mc_sample_path(P; init_x = 1, sample_size = 100) #100 states coded 1 to N
    @assert size(P)[1] == size(P)[2] # square matrix required, if not: AssertionError
    N = size(P)[1] # number of states -> N x N matrix

    # Translate rows of transition matrix P into a vector of distributions of discrete RV:
    #slide 5: entries of each row have to sum up to 1 (P as vector of vectors - each row is a single distribution)

    dists = [Categorical(P[i, :]) for i in 1:N] #each row will have categorical distr. with probabilities given by P entries
    
    # Setup the simulation
    X = Vector{Int64}(undef, sample_size) # allocate memory without initializing values
    #X is array storing sequence of states in the Markov chain
    X[1] = init_x # set the initial state; states coded from 1 to N

    for t in 2:sample_size
        previous_state_value = X[t-1] # Get the previous state (value from 1 to N)
        P_Xt = dists[previous_state_value] # Get categorical distr. corresponding to the previous state (row of P from 1 to N) 
        X[t] = rand(P_Xt) #picks a current state acc to probabilities given by the row of P corr. to previous state
    end
    return X
end


## Recall the Categorical function from the Distributions package:
dice = Categorical([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]); # 6 discrete states and their probabilities
#Categorical(prob of 1, prob of 2,..., prob of N (here: N = 6))
@show dice
## roll a dice 3 times!
@show rand(dice, 3);


#### EXAMPLE 1: Markov chain - unemployed/employed agent (slide 4)
α = 0.1;   # an unemployed finds job  
β = 0.05;  # an employed loses job

P = [1-α α;
β 1-β]


periods = 40
sample_path_initU = mc_sample_path(P, init_x = 1, sample_size = periods); #initially unemp
sample_path_initE = mc_sample_path(P, init_x = 2, sample_size = periods); #initially emp

plot(sample_path_initU, label = "start unemployed") #simulation for an unemployed person (with specified seed)
plot!(sample_path_initE, label = "start employed") #simulation for an employed person (with specified seed)


## How does the distribution of employed/unemployed agents evolve over time? 

ψ0 = [0.05 0.95] # let this be the initial distribution  
t = 200 # path length

## Allocate memory
U_vals = zeros(t) #will store unemployment rates
E_vals = similar(U_vals) #U_vals is Vector{Float64} -> E_vals will be too, filled with random numbers
U_vals[1] = ψ0[1] #first unemp rate given by initial distribution
E_vals[1] = ψ0[2] #initial employment rate

for i in 2:t #getting chains of unemployment and employment rates
    ψ = [U_vals[i-1] E_vals[i-1]] * P #slide 8: update the distribution
    U_vals[i] = ψ[1]
    E_vals[i] = ψ[2]

end

plt = scatter(U_vals,E_vals, xlim = [0, 1], ylim = [0, 1], label = false) #you can see some steady state after a while
plot!(xlabel="Unemployement rate", ylabel="Employment rate", title="Markov chain: Employment dynamics")


## get stationary distribution
## iterative approach (slide 13):
ψs = ψ0*P^1000 #1000 very large 


#### EXAMPLE 2: Markov chain Hamilton 2005 (slide 5)

P = [0.971 0.029 0; 0.145 0.778 0.077; 0 0.508 0.492] # normal growth, mild recession, severe recession
P12 = P^12 # prob of transition in one year (12 months)

## Let's do the simulation:
periods = 12 * 25 #300 months = 25 years
sample_path_initSR = mc_sample_path(P, init_x = 3, sample_size = periods); #starting at severe recession
gdp_growth = [3.0,1.0,-2.0]; # normal growth, mild recession, severe recession

time_series = gdp_growth[sample_path_initSR]; #states coded as 1 to 3 -> translating states (indices) to gdp_growth

plot(time_series,xlabel = "time",ylabel = "annualized growth rate", label=false)


#### APPROXIMATION (slides 14-19)
#### Approximation of AR(1) process using the Tauchen method [QuantEcon package]
## x_{t+1} = ρ⋅x_t + ε_t
## ε_t ~ N(0, σ^2)
## Let:
ρ = 0.9 #slide 14: with lower - process shrinks faster -> lower std(x_t) 
#slide 16: x_1 = - m*std(x_t); x_N = m*std(x_t) -> with lower ρ, smaller grid (distance between x_1 and x_N) needed
σ = 0.02 #slide 14: std(ε_t)
N_states = 5 #slide 16 -> 5 values of X_t in discretised grid
tauch_approximation_1 = tauchen(N_states,ρ, σ) #QuantEcon function; default m = n_std = 3 (slide 19); default μ = 0 (slide 14) 
tauch_approximation_1.p #slide 18: transition matrix P
tauch_approximation_1.state_values #range syntax; 5 values of X
state_space_1 = collect(tauch_approximation_1.state_values) #vector with 5 elements of the above range

#### Changing the m (n_std) from default 3 to 5
ρ = 0.9
σ = 0.02
N_states = 5
μ = 0 #default value (slide 14)
n_std = 5 #m; before: default 3
tauch_approximation_2 = tauchen(N_states,ρ, σ,μ,n_std)

tauch_approximation_2.p #new transition matrix P
state_space_2 = collect(tauch_approximation_2.state_values) #bigger grid (same N=5 elements, higher amplitude)



#### Approximation of AR(1) process using the Rouwenhorst method (slide 19: pros of it vs. Tauchen)
ρ = 0.9
σ = 0.02
N_states = 5
μ = 0
rouw_approximation  = rouwenhorst(N_states,ρ, σ,μ) #QuantEcon package function

rouw_approximation.p
state_space_2 = collect(rouw_approximation.state_values)