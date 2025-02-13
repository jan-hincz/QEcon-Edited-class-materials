# Option pricing

# In this example we study **American call options**
# They provide the right to buy a given asset at any time during some specified period at a strike price K


# The market price of the asset at time t is denoted by p_t

# Let p_t =  rho*p_(t-1) + b + ν*ϵ_t (nu * epsilon); ϵ_t ~ N(0,1) := AR(1) with drift and random normal shock (8.2, slide 14)
#Markovian process will come from evolution of p

# this is a pretty bad assumption, because it implies that price changes are easily predictable - expected value stays constant, variance stationary if -1 < rho < 1

# The discount rate is β = 1/(1+r), where r>0 is a risk-free interest rate

# Upon exercising the option, the reward is equal to p_t - K

# The option is purchased at time t = 1 and can be exercised until t=T

# Our task is to find the price of the option v(p,t) (this is not nu, but "vee"):
#it satisfies Bellman equ v(p,t) = max{p - K; β * E[v{p',t+1]} with the boundary condition v(p,T+1)=0 
#after T option is worthless
# This is a **finite horizon** problem.


# load some packages we will need today
using Distributions, QuantEcon, IterTools, Plots

function create_option_model(; T=200, # periods
    ρ=0.95, # price persistence (rho; high)
    ν=10, # nu := price volatility
    b=6.0, #drift (AR(1) with drift and Normal ϵ shock)
    K=85, # strike price
    β=0.99, # discount factor
    N=25) # grid size for Tauchen
    mc = tauchen(N, ρ, ν, b) #markov chain (8.2, slide 14); QuantEcon package
    return (; T, ρ, ν, b, K, β, N, mc)
end

function T_operator(v,model)
    (;T, ρ, ν, b, K, β, N, mc) = model # Extract parameters from the model
    P = mc.p # Transition probability matrix (N × N) (row i: transition probs from p_i)
    p_vec = mc.state_values # Vector of possible asset prices (length N)
    σ_new        = [(p - K) >= (β * P[i,:]' * v) for (i, p) in enumerate(p_vec)] #1 if exercise, 0 otherwise
    #p - K compared to expected continuation value; slide 6; 8.2 - slide 5
    v_new        = σ_new .* (p_vec .- K) .+ (1 .- σ_new) .* (β * P * v);
    return v_new, σ_new #σ = 1 if I exercise, 0 if not
end

function vfi(model)
    (;T, ρ, ν, b, K, β, N, mc) = model
    
    v_matrix = zeros(N,T+1); σ_matrix = zeros(N,T) #v(p,t) matrix: N prices, T+1 periods
    #last possibly optimal exercise at T, because after T option is worthless
    for t=T:-1:1 # backward induction from t = T until t = 1
        v_matrix[:,t], σ_matrix[:,t]  = T_operator(v_matrix[:,t+1],model) #T_operator() defined above; backward induction from t+1 to t
    end
    return v_matrix, σ_matrix
end

model = create_option_model()
v_matrix,σ_matrix = vfi(model)
model.mc.p #N x N (N = # of possible prices) transition probs matrix P

contour(σ_matrix, levels =1, fill=true,legend = false, cbar=false, xlabel="Time", ylabel="Asset price", title="Policy")
#black: don't exercise, yellow: exercise; asset price not in usd, # means index of price (increasing with price)
model.mc.state_values[8] #8th price corresponds to 79.96
contour(v_matrix,levels = 25, cbar=false,clabels=true, xlabel="Time", ylabel="Asset price", title="Option price")
#decreasing option price with time

function sim_option(model, σ_matrix; init = 1)
    (;T, ρ, ν, b, K, β, N, mc) = model
    p_ind = simulate_indices(mc, T, init = init);
    #QuantEcon f simulating a Markov chain path (indices of states) 
    p = mc.state_values[p_ind] #state (asset price) drawn by MC (index state) simulation
    strike = zeros(T)
    for t=1:T
        strike[t] = σ_matrix[p_ind[t],t] #σ_matrix N by T; 1 if exercise, 0 if not
    end
    return p, strike
end

p, strike = sim_option(model, σ_matrix; init = 1)

strike_time = findfirst(strike.==1) #when to exercise (σ = 1)
plot(p, label="Asset price", legend=:topleft)
scatter!([strike_time],[p[strike_time]], label="Exercise time", legend=:topleft)


stationary_distributions(model.mc)[1] #QuantEcon f computing stationary distribution of Markov chain 
#probability of each state (price)


#prob of exercising at t under Markov chain stationary distribution
T = model.T
prob_strike = zeros(T) #will store CMF(t) of exercising
distr_strike = zeros(T) ##will store PMF(t) of exercising
for t = 1:T
    prob_strike[t] = sum( σ_matrix[i,t] * stationary_distributions(model.mc)[1][i] for i=1:model.N)
#prob. of exercising up to t (that's why sum()); have to account for optimal decision given different prices (i) and their probs in stationary distr
    if t > 1
    distr_strike[t] = (1-sum(distr_strike[1:t-1])) * prob_strike[t] #going from CMF to PMF
    else distr_strike[t] = prob_strike[t]
    end
end
plot(1:T,prob_strike, label="Cumulative probability of exercise", legend=:topleft) #CMF
prob_strike[T] #86% prob we'll exercise at all

plot(1:T,distr_strike, label="Distribution of exercise time", legend=:topleft) #PMF