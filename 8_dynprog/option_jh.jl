# ## Option pricing

# In this example we study **American call options**.
# They provide the right to buy a given asset at any time during some specified period at a strike price K. 


# The market price of the asset at time t is denoted by $p_t$.

# Let $$p_t = \rho p_{t-1} + b + \nu \epsilon_t \quad \text{with } \epsilon_t \sim N(0,1).$$ #AR(1) with drift and random normal shock
#Markovian process will come from evolution of p

# (this is a pretty bad assumption, because it implies that price changes are easily predictable - expected value always, variance stationary if -1 < rho < 1)

# The discount rate is $\beta = \frac{1}{1+r}$, where $r>0$ is a risk-free interest rate.

# Upon exercising the option, the reward is equal to $p_t - K$.

# The option is purchased at time $t=1$ and can be exercised until $t=T$.

# Our task is find the price of the option $v(p,t)$. It satisfies Bellman equ $$v(p,t) = \max \left\{p - K, \; \beta E_{p} \left[ v(p^\prime,t+1) \right]  \right\}$$ with the boundary condition $$v(p,T+1)=0.$$
#after T option is worthless
# This is a **finite horizon** problem.

# load some packages we will need today
using Distributions, QuantEcon, IterTools, Plots

function create_option_model(; T=200, # periods
    ρ=0.95, # price persistence (high)
    ν=10, # price volatility
    b=6.0, #drift
    K=85, # strike price
    β=0.99, # discount factor
    N=25) # grid size for Tauchen
    mc = tauchen(N, ρ, ν, b) #markov chain; QuantEcon package
    return (; T, ρ, ν, b, K, β, N, mc)
end

function T_operator(v,model)
    (;T, ρ, ν, b, K, β, N, mc) = model
    P = mc.p
    p_vec = mc.state_values #
    σ_new        = [(p - K) >= (β * P[i,:]' * v) for (i, p) in enumerate(p_vec)]
    v_new        = σ_new .* (p_vec .- K) .+ (1 .- σ_new) .* (β * P * v);
    return v_new, σ_new #σ = 1 if I exercise, 0 if not
end

function vfi(model)
    (;T, ρ, ν, b, K, β, N, mc) = model
    
    v_matrix = zeros(N,T+1); σ_matrix = zeros(N,T)
    for t=T:-1:1 # backwards induction
        v_matrix[:,t], σ_matrix[:,t],  = T_operator(v_matrix[:,t+1],model)
    end
    return v_matrix, σ_matrix
end

model = create_option_model()
v_matrix,σ_matrix = vfi(model)
model.mc.p

contour(σ_matrix, levels =1, fill=true,legend = false, cbar=false, xlabel="Time", ylabel="Asset price", title="Policy")
#black: don't exercise, yellow: exercise, asset price not in usd, index of price (increasing with price)
model.mc.state_values[8] #8th price corresponds to 79.96
contour(v_matrix,levels = 25, cbar=false,clabels=true, xlabel="Time", ylabel="Asset price", title="Option price")
#decreasing option price with time

function sim_option(model, σ_matrix; init = 1)
    (;T, ρ, ν, b, K, β, N, mc) = model
    p_ind = simulate_indices(mc, T, init = init);
    p = mc.state_values[p_ind]
    strike = zeros(T)
    for t=1:T
        strike[t] = σ_matrix[p_ind[t],t]
    end
    return p, strike
end

p, strike = sim_option(model, σ_matrix; init = 1)

strike_time = findfirst(strike.==1)
plot(p, label="Asset price", legend=:topleft)
scatter!([strike_time],[p[strike_time]], label="Exercise time", legend=:topleft)


stationary_distributions(model.mc)[1]

T = model.T
prob_strike = zeros(T)
distr_strike = zeros(T)
for t = 1:T
    prob_strike[t] = sum( σ_matrix[i,t] * stationary_distributions(model.mc)[1][i] for i=1:model.N)
    if t > 1
    distr_strike[t] = (1-sum(distr_strike[1:t-1])) * prob_strike[t]
    else distr_strike[t] = prob_strike[t]
    end
end
plot(1:T,prob_strike, label="Cumulative probability of exercise", legend=:topleft)
prob_strike[T] #86% we'll exercise at all

plot(1:T,distr_strike, label="Distribution of exercise time", legend=:topleft) #pdf