
## McCall Job Search model

#Code based on Sargent and Stachursky (2023) - https://github.com/QuantEcon/book-dp1
# load some packages we will need today
using Distributions, Plots #Distributions package useful for statistics and calculus

function create_job_search_model(;
    n=100, # wage grid size - number of possible wage offers
    w_min=10.0, # lowest wage
    w_max=60.0, # highest wage
    a=200, # wage distribution parameter
    b=100, # wage distribution parameter
    β=0.96, # discount factor
    c=10.0 # unemployment compensation
    )
    w_vals = collect(LinRange(w_min, w_max, n)) #n
    ϕ = pdf(BetaBinomial(n-1, a, b)) #n-1 vs. n
    return (; n, w_vals, ϕ, β, c)
end

my_model = create_job_search_model()
my_model.w_vals
plot(my_model.w_vals,my_model.ϕ) #pdf of wage offers

#########tutorial
my_dist = Binomial(10, 0.25)
my_pdfs = pdf(my_dist, 0.12)
my_cdfs = cdf(my_dist, 0)
#even convolutions of functions are possible (but you gotta remember syntax)


### APPROACH I: directly solve for h*

get_h_new(h_old ;n, w_vals, ϕ, β, c) = c + β * sum(ϕ[i] * max(h_old,w_vals[i]/(1-β)) for i in 1:n) #ϕ[i] = prob of a given wage
wrap_h_new(h_old) = get_h_new(h_old; my_model...) #...: unpacks an object contained in my model - passes arguments n, w_vals, ϕ etc (everything stored in it)

h_vec = LinRange(0, 2500, 100)
plot(h_vec,wrap_h_new.(h_vec), label="h_new(h_old)",linewidth=4,xlabel = "h_old",ylabel = "h_new");
plot!(h_vec,h_vec, label="45 degree line", linestyle=:dash,linewidth=2)

#iteratively solving to find h -> solution where h_old = h_new ()
#this is primitive approach I from slide 22, probably better to use NLsolve, Roots or other package to do that

function get_h_1(model;tol=1e-8,maxiter=1000,h_init=model.c/(1-model.β)) # do it by hand
    
    (; n, w_vals, ϕ, β, c) = model # unpack the model parameters

    h_old = Float64(h_init)
    h_new = h_old
    h_history = [h_old]
    error = tol + 1.0 
    iter = 1

    while error > tol && iter < maxiter
        h_new = get_h_new(h_old ;n, w_vals, ϕ, β, c) 
        error = abs(h_new - h_old) # think about this error criterion... why is it not great?
        h_old = h_new
        push!(h_history,h_old)
        iter += 1
    end

    return h_new, iter, error, h_history
end

h, iter, error, h_history = get_h_1(my_model)

function get_v_from_h(model,h) 
    
    (; n, w_vals, ϕ, β, c) = model # unpack the model parameters
    σ = w_vals ./ (1-β) .>= h # this is a vector of booleans; σ = 1 when it's optimal to accept, σ = 0 when it's optimal to reject
    v = σ .* w_vals ./ (1-β) + (1 .- σ) .* h # this is a vector of floats
    return v, σ 
end

v, σ = get_v_from_h(my_model,h)

plot_v = plot(my_model.w_vals,v, label="v(w)",linewidth=4,xlabel = "w",ylabel = "v")
plot_σ = plot(my_model.w_vals,σ, label="policy: 1 = accept wage",xlabel = "w", linestyle=:dash,linewidth=2)

plot(plot_v,plot_σ,layout=(1,2),legend=:topleft) #optimal to reject under #42.82 = reservation wage - see below

reservation_wage = my_model.w_vals[σ][1] #42.82

# study how reservation wage depends on unemployment compensation
c_vec = LinRange(0, 60, 100)
reservation_wage_vec = []
for c in c_vec
    my_model = create_job_search_model(;c=c)
    h, iter, error, h_history = get_h_1(my_model)
    v, σ = get_v_from_h(my_model,h)
    push!(reservation_wage_vec,my_model.w_vals[σ][1])
end

plot_reservation_wage = plot(c_vec,reservation_wage_vec, label=false,linewidth=4,xlabel = "unemp. compensation",ylabel = "reservation wage")
#if Beta is low, then i'm impatient so the option value of waiting for a higher wage will be not very valuable to makes
#I'll accept lower wages then


### APPROACH II: dynamic programming #THIS APPROACH MIGHT NOT BE THE FASTEST, BUT IT'S GUARANTEED TO WORK (WITHIN REASON)

function T(v,model) # Bellman operator: T (slide 41)
    (; n, w_vals, ϕ, β, c) = model # unpack the model parameters
    return [max(w/(1-β) , c+β * v'ϕ) for w in w_vals] #logical statement -> returns 0 or 1
end

function get_policy(v,model) # this will be used after finding the fixed point of T
    (; n, w_vals, ϕ, β, c) = model # unpack the model parameters
    return σ = [w/(1-β) >= c+β * v'ϕ for w in w_vals]
end

function vfi(model;maxiter=1000,tol=1e-8) # value function iteration
    (; n, w_vals, ϕ, β, c) = model
    v_init = w_vals/(1-β); error = tol + 1.0; iter = 1 #  initialize
    v = v_init
    v_history = [v_init]
    while error > tol && iter < maxiter
        v_new = T(v,model)
        error = maximum(abs.(v_new - v)) #difference between 2 consecutive v
        push!(v_history,v_new)
        v = v_new
        iter += 1
    end
    σ = get_policy(v, model)

    return v, σ, iter, error, v_history #value function, policy function [...]
end

my_model = create_job_search_model()
v, σ, iter, error, v_history = vfi(my_model)
plot_v = plot(my_model.w_vals,v, label="v(w)",linewidth=4,xlabel = "w",ylabel = "v")
plot_σ = plot(my_model.w_vals,σ, label="policy: 1 = accept wage",xlabel = "w", linestyle=:dash,linewidth=2)
plot(plot_v,plot_σ,layout=(1,2),legend=:topleft)

anim = @animate for i in 1:length(v_history)
    plot(my_model.w_vals, v_history[i], label="iter = $i", linewidth=4, xlabel="w", ylabel="v",ylim=(1-my_model.β)^(-1).*[minimum(my_model.w_vals), maximum(my_model.w_vals)])
end

gif(anim, "v_history.gif", fps = 5)

iter #24, with Beta very close to 1 it might not converge with the maxiter = 1000 (then you would need more)