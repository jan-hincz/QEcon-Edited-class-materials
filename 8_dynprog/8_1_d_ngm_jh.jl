## Neoclassical growth model - slides 19 - 24
using  Plots, Parameters

#TAKEAWAYS
#findfirst() - line ca. 100
#at the end - what is wrong? - ask Piotr and Marcin!!! (line ca. 133)

@with_kw struct NGMProblem #@with_kw: macro of Parameters package

    β = 0.95 # discount factor
    α = 0.3 # production function parameter - slide 20
    δ = 0.1 # depreciation rate
    γ = 2.0 # intertemporal elasticity of substitution (inverse) - slide 20

    f = x -> x^α # production function
    u = γ == 1 ? c -> log(c) : c -> c ^ (1-γ) / (1-γ)   # CRRA utility function
    k_star = ((β^(-1) - 1 + δ) / α) ^(1/(α-1)) # steady state capital (slide 23)

    k_min = 0.75 * k_star # minimum capital
    k_max = 1.25 * k_star # maximum capital
    
    n = 100 # number of grid points
    k_grid = range(k_min,stop=k_max,length=n) #n-element grid for capital from k_min to k_max
    
end


### NGM PROBLEM

function T(v,model) # Bellman operator
    @unpack n, k_grid, β, α, δ, f, u = model #Parameters package -> unpack -> it will take
    #arguments named this way from model of choice

    #initialise the function
    v_new = zeros(n) #v(k) as a vector, each entry will be for a given k 
    reward = zeros(n,n) #matrix n by n of lifetime utility for all possible k and k'
    #max lifetime utility wrt k' for a given k = v(k)
    σ = zeros(n)

        for (k_index,k) in enumerate(k_grid) # loop over capital today
            for (k_next_index, k_next) in enumerate(k_grid) # loop over capital tomorrow

                c = k^α - k_next + (1-δ)*k # consumption (slide 20)
                if c > 0
                    reward[k_index,k_next_index] = u(c) + β * v[k_next_index] #v(k) entry; (slide 21)
                else
                    reward[k_index,k_next_index] = -Inf #penalty instead of constrained optimisation with c>0
                end

            end 

            v_new[k_index], k_next_index_opt = findmax(reward[k_index,:]) 
            # k_index -> for each k, find the maximum reward = v(k) and the optimal next level of capital k' (with k_next_index_opt)
            σ[k_index] = k_grid[k_next_index_opt] # store the optimal policy σ(k) = k'
        end
        
    return v_new, σ
end


function vfi(model;maxiter=1000,tol=1e-8) # value function iteration
    @unpack n, k_grid, β, α, δ, f, u = model
    v_init = zeros(n); err = tol + 1.0; iter = 1 #initial guess
    v = v_init
    v_history = [v_init]
    σ = zeros(n)
    while err > tol && iter < maxiter
        v_new, σ = T(v,model)
        err = maximum(abs.(v_new - v)) 
        push!(v_history,v_new)
        v = v_new
        iter += 1
    end


    return v, σ, iter, err, v_history
end


my_ngm = NGMProblem(n=200) #defined at ca. line 4
v, σ, iter, err, v_history = vfi(my_ngm)
v #v(k) for each of 200 possible k

plot_v = plot(my_ngm.k_grid,v, label="v(k)",linewidth=4,xlabel = "k",ylabel = "v");
plot_σ = plot(my_ngm.k_grid,σ, label="policy: k'(k)", linewidth=4,xlabel = "k",);

# add the 45 degree line
plot!(my_ngm.k_grid,my_ngm.k_grid, label="45 degree line",linewidth=2,linestyle=:dash);

# add the steady state
vline!([my_ngm.k_star], label="steady state",linewidth=2,linestyle=:dash);
plot(plot_v,plot_σ,layout=(1,2),legend=:topleft)

# obtain a sample path for the capital stock

Time = 100
k_path = zeros(Time)
k_path[1] = my_ngm.k_grid[1] # start at the lowest level of capital 1.969 = 0.75 * k_star

for i in 2:Time
    k_path[i] = σ[findfirst(x->x==k_path[i-1],my_ngm.k_grid)] #for an i-1th k it finds ith (next) k based on σ(k) = k' function 
    #in Julia you cannot call σ(k) = σ(k_path[i-1]),
    #you have to do σ[index of k] and that's what findfirst() does here
end

plot_k_path = plot(1:Time,k_path, label="k(t)",linewidth=4,xlabel = "t",ylabel = "k")

# compare the speed of convergence for two different elasticities of substitution

my_ngm_low_γ = NGMProblem(γ=0.5,n=300) #defined at line ca. 7
v_low_γ, σ_low_γ, iter_low_γ, err_low_γ, v_history_low_γ = vfi(my_ngm_low_γ) #defined at line ca. 60

my_ngm_high_γ = NGMProblem(γ=5.0,n=300)
v_high_γ, σ_high_γ, iter_high_γ, err_high_γ, v_history_high_γ = vfi(my_ngm_high_γ)

Time = 100
k_path_low_γ = zeros(Time)
k_path_high_γ = zeros(Time)
k_path_low_γ[1] = my_ngm_low_γ.k_grid[1] # start at the lowest level of capital
k_path_high_γ[1] = my_ngm_high_γ.k_grid[1] # start at the lowest level of capital

for i in 2:Time
    k_path_low_γ[i] = σ_low_γ[findfirst(x->x==k_path_low_γ[i-1],my_ngm_low_γ.k_grid)]
    k_path_high_γ[i] = σ_high_γ[findfirst(x->x==k_path_high_γ[i-1],my_ngm_high_γ.k_grid)]
end


plot_k_convergence = plot(1:Time,k_path_low_γ, label="γ = 0.5",linewidth=4,xlabel = "t",ylabel = "k",legend=:topleft);
plot!(1:Time,k_path_high_γ, label="γ = 5.0",linewidth=4,xlabel = "t",ylabel = "k",legend=:topleft)
#quicker convergence for lower γ - less concave function - lower consumption smoothing motive
#-> will get quicker to steady state of c and k by in the meantime saving more and sacrificing c

# what is wrong here? 
plot(σ_high_γ - my_ngm_high_γ.k_grid) #IMO: k' - k, it should be 0 for k_star = 2.625 and it's 0 for k +-= 130
plot(σ_low_γ - my_ngm_low_γ.k_grid)

#Chat GPT suggests
#The discrepancy likely arises from grid resolution or numerical precision issues. 
#Adjusting the grid size or refining the numerical methods can help achieve a more accurate solution,
#aligning the policy function's steady state with the theoretical steady state capital k*